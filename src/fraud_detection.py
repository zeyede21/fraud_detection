# fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

def run_fraud_detection(data_path):
    # 1. Load dataset
    df = pd.read_csv(data_path)

    # 2. Basic EDA - Check class imbalance
    print("Fraud Class Distribution:\n", df['class'].value_counts(normalize=True))

    # 3. Feature selection
    drop_cols = ['user_id', 'signup_time', 'purchase_time']
    target_col = 'class'

    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col]

    # 4. Identify categorical and numerical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 5. Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols)
    ])

    # 6. Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 7. Build pipelines for Logistic Regression
    logreg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])

    # 8. Train Logistic Regression
    print("\nTraining Logistic Regression...")
    logreg_pipeline.fit(X_train, y_train)

    # Predict and evaluate Logistic Regression
    y_pred_logreg = logreg_pipeline.predict(X_test)
    y_proba_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]

    print("\n📊 Logistic Regression Results:")
    print(confusion_matrix(y_test, y_pred_logreg))
    print(classification_report(y_test, y_pred_logreg))
    print("F1 Score:", f1_score(y_test, y_pred_logreg))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_proba_logreg))

    # 9. Prepare data for XGBoost
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dtest = xgb.DMatrix(X_test_processed, label=y_test)

    # 10. Define and train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }

    print("\nTraining XGBoost...")
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Predict and evaluate XGBoost
    y_pred_proba_xgb = xgb_model.predict(dtest)
    y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

    print("\n📊 XGBoost Results:")
    print(confusion_matrix(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    print("F1 Score:", f1_score(y_test, y_pred_xgb))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_pred_proba_xgb))

    # 11. Model comparison
    print("\n📌 Model Comparison:")
    print(f"Logistic Regression → F1: {f1_score(y_test, y_pred_logreg):.4f}, AUC-PR: {average_precision_score(y_test, y_proba_logreg):.4f}")
    print(f"XGBoost             → F1: {f1_score(y_test, y_pred_xgb):.4f}, AUC-PR: {average_precision_score(y_test, y_pred_proba_xgb):.4f}")

    if f1_score(y_test, y_pred_xgb) > f1_score(y_test, y_pred_logreg):
        print("\n🏆 XGBoost outperforms Logistic Regression on F1 and AUC-PR metrics.")
    else:
        print("\n🏆 Logistic Regression outperforms XGBoost on F1 and AUC-PR metrics.")