# fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import gc
from sklearn.impute import SimpleImputer
import os

def run_fraud_detection(data_path, model_dir):
    # 1. Load the processed fraud data
    df = pd.read_csv(data_path)

    # 2. Define target and features
    target = 'class'
    X = df.drop(columns=[target, 'user_id', 'signup_time', 'purchase_time'])
    y = df[target]

    # 3. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4. Identify categorical and numerical columns
    categorical_cols = ['device_id', 'source', 'browser', 'sex', 'country']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # 5. Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]), categorical_cols)
    ])

    # 6. Apply preprocessing to training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 7. Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

    # 8. Logistic Regression pipeline
    lr_pipeline = Pipeline([
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1))
    ])

    print("Training Logistic Regression...")
    lr_pipeline.fit(X_train_res, y_train_res)

    # Predict and evaluate Logistic Regression
    y_pred_lr = lr_pipeline.predict(X_test_processed)
    y_pred_proba_lr = lr_pipeline.predict_proba(X_test_processed)[:, 1]

    print("Logistic Regression Results:")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    print("F1 Score:", f1_score(y_test, y_pred_lr))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_pred_proba_lr))

    # 9. Prepare data for XGBoost
    dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
    dtest = xgb.DMatrix(X_test_processed, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': (y_train_res == 0).sum() / (y_train_res == 1).sum(),
        'max_depth': 6,
        'eta': 0.1,
        'verbosity': 0
    }

    print("Training XGBoost...")
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Predict and evaluate XGBoost
    y_pred_proba_xgb = xgb_model.predict(dtest)
    y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

    print("XGBoost Results:")
    print(confusion_matrix(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    print("F1 Score:", f1_score(y_test, y_pred_xgb))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_pred_proba_xgb))

    # 10. Summary model comparison
    print("\nModel Comparison:")
    print(f"Logistic Regression → F1: {f1_score(y_test, y_pred_lr):.4f}, AUC-PR: {average_precision_score(y_test, y_pred_proba_lr):.4f}")
    print(f"XGBoost             → F1: {f1_score(y_test, y_pred_xgb):.4f}, AUC-PR: {average_precision_score(y_test, y_pred_proba_xgb):.4f}")

    # 11. Cleanup
    gc.collect()

    # 12. Save models and preprocessor
    os.makedirs(model_dir, exist_ok=True)

    # Save the preprocessing pipeline
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))

    # Save the Logistic Regression pipeline
    joblib.dump(lr_pipeline, os.path.join(model_dir, "logistic_regression_pipeline.pkl"))

    # Save the XGBoost model
    xgb_model.save_model(os.path.join(model_dir, "xgboost_model.json"))

    print("✅ Models and preprocessing pipeline saved successfully.")