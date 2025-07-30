# fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
import xgboost as xgb

def run_fraud_detection(data_path):
    # 1. Load dataset
    df = pd.read_csv(data_path)

    # 2. Check class distribution
    print("Class distribution:\n", df['Class'].value_counts(normalize=True))

    # 3. Features and labels
    X = df.drop(columns=['Class'])
    y = df['Class']

    # 4. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 5. Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Logistic Regression
    print("\n🔹 Logistic Regression Results:")
    logreg = LogisticRegression(class_weight='balanced', max_iter=500)
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    y_proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    print("F1 Score:", f1_score(y_test, y_pred_lr))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_proba_lr))

    # 7. XGBoost
    print("\n🔹 XGBoost Results:")
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'max_depth': 4,
        'eta': 0.1,
        'seed': 42
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    y_proba_xgb = xgb_model.predict(dtest)
    y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)

    print(confusion_matrix(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    print("F1 Score:", f1_score(y_test, y_pred_xgb))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_proba_xgb))

    # 8. Model Comparison
    print("\n📌 Credit Card Dataset - Model Comparison:")
    print(f"Logistic Regression → F1: {f1_score(y_test, y_pred_lr):.4f}, AUC-PR: {average_precision_score(y_test, y_proba_lr):.4f}")
    print(f"XGBoost             → F1: {f1_score(y_test, y_pred_xgb):.4f}, AUC-PR: {average_precision_score(y_test, y_proba_xgb):.4f}")

    # 9. Model comparison & justification
    f1_lr = f1_score(y_test, y_pred_lr)
    aucpr_lr = average_precision_score(y_test, y_proba_lr)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    aucpr_xgb = average_precision_score(y_test, y_proba_xgb)

    if (f1_xgb > f1_lr) and (aucpr_xgb > aucpr_lr):
        print("\n🏆 XGBoost outperforms Logistic Regression on both F1 and AUC-PR metrics.")
    elif (f1_lr > f1_xgb) and (aucpr_lr > aucpr_xgb):
        print("\n🏆 Logistic Regression outperforms XGBoost on both F1 and AUC-PR metrics.")
    else:
        print("\n⚠️ Models have mixed results:")
        if f1_xgb > f1_lr:
            print("- XGBoost has a better F1 score.")
        else:
            print("- Logistic Regression has a better F1 score.")
        if aucpr_xgb > aucpr_lr:
            print("- XGBoost has a better AUC-PR.")
        else:
            print("- Logistic Regression has a better AUC-PR.")
        print("Choose based on your priority metric or consider further evaluation.")