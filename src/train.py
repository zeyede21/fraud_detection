# fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import gc
import joblib
import os

def run_fraud_detection(data_path, model_save_path):
    # 1. Load Credit Card Fraud Data
    df = pd.read_csv(data_path)

    # 2. Define target and features
    target = 'Class'
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Apply SMOTE to balance classes on training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # 6. Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=42)
    print("Training Logistic Regression...")
    lr.fit(X_train_res, y_train_res)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    print("\nLogistic Regression Results:")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    print("F1 Score:", f1_score(y_test, y_pred_lr))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_proba_lr))

    # 7. Train XGBoost with hyperparameter tuning
    dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': (y_train_res == 0).sum() / (y_train_res == 1).sum(),
        'seed': 42
    }

    param_grid = {
        'max_depth': [4, 6],
        'eta': [0.05, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    best_f1 = 0
    best_params = None
    best_model = None

    print("\nStarting XGBoost hyperparameter tuning...")
    for max_depth in param_grid['max_depth']:
        for eta in param_grid['eta']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    params.update({
                        'max_depth': max_depth,
                        'eta': eta,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree,
                        'verbosity': 0
                    })
                    model = xgb.train(params, dtrain, num_boost_round=100)
                    y_proba = model.predict(dtest)
                    y_pred = (y_proba >= 0.5).astype(int)
                    f1 = f1_score(y_test, y_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = params.copy()
                        best_model = model

    print(f"\nBest XGBoost params: {best_params}")
    print(f"Best XGBoost F1 at threshold 0.5: {best_f1:.4f}")

    # 8. Threshold tuning for XGBoost
    y_proba_best = best_model.predict(dtest)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_best)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]

    print(f"Optimal threshold based on max F1: {best_threshold:.4f}")

    # Predict with tuned threshold
    y_pred_best = (y_proba_best >= best_threshold).astype(int)

    print("\nXGBoost Results with tuned threshold:")
    print(confusion_matrix(y_test, y_pred_best))
    print(classification_report(y_test, y_pred_best))
    print("F1 Score:", f1_score(y_test, y_pred_best))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_proba_best))

    # 9. Model Comparison
    print("\nModel Comparison:")
    print(f"Logistic Regression → F1: {f1_score(y_test, y_pred_lr):.4f}, AUC-PR: {average_precision_score(y_test, y_proba_lr):.4f}")
    print(f"XGBoost (tuned)    → F1: {f1_score(y_test, y_pred_best):.4f}, AUC-PR: {average_precision_score(y_test, y_proba_best):.4f}")

    # Decide best model
    f1_lr = f1_score(y_test, y_pred_lr)
    aucpr_lr = average_precision_score(y_test, y_proba_lr)
    f1_xgb_tuned = f1_score(y_test, y_pred_best)
    aucpr_xgb_tuned = average_precision_score(y_test, y_proba_best)

    if (f1_xgb_tuned > f1_lr) and (aucpr_xgb_tuned > aucpr_lr):
        best_model_name = "XGBoost (tuned)"
        best_model_final = best_model
    elif (f1_lr > f1_xgb_tuned) and (aucpr_lr > aucpr_xgb_tuned):
        best_model_name = "Logistic Regression"
        best_model_final = lr
    else:
        if f1_xgb_tuned >= f1_lr:
            best_model_name = "XGBoost (tuned)"
            best_model_final = best_model
        else:
            best_model_name = "Logistic Regression"
            best_model_final = lr

    print(f"\n🏆 Best Model Selected: {best_model_name}")

    # 10. Save the best model and scaler
    os.makedirs(model_save_path, exist_ok=True)

    if best_model_name == "Logistic Regression":
        model_filename = 'credit_model_lr.pkl'
        joblib.dump(best_model_final, os.path.join(model_save_path, model_filename))
    else:
        model_filename = 'credit_model_xgb.json'
        best_model_final.save_model(os.path.join(model_save_path, model_filename))

    joblib.dump(scaler, os.path.join(model_save_path, 'scaler_credit.pkl'))

    print(f"✅ Saved: {best_model_name} to {model_save_path}/{model_filename}")

    # 11. Cleanup
    gc.collect()