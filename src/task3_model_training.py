# task3_model_training.py

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
import xgboost as xgb
import os

DATA_PATH = "../data"
MODEL_PATH = "../models"

def load_data(file):
    """Load data from a CSV file."""
    return pd.read_csv(os.path.join(DATA_PATH, file))

def preprocess_and_split(df, target_column):
    """Preprocess the data and split into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC-PR:", average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))
    print(classification_report(y_test, y_pred))

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(class_weight='balanced', max_iter=500)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train an XGBoost model without tuning."""
    xgb_clf = xgb.XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def tune_xgboost(X_train, y_train):
    """Tune XGBoost hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [5, 10]
    }
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best XGBoost Params:", grid.best_params_)
    return grid.best_estimator_

def save_model(model, scaler, name_prefix):
    """Save the model and scaler to disk."""
    joblib.dump(model, os.path.join(MODEL_PATH, f"{name_prefix}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, f"{name_prefix}_scaler.pkl"))

def process_pipeline(dataset_file, target_column, prefix):
    """Process the data pipeline for model training and evaluation."""
    print(f"\n🔄 Processing {dataset_file}...")
    df = load_data(dataset_file)
    X_train, X_test, y_train, y_test = preprocess_and_split(df, target_column)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Logistic Regression
    print("\n📊 Logistic Regression:")
    logreg_model = train_logistic_regression(X_train_scaled, y_train)
    evaluate_model(logreg_model, X_test_scaled, y_test)
    save_model(logreg_model, scaler, f"logreg_{prefix}")

    # XGBoost
    print("\n🚀 XGBoost:")
    xgb_model = train_xgboost(X_train_scaled, y_train)
    evaluate_model(xgb_model, X_test_scaled, y_test)
    save_model(xgb_model, scaler, f"xgb_{prefix}")

    # Hyperparameter tuning (optional)
    print("\n🔍 Hyperparameter Tuning (XGBoost):")
    tuned_model = tune_xgboost(X_train_scaled, y_train)
    evaluate_model(tuned_model, X_test_scaled, y_test)
    joblib.dump(tuned_model, os.path.join(MODEL_PATH, f"xgb_tuned_{prefix}.pkl"))

if __name__ == "__main__":
    process_pipeline("fraud_featurized.csv", "is_fraud", "fraud")
    process_pipeline("creditcard.csv", "Class", "credit")