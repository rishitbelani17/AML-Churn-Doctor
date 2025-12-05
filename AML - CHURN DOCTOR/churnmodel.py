# churn_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier  # pip install xgboost
from config import MODEL_DIR
from features import build_features_all_businesses

def train_churn_model():
    Xy, _ = build_features_all_businesses()
    y = Xy["is_churned"].astype(int)
    # simple feature selection - drop non-numeric columns
    drop_cols = ["customer_id", "business_id", "is_churned"]
    X = Xy.drop(columns=drop_cols)
    
    # Ensure all columns are numeric (convert any remaining object types)
    for col in X.columns:
        if X[col].dtype == 'object':
            X = X.drop(columns=[col])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Validation AUC: {auc:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": list(X.columns)}, f"{MODEL_DIR}/churn_model.pkl")
    print("Model saved")

if __name__ == "__main__":
    train_churn_model()
