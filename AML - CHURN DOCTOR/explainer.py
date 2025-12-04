# shap_explainer.py
import joblib
import shap
import numpy as np
import pandas as pd
from config import MODEL_DIR

_shap_explainer = None
_model_meta = None

def load_model_and_explainer():
    global _shap_explainer, _model_meta
    if _model_meta is None:
        _model_meta = joblib.load(f"{MODEL_DIR}/churn_model.pkl")
        model = _model_meta["model"]
        _shap_explainer = shap.TreeExplainer(model)
    return _model_meta, _shap_explainer

def score_with_shap(df_features: pd.DataFrame) -> pd.DataFrame:
    model_meta, explainer = load_model_and_explainer()
    model = model_meta["model"]
    feature_cols = model_meta["feature_cols"]
    X = df_features[feature_cols]

    proba = model.predict_proba(X)[:, 1]
    shap_values = explainer.shap_values(X)

    df_out = df_features.copy()
    df_out["churn_score"] = proba

    # store top shap features per row as JSON-like dict in a column
    top_features = []
    for i in range(X.shape[0]):
        row_vals = shap_values[i]
        abs_vals = np.abs(row_vals)
        top_idx = np.argsort(-abs_vals)[:5]
        feats = []
        for idx in top_idx:
            feats.append({"feature": feature_cols[idx],
                          "shap_value": float(row_vals[idx])})
        top_features.append(feats)

    df_out["top_shap_features"] = top_features
    return df_out
