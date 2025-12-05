# insight_engine.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

from features import load_raw, build_features_for_business
from explainer import score_with_shap
from llm import generate_insights


def build_evidence_bundle_for_business(
    business_id: str, 
    ref_date: datetime, 
    lookback_days: int = 30
) -> Dict[str, Any]:
    """
    Build an evidence bundle containing metrics, SHAP importances, and complaint stats.
    
    Returns a dict with:
    - metrics: churn_rate, revenue, active_customers, etc.
    - shap_importances: feature importance scores
    - complaint_reasons: counts by reason label
    - period: start and end dates
    """
    businesses, customers, orders, interactions = load_raw()
    
    # Calculate period
    period_start = ref_date - timedelta(days=lookback_days)
    period_end = ref_date
    
    # Get business data
    orders_b = orders[orders["business_id"] == business_id].copy()
    orders_b["order_ts"] = pd.to_datetime(orders_b["order_ts"])
    
    # Filter orders in period
    mask_curr = (orders_b["order_ts"] >= period_start) & (orders_b["order_ts"] <= period_end)
    curr_orders = orders_b[mask_curr]
    
    # Calculate metrics
    revenue = float(curr_orders["revenue"].sum())
    orders_count = int(curr_orders["order_id"].nunique())
    active_customers = int(curr_orders["customer_id"].nunique())
    
    # Get features and scores
    df_feat, _ = build_features_for_business(business_id, ref_date)
    scored = score_with_shap(df_feat)
    churn_rate = float(scored["is_churned"].mean())
    
    # Calculate SHAP importances (global)
    shap_importances = {}
    for feats in scored["top_shap_features"]:
        for f in feats:
            feature_name = f["feature"]
            shap_value = f["shap_value"]
            if feature_name not in shap_importances:
                shap_importances[feature_name] = 0.0
            shap_importances[feature_name] += abs(shap_value)
    
    # Normalize SHAP importances
    total_importance = sum(shap_importances.values())
    if total_importance > 0:
        shap_importances = {k: v / total_importance for k, v in shap_importances.items()}
    
    # Get complaint statistics
    ints_b = interactions[interactions["business_id"] == business_id].copy()
    ints_b["ts"] = pd.to_datetime(ints_b["ts"])
    ints_curr = ints_b[(ints_b["ts"] >= period_start) & (ints_b["ts"] <= period_end)]
    
    complaint_reasons = {}
    if not ints_curr.empty and "reason_label" in ints_curr.columns:
        reason_counts = ints_curr["reason_label"].value_counts().to_dict()
        complaint_reasons = {str(k): int(v) for k, v in reason_counts.items()}
    
    bundle = {
        "period": {
            "start": period_start.isoformat(),
            "end": period_end.isoformat(),
            "lookback_days": lookback_days
        },
        "metrics": {
            "churn_rate": churn_rate,
            "revenue": revenue,
            "orders_count": orders_count,
            "active_customers": active_customers
        },
        "shap_importances": shap_importances,
        "complaint_reasons": complaint_reasons
    }
    
    return bundle


def generate_insights_for_business(
    business_id: str, 
    ref_date: datetime,
    lookback_days: int = 30
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate insights for a business using LLM.
    
    Returns:
        (bundle, insights) tuple where:
        - bundle: evidence bundle dict
        - insights: LLM-generated insights dict
    """
    bundle = build_evidence_bundle_for_business(business_id, ref_date, lookback_days)
    insights = generate_insights(bundle)
    
    return bundle, insights

