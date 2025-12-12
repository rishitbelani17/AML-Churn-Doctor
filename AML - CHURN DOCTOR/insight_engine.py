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
    
    # Calculate basic metrics
    revenue = float(curr_orders["revenue"].sum())
    orders_count = int(curr_orders["order_id"].nunique())
    active_customers = int(curr_orders["customer_id"].nunique())
    avg_order_value = float(revenue / orders_count) if orders_count > 0 else 0.0
    
    # Calculate previous period for comparison
    period_prev_start = period_start - timedelta(days=lookback_days)
    mask_prev = (orders_b["order_ts"] >= period_prev_start) & (orders_b["order_ts"] < period_start)
    prev_orders = orders_b[mask_prev]
    prev_revenue = float(prev_orders["revenue"].sum())
    prev_orders_count = int(prev_orders["order_id"].nunique())
    prev_active_customers = int(prev_orders["customer_id"].nunique())
    
    # Revenue and customer trends
    revenue_change = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0.0
    orders_change = ((orders_count - prev_orders_count) / prev_orders_count * 100) if prev_orders_count > 0 else 0.0
    customers_change = ((active_customers - prev_active_customers) / prev_active_customers * 100) if prev_active_customers > 0 else 0.0
    
    # Get features and scores
    df_feat, _ = build_features_for_business(business_id, ref_date)
    scored = score_with_shap(df_feat)
    churn_rate = float(scored["is_churned"].mean())
    
    # Customer segment analysis
    high_risk_customers = int((scored["churn_score"] > 0.7).sum())
    medium_risk_customers = int(((scored["churn_score"] > 0.4) & (scored["churn_score"] <= 0.7)).sum())
    low_risk_customers = int((scored["churn_score"] <= 0.4).sum())
    
    # Revenue at risk (from high-risk customers)
    high_risk_revenue = float(scored[scored["churn_score"] > 0.7]["revenue_90d"].sum())
    total_revenue_90d = float(scored["revenue_90d"].sum())
    revenue_at_risk_pct = (high_risk_revenue / total_revenue_90d * 100) if total_revenue_90d > 0 else 0.0
    
    # Average metrics for customers
    avg_recency = float(scored["recency_days"].mean())
    avg_orders_90d = float(scored["orders_90d"].mean())
    avg_revenue_90d = float(scored["revenue_90d"].mean())
    avg_tenure = float(scored["tenure_days"].mean())
    
    # Customer segment distribution
    if "segment" in customers.columns:
        segment_counts = customers[customers["business_id"] == business_id]["segment"].value_counts().to_dict()
    else:
        segment_counts = {}
    
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
            "active_customers": active_customers,
            "avg_order_value": avg_order_value,
            "revenue_change_pct": revenue_change,
            "orders_change_pct": orders_change,
            "customers_change_pct": customers_change,
            "high_risk_customers": high_risk_customers,
            "medium_risk_customers": medium_risk_customers,
            "low_risk_customers": low_risk_customers,
            "revenue_at_risk_pct": revenue_at_risk_pct,
            "avg_recency_days": avg_recency,
            "avg_orders_90d": avg_orders_90d,
            "avg_revenue_90d": avg_revenue_90d,
            "avg_tenure_days": avg_tenure,
            "prev_revenue": prev_revenue,
            "prev_orders_count": prev_orders_count,
            "prev_active_customers": prev_active_customers
        },
        "customer_segments": {str(k): int(v) for k, v in segment_counts.items()},
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

