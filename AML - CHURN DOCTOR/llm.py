# llm.py - Free LLM implementation using Hugging Face Transformers

import json
from typing import Dict, Any
from transformers import pipeline
import torch

REASON_LABELS = [
    "PRICE_SENSITIVITY",
    "PRODUCT_QUALITY",
    "SHIPPING_DELAY",
    "CUSTOMER_SERVICE",
    "LACK_OF_STOCK",
    "USER_EXPERIENCE",
    "COMPETITOR_SWITCH",
    "OTHER",
]

# Initialize models (lazy loading)
_sentiment_classifier = None
_zero_shot_classifier = None

def _get_sentiment_classifier():
    """Lazy load sentiment classifier"""
    global _sentiment_classifier
    if _sentiment_classifier is None:
        _sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
    return _sentiment_classifier

def _get_zero_shot_classifier():
    """Lazy load zero-shot classifier"""
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        _zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
    return _zero_shot_classifier

def _classify_reason_keywords(text: str) -> str:
    """Fallback keyword-based classification"""
    text_lower = text.lower()
    
    keywords = {
        "PRICE_SENSITIVITY": ["price", "expensive", "cost", "cheaper", "pricing", "overpriced", "afford", "budget"],
        "PRODUCT_QUALITY": ["quality", "defective", "broken", "damaged", "poor quality", "low quality", "faulty"],
        "SHIPPING_DELAY": ["shipping", "delivery", "late", "delay", "slow", "arrived", "shipment", "delayed"],
        "CUSTOMER_SERVICE": ["service", "support", "help", "representative", "agent", "response", "unhelpful"],
        "LACK_OF_STOCK": ["out of stock", "unavailable", "sold out", "backorder", "stock", "inventory"],
        "USER_EXPERIENCE": ["website", "app", "interface", "difficult", "confusing", "hard to use", "navigation"],
        "COMPETITOR_SWITCH": ["competitor", "switching", "better", "alternative", "other company"],
    }
    
    scores = {}
    for label, words in keywords.items():
        score = sum(1 for word in words if word in text_lower)
        scores[label] = score
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "OTHER"

def classify_complaint(text: str) -> Dict[str, Any]:
    """
    Classify a complaint into a single reason label + sentiment.

    Returns a dict like:
    {
        "reason_label": "SHIPPING_DELAY",
        "sentiment": "negative",
        "confidence": 0.87
    }
    """
    # Classify reason using zero-shot classification
    try:
        zero_shot = _get_zero_shot_classifier()
        reason_result = zero_shot(text, REASON_LABELS)
        reason_label = reason_result["labels"][0]
        reason_confidence = reason_result["scores"][0]
    except Exception as e:
        # Fallback to keyword-based classification
        print(f"Zero-shot classification failed: {e}, using keyword-based fallback")
        reason_label = _classify_reason_keywords(text)
        reason_confidence = 0.7
    
    # Classify sentiment
    try:
        sentiment_classifier = _get_sentiment_classifier()
        sentiment_result = sentiment_classifier(text)[0]
        sentiment_label = sentiment_result["label"].lower()
        sentiment_score = sentiment_result["score"]
        
        # Map sentiment labels to standard format
        if "positive" in sentiment_label:
            sentiment = "positive"
        elif "negative" in sentiment_label:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    except Exception as e:
        print(f"Sentiment classification failed: {e}, defaulting to negative")
        sentiment = "negative"
        sentiment_score = 0.8
    
    # Overall confidence is average of both
    confidence = (reason_confidence + sentiment_score) / 2
    
    return {
        "reason_label": reason_label,
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }


def generate_insights(evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn structured evidence (metrics, SHAP importances, complaint stats)
    into a structured insight JSON using template-based approach.

    Expected output format:
    {
      "headline": "...",
      "executive_summary": "...",
      "key_metrics": {...},
      "trends": {...},
      "top_reasons": [
        {
          "name": "...",
          "evidence": ["...", "..."],
          "impact_estimate": "Low" | "Medium" | "High"
        },
        ...
      ],
      "recommendations": ["...", "..."],
      "detailed_analysis": "..."
    }
    """
    # Extract data from evidence bundle
    metrics = evidence_bundle.get("metrics", {})
    shap_features = evidence_bundle.get("shap_importances", {})
    complaint_stats = evidence_bundle.get("complaint_reasons", {})
    customer_segments = evidence_bundle.get("customer_segments", {})
    
    churn_rate = metrics.get("churn_rate", 0)
    revenue = metrics.get("revenue", 0)
    active_customers = metrics.get("active_customers", 0)
    orders_count = metrics.get("orders_count", 0)
    avg_order_value = metrics.get("avg_order_value", 0)
    revenue_change = metrics.get("revenue_change_pct", 0)
    orders_change = metrics.get("orders_change_pct", 0)
    customers_change = metrics.get("customers_change_pct", 0)
    high_risk_customers = metrics.get("high_risk_customers", 0)
    medium_risk_customers = metrics.get("medium_risk_customers", 0)
    low_risk_customers = metrics.get("low_risk_customers", 0)
    revenue_at_risk_pct = metrics.get("revenue_at_risk_pct", 0)
    avg_recency = metrics.get("avg_recency_days", 0)
    avg_orders_90d = metrics.get("avg_orders_90d", 0)
    avg_revenue_90d = metrics.get("avg_revenue_90d", 0)
    avg_tenure = metrics.get("avg_tenure_days", 0)
    
    # Generate headline
    if churn_rate > 0.3:
        headline = f"High churn rate detected ({churn_rate:.1%}) - Immediate action required"
    elif churn_rate > 0.15:
        headline = f"Moderate churn rate ({churn_rate:.1%}) - Monitor closely"
    else:
        headline = f"Low churn rate ({churn_rate:.1%}) - Business is stable"
    
    # Generate executive summary
    summary_parts = []
    summary_parts.append(f"Current churn rate is {churn_rate:.1%} with {active_customers} active customers.")
    
    if revenue_change > 5:
        summary_parts.append(f"Revenue increased by {revenue_change:.1f}% compared to previous period - positive trend.")
    elif revenue_change < -5:
        summary_parts.append(f"Revenue decreased by {abs(revenue_change):.1f}% compared to previous period - concerning trend.")
    else:
        summary_parts.append(f"Revenue is relatively stable ({revenue_change:+.1f}% change).")
    
    if high_risk_customers > 0:
        summary_parts.append(f"{high_risk_customers} customers are at high risk of churning, representing {revenue_at_risk_pct:.1f}% of recent revenue.")
    
    executive_summary = " ".join(summary_parts)
    
    # Identify top reasons from SHAP features and complaints
    top_reasons = []
    
    # Process complaint reasons
    if complaint_stats:
        sorted_complaints = sorted(complaint_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        for reason, count in sorted_complaints:
            if count > 0:
                impact = "High" if count > 10 else "Medium" if count > 5 else "Low"
                top_reasons.append({
                    "name": reason.replace("_", " ").title(),
                    "evidence": [f"{count} complaints related to {reason.replace('_', ' ').lower()}"],
                    "impact_estimate": impact
                })
    
    # Process top SHAP features
    if shap_features:
        sorted_shap = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feature, importance in sorted_shap:
            if abs(importance) > 0.1:  # Only significant features
                impact = "High" if abs(importance) > 0.3 else "Medium" if abs(importance) > 0.15 else "Low"
                direction = "increasing" if importance > 0 else "decreasing"
                top_reasons.append({
                    "name": feature.replace("_", " ").title(),
                    "evidence": [f"Feature importance: {importance:.3f} ({direction} churn risk)"],
                    "impact_estimate": impact
                })
    
    # Limit to top 5 reasons
    top_reasons = top_reasons[:5]
    
    # Generate recommendations based on insights
    recommendations = []
    
    if churn_rate > 0.2:
        recommendations.append("Implement customer retention program with targeted offers and personalized outreach")
    
    if high_risk_customers > active_customers * 0.1:
        recommendations.append(f"Prioritize outreach to {high_risk_customers} high-risk customers with win-back campaigns")
    
    if avg_recency > 60:
        recommendations.append(f"Average customer recency is {avg_recency:.0f} days - implement re-engagement campaigns")
    
    if complaint_stats.get("SHIPPING_DELAY", 0) > 5:
        recommendations.append("Review and optimize shipping and delivery processes - shipping delays are a major concern")
    
    if complaint_stats.get("CUSTOMER_SERVICE", 0) > 5:
        recommendations.append("Improve customer service response time and quality - invest in training and support tools")
    
    if complaint_stats.get("PRICE_SENSITIVITY", 0) > 5:
        recommendations.append("Consider pricing strategy review and competitive analysis - price sensitivity is high")
    
    if revenue_change < -10:
        recommendations.append("Revenue decline detected - investigate root causes and implement revenue recovery initiatives")
    
    if avg_orders_90d < 1:
        recommendations.append("Low order frequency detected - implement programs to increase purchase frequency")
    
    if shap_features:
        top_feature = max(shap_features.items(), key=lambda x: abs(x[1]))
        feature_name = top_feature[0].replace("_", " ").title()
        recommendations.append(f"Focus on improving {feature_name} as it's the strongest churn driver (importance: {abs(top_feature[1]):.3f})")
    
    if not recommendations:
        recommendations.append("Continue monitoring key metrics and customer feedback")
        recommendations.append("Maintain current customer engagement strategies")
    
    # Generate detailed analysis
    analysis_parts = []
    analysis_parts.append(f"**Customer Risk Distribution:** {high_risk_customers} high-risk, {medium_risk_customers} medium-risk, and {low_risk_customers} low-risk customers.")
    analysis_parts.append(f"**Customer Behavior:** Average customer has {avg_orders_90d:.1f} orders in the last 90 days, with average revenue of ${avg_revenue_90d:.2f}.")
    analysis_parts.append(f"**Customer Engagement:** Average recency is {avg_recency:.0f} days, and average customer tenure is {avg_tenure:.0f} days.")
    
    if revenue_at_risk_pct > 20:
        analysis_parts.append(f"**Revenue Risk:** {revenue_at_risk_pct:.1f}% of recent revenue is at risk from high-risk customers - significant financial impact.")
    
    if customer_segments:
        segment_str = ", ".join([f"{k}: {v}" for k, v in customer_segments.items()])
        analysis_parts.append(f"**Customer Segments:** {segment_str}")
    
    detailed_analysis = "\n\n".join(analysis_parts)
    
    # Key metrics summary
    key_metrics = {
        "churn_rate": f"{churn_rate:.1%}",
        "revenue": f"${revenue:,.0f}",
        "avg_order_value": f"${avg_order_value:.2f}",
        "active_customers": f"{active_customers:,}",
        "high_risk_customers": f"{high_risk_customers}",
        "revenue_at_risk": f"{revenue_at_risk_pct:.1f}%"
    }
    
    # Trends
    trends = {
        "revenue_change": f"{revenue_change:+.1f}%",
        "orders_change": f"{orders_change:+.1f}%",
        "customers_change": f"{customers_change:+.1f}%"
    }
    
    return {
        "headline": headline,
        "executive_summary": executive_summary,
        "key_metrics": key_metrics,
        "trends": trends,
        "top_reasons": top_reasons,
        "recommendations": recommendations,
        "detailed_analysis": detailed_analysis
    }


def generate_email(owner_name: str, business_name: str, period_str: str, insights: Dict[str, Any]) -> str:
    """
    Generate an email body from insights.
    """
    body = f"""
Dear {owner_name},

Here is your Monthly churn analysis report for {business_name} covering the period: {period_str}

{insights.get('headline', 'Churn Analysis Summary')}

EXECUTIVE SUMMARY:
{insights.get('executive_summary', 'No summary available')}

KEY METRICS:
"""
    if insights.get('key_metrics'):
        for key, value in insights.get('key_metrics', {}).items():
            body += f"  • {key.replace('_', ' ').title()}: {value}\n"
    
    body += "\nTRENDS (vs Previous Period):\n"
    if insights.get('trends'):
        for key, value in insights.get('trends', {}).items():
            body += f"  • {key.replace('_', ' ').title()}: {value}\n"
    
    body += "\nDETAILED ANALYSIS:\n"
    if insights.get('detailed_analysis'):
        body += insights.get('detailed_analysis') + "\n"
    
    body += "\nTOP CHURN DRIVERS:\n"
    for i, reason in enumerate(insights.get('top_reasons', [])[:5], 1):
        body += f"\n{i}. {reason['name']} (Impact: {reason['impact_estimate']})\n"
        for evidence in reason.get('evidence', []):
            body += f"   - {evidence}\n"
    
    body += "\nRECOMMENDATIONS:\n"
    for i, rec in enumerate(insights.get('recommendations', []), 1):
        body += f"{i}. {rec}\n"
    
    body += "\nBest regards,\nChurn Doctor Analytics Team"
    
    return body
