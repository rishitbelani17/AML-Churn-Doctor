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
      "top_reasons": [
        {
          "name": "...",
          "evidence": ["...", "..."],
          "impact_estimate": "Low" | "Medium" | "High"
        },
        ...
      ],
      "recommendations": ["...", "..."]
    }
    """
    # Extract data from evidence bundle
    metrics = evidence_bundle.get("metrics", {})
    shap_features = evidence_bundle.get("shap_importances", {})
    complaint_stats = evidence_bundle.get("complaint_reasons", {})
    
    churn_rate = metrics.get("churn_rate", 0)
    revenue = metrics.get("revenue", 0)
    active_customers = metrics.get("active_customers", 0)
    
    # Generate headline
    if churn_rate > 0.3:
        headline = f"High churn rate detected ({churn_rate:.1%}) - Immediate action required"
    elif churn_rate > 0.15:
        headline = f"Moderate churn rate ({churn_rate:.1%}) - Monitor closely"
    else:
        headline = f"Low churn rate ({churn_rate:.1%}) - Business is stable"
    
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
        recommendations.append("Implement customer retention program with targeted offers")
    
    if complaint_stats.get("SHIPPING_DELAY", 0) > 5:
        recommendations.append("Review and optimize shipping and delivery processes")
    
    if complaint_stats.get("CUSTOMER_SERVICE", 0) > 5:
        recommendations.append("Improve customer service response time and quality")
    
    if complaint_stats.get("PRICE_SENSITIVITY", 0) > 5:
        recommendations.append("Consider pricing strategy review and competitive analysis")
    
    if shap_features:
        top_feature = max(shap_features.items(), key=lambda x: abs(x[1]))
        recommendations.append(f"Focus on improving {top_feature[0].replace('_', ' ')} as it's the strongest churn driver")
    
    if not recommendations:
        recommendations.append("Continue monitoring key metrics and customer feedback")
        recommendations.append("Maintain current customer engagement strategies")
    
    return {
        "headline": headline,
        "top_reasons": top_reasons,
        "recommendations": recommendations
    }


def generate_email(owner_name: str, business_name: str, period_str: str, insights: Dict[str, Any]) -> str:
    """
    Generate an email body from insights.
    """
    body = f"""
Dear {owner_name},

Here is your weekly churn analysis report for {business_name} covering the period: {period_str}

{insights.get('headline', 'Churn Analysis Summary')}

Top Churn Drivers:
"""
    for i, reason in enumerate(insights.get('top_reasons', [])[:5], 1):
        body += f"\n{i}. {reason['name']} (Impact: {reason['impact_estimate']})\n"
        for evidence in reason.get('evidence', []):
            body += f"   - {evidence}\n"
    
    body += "\nRecommendations:\n"
    for i, rec in enumerate(insights.get('recommendations', [])[:5], 1):
        body += f"{i}. {rec}\n"
    
    body += "\nBest regards,\nChurn Doctor Analytics Team"
    
    return body
