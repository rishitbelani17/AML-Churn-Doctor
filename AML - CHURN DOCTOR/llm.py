# llm_client.py

import os
import json
from typing import Dict, Any

from openai import OpenAI  # pip install openai
from config import LLM_MODEL_NAME

# Read API key from env var for safety
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. "
        "Export it before running the app."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

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
    prompt = f"""
You are a classifier for customer complaints.

Possible reason labels: {REASON_LABELS}.

Given this complaint text:
\"\"\"{text}\"\"\"

Return JSON with keys:
- reason_label: one of {REASON_LABELS}
- sentiment: "negative", "neutral", or "positive"
- confidence: a float between 0 and 1 (your confidence)
"""
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    return json.loads(content)


def generate_insights(evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn structured evidence (metrics, SHAP importances, complaint stats)
    into a structured insight JSON.

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
    prompt = f"""
You are an analytics assistant for small businesses.

You are given structured data describing churn metrics, model feature importances
(from SHAP), and complaint reason statistics for a given business and time period.

You MUST:
- Use ONLY the numbers and signals present in the input JSON.
- NOT invent any new metrics or numbers.

Input JSON:
```json
{json.dumps(evidence_bundle)}
