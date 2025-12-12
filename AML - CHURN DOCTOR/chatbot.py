# chatbot.py - Business-specific chatbot using free Hugging Face models
from typing import Dict, Any, List
from transformers import pipeline
import torch

# Initialize chatbot (lazy loading)
_chatbot_pipeline = None

def _get_chatbot():
    """Lazy load text generation pipeline for chatbot"""
    global _chatbot_pipeline
    if _chatbot_pipeline is None:
        # Try to use a text-to-text model for better question answering
        try:
            # Using flan-t5-base which is good for instruction following
            _chatbot_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
            )
        except Exception as e:
            print(f"Failed to load flan-t5-base: {e}")
            _chatbot_pipeline = None
    return _chatbot_pipeline

def format_business_context(business_name: str, bundle: Dict[str, Any], insights: Dict[str, Any]) -> str:
    """
    Format business data into a context string for the chatbot.
    """
    metrics = bundle.get("metrics", {})
    trends = insights.get("trends", {})
    key_metrics = insights.get("key_metrics", {})
    
    context = f"""
Business Information for {business_name}:

KEY METRICS:
- Churn Rate: {metrics.get('churn_rate', 0):.1%}
- Revenue: ${metrics.get('revenue', 0):,.0f}
- Active Customers: {metrics.get('active_customers', 0):,}
- Orders: {metrics.get('orders_count', 0):,}
- Average Order Value: ${metrics.get('avg_order_value', 0):.2f}
- High Risk Customers: {metrics.get('high_risk_customers', 0)}
- Medium Risk Customers: {metrics.get('medium_risk_customers', 0)}
- Low Risk Customers: {metrics.get('low_risk_customers', 0)}
- Revenue at Risk: {metrics.get('revenue_at_risk_pct', 0):.1f}%

TRENDS (vs Previous Period):
- Revenue Change: {trends.get('revenue_change', 'N/A')}
- Orders Change: {trends.get('orders_change', 'N/A')}
- Customers Change: {trends.get('customers_change', 'N/A')}

CUSTOMER BEHAVIOR:
- Average Recency: {metrics.get('avg_recency_days', 0):.0f} days
- Average Orders (90d): {metrics.get('avg_orders_90d', 0):.1f}
- Average Revenue (90d): ${metrics.get('avg_revenue_90d', 0):.2f}
- Average Tenure: {metrics.get('avg_tenure_days', 0):.0f} days

TOP CHURN DRIVERS:
"""
    for i, reason in enumerate(insights.get('top_reasons', [])[:3], 1):
        context += f"{i}. {reason['name']} (Impact: {reason['impact_estimate']})\n"
        for evidence in reason.get('evidence', [])[:1]:
            context += f"   - {evidence}\n"
    
    complaint_reasons = bundle.get("complaint_reasons", {})
    if complaint_reasons:
        context += "\nCOMPLAINT STATISTICS:\n"
        for reason, count in list(complaint_reasons.items())[:5]:
            context += f"- {reason.replace('_', ' ').title()}: {count} complaints\n"
    
    context += f"\nEXECUTIVE SUMMARY:\n{insights.get('executive_summary', 'N/A')}\n"
    
    return context

def answer_question(question: str, business_name: str, bundle: Dict[str, Any], insights: Dict[str, Any]) -> str:
    """
    Answer a question about the business using the provided context.
    
    Args:
        question: User's question
        business_name: Name of the business
        bundle: Evidence bundle with metrics
        insights: Generated insights
    
    Returns:
        Answer string
    """
    # Format context
    context = format_business_context(business_name, bundle, insights)
    
    # Create prompt for the model
    prompt = f"""Answer the question based on the business data provided. Only use information from the context.

Context: {context}

Question: {question}

Answer:"""
    
    try:
        chatbot = _get_chatbot()
        if chatbot is None:
            # Fallback to rule-based responses (this is the primary method)
            return _rule_based_answer(question, business_name, bundle, insights)
        
        # Generate answer using text2text generation
        result = chatbot(prompt, max_length=200, do_sample=False, num_beams=2)
        
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("generated_text", "").strip()
            # Clean up the answer
            if answer:
                # Remove any repeated context
                if "Context:" in answer:
                    answer = answer.split("Context:")[0].strip()
                if "Question:" in answer:
                    answer = answer.split("Question:")[-1].strip()
                if answer and len(answer) > 10:  # Only use if we got a meaningful answer
                    return answer
        
        # Fallback if LLM didn't produce good answer
        return _rule_based_answer(question, business_name, bundle, insights)
    
    except Exception as e:
        print(f"Chatbot error: {e}")
        # Always fallback to rule-based for reliability
        return _rule_based_answer(question, business_name, bundle, insights)

def _rule_based_answer(question: str, business_name: str, bundle: Dict[str, Any], insights: Dict[str, Any]) -> str:
    """
    Fallback rule-based answer system when LLM is not available.
    """
    question_lower = question.lower()
    metrics = bundle.get("metrics", {})
    trends = insights.get("trends", {})
    key_metrics = insights.get("key_metrics", {})
    
    # Churn rate questions
    if "churn" in question_lower or "churn rate" in question_lower:
        churn_rate = metrics.get("churn_rate", 0)
        return f"The current churn rate for {business_name} is {churn_rate:.1%}. " + \
               insights.get("executive_summary", "")[:200]
    
    # Revenue questions
    if "revenue" in question_lower:
        revenue = metrics.get("revenue", 0)
        revenue_change = trends.get("revenue_change", "0%")
        return f"{business_name} has generated ${revenue:,.0f} in revenue during this period. " + \
               f"The revenue change compared to the previous period is {revenue_change}."
    
    # Customer questions
    if "customer" in question_lower or "customers" in question_lower:
        active = metrics.get("active_customers", 0)
        high_risk = metrics.get("high_risk_customers", 0)
        return f"{business_name} has {active:,} active customers. " + \
               f"Of these, {high_risk} are at high risk of churning."
    
    # Risk questions
    if "risk" in question_lower or "at risk" in question_lower:
        high_risk = metrics.get("high_risk_customers", 0)
        revenue_at_risk = metrics.get("revenue_at_risk_pct", 0)
        return f"There are {high_risk} high-risk customers, representing {revenue_at_risk:.1f}% of recent revenue."
    
    # Order questions
    if "order" in question_lower:
        orders = metrics.get("orders_count", 0)
        avg_value = metrics.get("avg_order_value", 0)
        return f"{business_name} has {orders:,} orders with an average order value of ${avg_value:.2f}."
    
    # Complaint questions
    if "complaint" in question_lower or "issue" in question_lower:
        complaints = bundle.get("complaint_reasons", {})
        if complaints:
            top_complaint = max(complaints.items(), key=lambda x: x[1])
            return f"The top complaint category is {top_complaint[0].replace('_', ' ').title()} with {top_complaint[1]} complaints."
        else:
            return f"No complaints recorded for {business_name} in this period."
    
    # Recommendation questions
    if "recommend" in question_lower or "suggest" in question_lower or "what should" in question_lower or "advice" in question_lower:
        recommendations = insights.get("recommendations", [])
        if recommendations:
            return f"Based on the analysis of {business_name}, here are the top recommendations:\n\n" + \
                   "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations[:3])])
        else:
            return "Continue monitoring key metrics and customer feedback."
    
    # Trend questions
    if "trend" in question_lower or "change" in question_lower or "compare" in question_lower:
        revenue_change = trends.get("revenue_change", "0%")
        orders_change = trends.get("orders_change", "0%")
        customers_change = trends.get("customers_change", "0%")
        return f"Comparing to the previous period:\n" + \
               f"- Revenue: {revenue_change}\n" + \
               f"- Orders: {orders_change}\n" + \
               f"- Customers: {customers_change}"
    
    # Average or mean questions
    if "average" in question_lower or "avg" in question_lower or "mean" in question_lower:
        if "order" in question_lower:
            return f"The average order value is ${metrics.get('avg_order_value', 0):.2f}."
        elif "revenue" in question_lower:
            return f"Average revenue per customer (90 days) is ${metrics.get('avg_revenue_90d', 0):.2f}."
        elif "recency" in question_lower or "last order" in question_lower:
            return f"Average customer recency is {metrics.get('avg_recency_days', 0):.0f} days."
        else:
            return f"Average metrics: Order value ${metrics.get('avg_order_value', 0):.2f}, " + \
                   f"Orders (90d): {metrics.get('avg_orders_90d', 0):.1f}, " + \
                   f"Revenue (90d): ${metrics.get('avg_revenue_90d', 0):.2f}"
    
    # Default response
    return f"I can answer questions about {business_name}'s metrics, trends, and insights. " + \
           f"Current churn rate is {metrics.get('churn_rate', 0):.1%} and revenue is ${metrics.get('revenue', 0):,.0f}. " + \
           "Try asking about churn rate, revenue, customers, orders, complaints, trends, or recommendations."

