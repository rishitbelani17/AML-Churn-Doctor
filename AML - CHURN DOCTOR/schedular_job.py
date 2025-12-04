# scheduler_job.py
from features import build_features_for_business
from shap_explainer import score_with_shap
from insight_engine import generate_insights_for_business
from emailer import send_email
from features import load_raw
from llm_client import generate_email

def churn_analysis_job(business_id: str, owner_email: str, owner_name: str = "Owner"):
    businesses, *_ = load_raw()
    b_row = businesses[businesses["business_id"] == business_id].iloc[0]

    # compute features & scores
    df_feat, ref_date = build_features_for_business(business_id)
    scored = score_with_shap(df_feat)

    # generate insights via LLM
    bundle, insights = generate_insights_for_business(business_id, ref_date)

    period_str = f"{bundle['period']['start']} to {bundle['period']['end']}"
    email_body = generate_email(
        owner_name=owner_name,
        business_name=b_row['name'],
        period_str=period_str,
        insights=insights
    )

    subject = f"[Churn Doctor] Weekly churn summary for {b_row['name']}"
    send_email(owner_email, subject, email_body)

if __name__ == "__main__":
    # Example manual trigger (replace with real email)
    churn_analysis_job(business_id="b_1", owner_email="you@example.com", owner_name="Rishit")
