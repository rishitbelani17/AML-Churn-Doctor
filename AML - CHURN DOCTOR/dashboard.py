# dashboard_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from features import load_raw, build_features_for_business
from explainer import score_with_shap
from insight_engine import build_evidence_bundle_for_business, generate_insights_for_business
from emailer import send_email, is_email_configured
from llm import generate_email

st.set_page_config(page_title="Churn Doctor", layout="wide")

@st.cache_data
def load_all():
    return load_raw()

businesses, customers, orders, interactions = load_raw()
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

st.sidebar.title("Churn Doctor")
business_id = st.sidebar.selectbox(
    "Select business",
    options=businesses["business_id"],
    format_func=lambda x: businesses.set_index("business_id").loc[x, "name"]
)

lookback_days = st.sidebar.slider("Lookback window (days)", min_value=14, max_value=90, value=30, step=7)

# ref date = last order date for that business
orders_b = orders[orders["business_id"] == business_id]
ref_date = orders_b["order_ts"].max()

st.title("Churn Doctor – Analytics")

# Features & scoring
df_feat, _ = build_features_for_business(business_id, ref_date)
scored = score_with_shap(df_feat)
scored_sorted = scored.sort_values("churn_score", ascending=False)

# KPIs
period_start = ref_date - timedelta(days=lookback_days)
mask_curr = (orders_b["order_ts"] >= period_start) & (orders_b["order_ts"] <= ref_date)
curr_orders = orders_b[mask_curr]

revenue = curr_orders["revenue"].sum()
orders_count = curr_orders["order_id"].nunique()
active_customers = curr_orders["customer_id"].nunique()
churn_rate = scored["is_churned"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", f"${revenue:,.0f}")
col2.metric("Orders", f"{orders_count}")
col3.metric("Active customers", f"{active_customers}")
col4.metric("Churn rate", f"{churn_rate:.1%}")

st.markdown("---")

# Charts
st.subheader("Churn score distribution")
st.bar_chart(scored["churn_score"])

st.subheader("Top high-risk customers")
st.dataframe(
    scored_sorted[["customer_id", "churn_score", "recency_days", "orders_90d", "revenue_90d", "top_shap_features"]]
    .head(20)
)

st.markdown("---")

st.subheader("Complaint reasons (LLM-derived)")

ints_b = interactions[interactions["business_id"] == business_id].copy()
ints_b["ts"] = pd.to_datetime(ints_b["ts"])
ints_curr = ints_b[(ints_b["ts"] >= period_start) & (ints_b["ts"] <= ref_date)]

if not ints_curr.empty:
    reason_counts = ints_curr["reason_label"].value_counts()
    st.bar_chart(reason_counts)
    st.write("Sample complaints:")
    st.dataframe(ints_curr[["ts", "customer_id", "channel", "reason_label", "raw_text"]].head(20))
else:
    st.info("No complaints in this period.")

st.markdown("---")

st.subheader("Explanation drivers (SHAP global importance)")
# approximate global importance from top_shap_features
global_imp = {}
for feats in scored["top_shap_features"]:
    for f in feats:
        global_imp[f["feature"]] = global_imp.get(f["feature"], 0.0) + abs(f["shap_value"])

imp_df = pd.DataFrame(
    [{"feature": k, "importance": v} for k, v in global_imp.items()]
).sort_values("importance", ascending=False).head(15)

st.bar_chart(imp_df.set_index("feature"))

st.markdown("---")

st.subheader("LLM-generated Insights")

# Generate insights
with st.spinner("Generating insights..."):
    bundle, insights = generate_insights_for_business(business_id, ref_date, lookback_days)

# Display insights
st.markdown(f"### {insights.get('headline', 'Churn Analysis Summary')}")

if insights.get('top_reasons'):
    st.markdown("#### Top Churn Drivers")
    for i, reason in enumerate(insights.get('top_reasons', [])[:5], 1):
        with st.expander(f"{i}. {reason['name']} (Impact: {reason['impact_estimate']})"):
            for evidence in reason.get('evidence', []):
                st.write(f"• {evidence}")

if insights.get('recommendations'):
    st.markdown("#### Recommendations")
    for i, rec in enumerate(insights.get('recommendations', [])[:5], 1):
        st.write(f"{i}. {rec}")

st.markdown("---")

# Email sending section
st.subheader("📧 Send Report via Email")

# Add refresh button to reload email config
if st.button("🔄 Refresh Email Status", help="Click if you just updated email settings"):
    st.cache_data.clear()
    st.rerun()

# Check if email is configured
try:
    email_configured = is_email_configured()
except Exception as e:
    st.error(f"Error checking email config: {e}")
    email_configured = False

if not email_configured:
    st.warning("⚠️ Email not configured. To enable email sending:")
    with st.expander("📝 How to configure email"):
        st.markdown("""
        **Option 1: Edit emailer.py**
        - Open `emailer.py`
        - Replace `SMTP_USER` with your Gmail address
        - Replace `SMTP_PASS` with your Gmail App Password
        
        **Option 2: Use environment variables**
        ```bash
        export SMTP_USER="your-email@gmail.com"
        export SMTP_PASS="your-app-password"
        ```
        
        **How to get Gmail App Password:**
        1. Go to your Google Account settings
        2. Enable 2-Step Verification
        3. Go to "App passwords"
        4. Generate a new app password for "Mail"
        5. Use that password in the configuration
        """)
else:
    st.success("✓ Email is configured")

# Email form
with st.form("email_form", clear_on_submit=False):
    recipient_email = st.text_input(
        "Recipient Email",
        value=st.session_state.get("recipient_email", ""),
        placeholder="recipient@example.com",
        key="email_input"
    )
    
    owner_name = st.text_input(
        "Recipient Name",
        value=st.session_state.get("owner_name", "Business Owner"),
        key="name_input"
    )
    
    # Enable button if email is configured and recipient email is provided
    has_recipient = bool(recipient_email and recipient_email.strip())
    can_send = email_configured and has_recipient
    
    # Button should be enabled if we can send AND have a recipient
    button_disabled = not can_send
    
    # Store values in session state
    st.session_state["recipient_email"] = recipient_email
    st.session_state["owner_name"] = owner_name
    
    # Show status
    if not email_configured:
        st.info("ℹ️ Configure email settings to enable sending")
    elif not has_recipient:
        st.info("ℹ️ Enter recipient email address")
    else:
        st.success("✓ Ready to send")
    
    send_button = st.form_submit_button(
        "📤 Send Email Report",
        # disabled=button_disabled,
        use_container_width=True,
        type="primary" if not button_disabled else "secondary"
    )
    
    if send_button and recipient_email:
        if not email_configured:
            st.error("Please configure email settings first. Edit emailer.py to set SMTP_USER and SMTP_PASS.")
        else:
            try:
                # Get business name
                business_name = businesses[businesses["business_id"] == business_id].iloc[0]["name"]
                period_str = f"{bundle['period']['start'][:10]} to {bundle['period']['end'][:10]}"
                
                # Generate email body
                email_body = generate_email(
                    owner_name=owner_name,
                    business_name=business_name,
                    period_str=period_str,
                    insights=insights
                )
                
                subject = f"[Churn Doctor] Churn Analysis Report for {business_name}"
                
                # Send email
                with st.spinner("Sending email..."):
                    success, message = send_email(
                        to_email=recipient_email,
                        subject=subject,
                        body=email_body
                    )
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Show raw bundle data (collapsible)
with st.expander("📊 Raw Evidence Bundle Data"):
    st.json(bundle)

