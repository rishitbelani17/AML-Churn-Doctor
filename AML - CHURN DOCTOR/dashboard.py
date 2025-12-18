# dashboard_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt

from features import load_raw, build_features_for_business
from explainer import score_with_shap
from insight_engine import build_evidence_bundle_for_business, generate_insights_for_business
from emailer import send_email, is_email_configured
from llm import generate_email
from chatbot import answer_question

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
max_date = orders_b["order_ts"].max()
ref_date = max_date - timedelta(days=lookback_days)

st.title("Churn Doctor - Analytics")

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
# Create histogram of churn scores
churn_score_df = pd.DataFrame({
    "Churn Score": scored["churn_score"],
    "Count": 1
})
churn_hist = alt.Chart(churn_score_df).mark_bar().encode(
    x=alt.X("Churn Score:Q", bin=alt.Bin(maxbins=20), title="Churn Score"),
    y=alt.Y("count():Q", title="Number of Customers")
).properties(
    width=600,
    height=400
)
st.altair_chart(churn_hist, use_container_width=True)

st.subheader("Top high-risk customers")
# Format top_shap_features for display
display_df = scored_sorted[["customer_id", "churn_score", "recency_days", "orders_90d", "revenue_90d", "top_shap_features"]].copy()
display_df["top_shap_features"] = display_df["top_shap_features"].apply(
    lambda feats: ", ".join([f"{f['feature']}: {f['shap_value']:.3f}" for f in feats[:3]])
)
st.dataframe(display_df.head(20))

st.markdown("---")

st.subheader("Complaint reasons (LLM-derived)")

ints_b = interactions[interactions["business_id"] == business_id].copy()
ints_b["ts"] = pd.to_datetime(ints_b["ts"])
ints_curr = ints_b[(ints_b["ts"] >= period_start) & (ints_b["ts"] <= ref_date)]

if not ints_curr.empty:
    reason_counts = ints_curr["reason_label"].value_counts().reset_index()
    reason_counts.columns = ["Complaint Reason", "Count"]
    reason_chart = alt.Chart(reason_counts).mark_bar().encode(
        x=alt.X("Complaint Reason:N", title="Complaint Reason", sort="-y"),
        y=alt.Y("Count:Q", title="Number of Complaints")
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(reason_chart, use_container_width=True)
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
)

# Filter out features with zero importance
imp_df = imp_df[imp_df["importance"] > 0]

# Sort by importance in descending order and take top 15
imp_df = imp_df.sort_values("importance", ascending=False).head(15)

# Format feature names for better readability
imp_df["feature_formatted"] = imp_df["feature"].str.replace("_", " ").str.title()

shap_chart = alt.Chart(imp_df).mark_bar().encode(
    x=alt.X("importance:Q", title="SHAP Importance Score"),
    y=alt.Y("feature_formatted:N", title="Feature", sort=alt.EncodingSortField(field="importance", order="descending"))
).properties(
    width=600,
    height=400
)
st.altair_chart(shap_chart, use_container_width=True)

st.markdown("---")

# Additional Metrics Section
st.subheader("Customer Risk Analysis")

# Customer risk distribution
risk_distribution = pd.DataFrame({
    "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
    "Count": [
        int((scored["churn_score"] > 0.7).sum()),
        int(((scored["churn_score"] > 0.4) & (scored["churn_score"] <= 0.7)).sum()),
        int((scored["churn_score"] <= 0.4).sum())
    ]
})

risk_chart = alt.Chart(risk_distribution).mark_bar().encode(
    x=alt.X("Risk Level:N", title="Risk Level", sort=["High Risk", "Medium Risk", "Low Risk"]),
    y=alt.Y("Count:Q", title="Number of Customers")
).properties(
    width=400,
    height=300
)
st.altair_chart(risk_chart, use_container_width=True)

# Additional metrics in columns
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Order Value", f"${curr_orders['revenue'].mean():.2f}" if not curr_orders.empty else "$0.00")
col2.metric("Avg Recency", f"{scored['recency_days'].mean():.0f} days")
col3.metric("Avg Orders (90d)", f"{scored['orders_90d'].mean():.1f}")
col4.metric("Avg Revenue (90d)", f"${scored['revenue_90d'].mean():.2f}")

st.markdown("---")

st.subheader("LLM-generated Insights")

# Generate insights
with st.spinner("Generating insights..."):
    bundle, insights = generate_insights_for_business(business_id, ref_date, lookback_days)

# Display insights
st.markdown(f"### {insights.get('headline', 'Churn Analysis Summary')}")

# Executive Summary
if insights.get('executive_summary'):
    st.markdown("#### 📊 Executive Summary")
    st.info(insights.get('executive_summary'))

# Key Metrics and Trends
col1, col2 = st.columns(2)

with col1:
    if insights.get('key_metrics'):
        st.markdown("#### 📈 Key Metrics")
        metrics = insights.get('key_metrics', {})
        for key, value in metrics.items():
            st.metric(key.replace("_", " ").title(), value)

with col2:
    if insights.get('trends'):
        st.markdown("#### 📉 Trends (vs Previous Period)")
        trends = insights.get('trends', {})
        for key, value in trends.items():
            # Determine if trend is positive or negative
            change_val = float(value.replace("%", "").replace("+", ""))
            delta_color = "normal"
            if "revenue" in key or "orders" in key or "customers" in key:
                delta_color = "inverse" if change_val < 0 else "normal"
            st.metric(key.replace("_", " ").title(), value, delta_color=delta_color)

st.markdown("---")

# Detailed Analysis
if insights.get('detailed_analysis'):
    st.markdown("#### 🔍 Detailed Analysis")
    st.markdown(insights.get('detailed_analysis'))

st.markdown("---")

# Top Churn Drivers
if insights.get('top_reasons'):
    st.markdown("#### 🎯 Top Churn Drivers")
    for i, reason in enumerate(insights.get('top_reasons', [])[:5], 1):
        with st.expander(f"{i}. {reason['name']} (Impact: {reason['impact_estimate']})"):
            for evidence in reason.get('evidence', []):
                st.write(f"• {evidence}")

st.markdown("---")

# Recommendations
if insights.get('recommendations'):
    st.markdown("#### 💡 Recommendations")
    for i, rec in enumerate(insights.get('recommendations', []), 1):
        st.write(f"**{i}.** {rec}")

st.markdown("---")

# Chatbot Section
st.subheader("💬 Ask Questions About This Business")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_business_id" not in st.session_state:
    st.session_state.last_business_id = None

# Get business name
business_name = businesses[businesses["business_id"] == business_id].iloc[0]["name"]

# Clear chat history if business changed
if st.session_state.last_business_id != business_id:
    st.session_state.chat_history = []
    st.session_state.last_business_id = business_id

# Display chat history
chat_container = st.container()
with chat_container:
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)

# Chat input
user_question = st.chat_input(f"Ask a question about {business_name}...")

if user_question:
    # Add user message to history
    st.session_state.chat_history.append(("user", user_question))
    
    # Generate answer
    with st.spinner("Analyzing business data..."):
        try:
            answer = answer_question(
                question=user_question,
                business_name=business_name,
                bundle=bundle,
                insights=insights
            )
            st.session_state.chat_history.append(("assistant", answer))
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append(("assistant", error_msg))
    
    # Rerun to update chat display
    st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Suggested questions
st.markdown("**💡 Suggested Questions:**")
suggested_questions = [
    "What is the current churn rate?",
    "How is revenue performing?",
    "How many customers are at risk?",
    "What are the top complaints?",
    "What recommendations do you have?",
    "What is the average order value?",
    "How many active customers do we have?"
]

cols = st.columns(3)
for i, q in enumerate(suggested_questions):
    with cols[i % 3]:
        if st.button(q, key=f"suggest_{i}", use_container_width=True):
            # Add question to chat
            st.session_state.chat_history.append(("user", q))
            with st.spinner("Analyzing..."):
                try:
                    answer = answer_question(q, business_name, bundle, insights)
                    st.session_state.chat_history.append(("assistant", answer))
                except Exception as e:
                    st.session_state.chat_history.append(("assistant", f"Error: {str(e)}"))
            st.rerun()

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

