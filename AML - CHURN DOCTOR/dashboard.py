# dashboard_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from features import load_raw, build_features_for_business
from shap_explainer import score_with_shap
from insight_engine import build_evidence_bundle_for_business

st.set_page_config(page_title="Churn Doctor", layout="wide")

@st.cache_data
def load_all():
    return load_raw()

businesses, customers, orders, interactions = load_all()
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

st.subheader("LLM-generated overview")

bundle = build_evidence_bundle_for_business(business_id, ref_date, lookback_days)
st.json(bundle, expanded=False)
st.write("You can call `generate_insights_for_business` and render the LLM's narrative here if desired.")
