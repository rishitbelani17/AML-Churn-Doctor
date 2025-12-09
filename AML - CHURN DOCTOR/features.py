# features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DATA_DIR, CHURN_WINDOW_DAYS_DEFAULT

def load_raw():
    businesses = pd.read_csv(f"{DATA_DIR}/businesses.csv")
    customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    interactions = pd.read_csv(f"{DATA_DIR}/interactions.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    interactions["ts"] = pd.to_datetime(interactions["ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    return businesses, customers, orders, interactions

def get_reference_date(orders: pd.DataFrame) -> datetime:
    return orders["order_ts"].max()

def build_features_for_business(business_id: str, ref_date: datetime = None) -> pd.DataFrame:
    businesses, customers, orders, interactions = load_raw()
    b_row = businesses[businesses["business_id"] == business_id].iloc[0]
    churn_window_days = b_row.get("churn_window_days", CHURN_WINDOW_DAYS_DEFAULT)

    cust = customers[customers["business_id"] == business_id].copy()
    ords = orders[orders["business_id"] == business_id].copy()
    ints = interactions[interactions["business_id"] == business_id].copy()

    if ref_date is None:
        ref_date = get_reference_date(ords)

    # basic aggregates
    ords = ords.sort_values("order_ts")
    last_order = ords.groupby("customer_id")["order_ts"].max().rename("last_order_ts")
    first_order = ords.groupby("customer_id")["order_ts"].min().rename("first_order_ts")
    order_count = ords.groupby("customer_id")["order_id"].nunique().rename("order_count")
    revenue_sum = ords.groupby("customer_id")["revenue"].sum().rename("revenue_sum")
    revenue_90d = ords[ords["order_ts"] >= ref_date - timedelta(days=90)] \
        .groupby("customer_id")["revenue"].sum().rename("revenue_90d")
    orders_90d = ords[ords["order_ts"] >= ref_date - timedelta(days=90)] \
        .groupby("customer_id")["order_id"].nunique().rename("orders_90d")

    avg_discount_pct = (ords.groupby("customer_id")[["discount_amount", "revenue"]]
                        .apply(lambda g: (g["discount_amount"].sum() / g["revenue"].sum()) * 100
                               if g["revenue"].sum() > 0 else 0.0)
                        .rename("avg_discount_pct"))

    avg_delay = ords.groupby("customer_id")["delivery_delay_days"].mean().rename("avg_delivery_delay")

    # category shares
    category_share = (ords
        .groupby(["customer_id", "product_category"])["order_id"]
        .count()
        .unstack(fill_value=0))
    category_share = category_share.div(category_share.sum(axis=1), axis=0).add_prefix("cat_share_")

    # interactions
    ints_last_90 = ints[ints["ts"] >= ref_date - timedelta(days=90)]
    num_complaints_90d = ints_last_90.groupby("customer_id")["interaction_id"] \
        .nunique().rename("num_complaints_90d")

    reason_counts = (ints_last_90
                     .groupby(["customer_id", "reason_label"])["interaction_id"]
                     .count()
                     .unstack(fill_value=0)
                     .add_prefix("complaints_"))

    # merge all
    df = cust.set_index("customer_id")
    df["tenure_days"] = (ref_date - df["signup_date"]).dt.days

    df = df.join([last_order, first_order, order_count, revenue_sum,
                  revenue_90d, orders_90d, avg_discount_pct, avg_delay,
                  category_share, num_complaints_90d, reason_counts])

    df["recency_days"] = (ref_date - df["last_order_ts"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(df["tenure_days"])  # no orders yet

    df["orders_90d"] = df["orders_90d"].fillna(0)
    df["revenue_90d"] = df["revenue_90d"].fillna(0)
    df["num_complaints_90d"] = df["num_complaints_90d"].fillna(0)
    df["avg_discount_pct"] = df["avg_discount_pct"].fillna(0)
    df["avg_delivery_delay"] = df["avg_delivery_delay"].fillna(0)

    # churn label
    df["is_churned"] = (df["last_order_ts"].isna()) | \
                       ((ref_date - df["last_order_ts"]).dt.days > churn_window_days)

    # simple encoding for segment
    df["segment_value"] = df["segment"].map({"value": 0, "standard": 1, "premium": 2})

    feature_cols = [c for c in df.columns if c not in
                    ["signup_date", "last_order_ts", "first_order_ts",
                     "segment", "city", "acquisition_channel"]]

    df_features = df[feature_cols].reset_index()
    df_features["business_id"] = business_id

    return df_features, ref_date

def build_features_all_businesses():
    businesses, *_ = load_raw()
    all_df = []
    ref_dict = {}
    for b in businesses["business_id"]:
        df_b, ref_b = build_features_for_business(b)
        all_df.append(df_b)
        ref_dict[b] = ref_b
    full = pd.concat(all_df, ignore_index=True)
    return full, ref_dict
