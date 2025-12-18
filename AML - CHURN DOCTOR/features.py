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
        ref_date = get_reference_date(ords) - timedelta(days=int(churn_window_days))

    ords_past = ords[ords["order_ts"] < ref_date].copy()
    ints_past = ints[ints["ts"] < ref_date].copy()
    ords_future = ords[ords["order_ts"] >= ref_date]

    last_order = ords_past.groupby("customer_id")["order_ts"].max().rename("last_order_ts")
    order_count = ords_past.groupby("customer_id")["order_id"].nunique().rename("order_count")
    revenue_sum = ords_past.groupby("customer_id")["revenue"].sum().rename("revenue_sum")
    
    ninety_days_ago = ref_date - timedelta(days=90)
    revenue_90d = ords_past[ords_past["order_ts"] >= ninety_days_ago].groupby("customer_id")["revenue"].sum().rename("revenue_90d")
    orders_90d = ords_past[ords_past["order_ts"] >= ninety_days_ago].groupby("customer_id")["order_id"].nunique().rename("orders_90d")
    num_complaints_90d = ints_past[ints_past["ts"] >= ninety_days_ago].groupby("customer_id")["interaction_id"].nunique().rename("num_complaints_90d")

    df = cust.set_index("customer_id")
    df["tenure_days"] = (ref_date - df["signup_date"]).dt.days
    
    df = df.join([last_order, order_count, revenue_sum, revenue_90d, orders_90d, num_complaints_90d])

    df["recency_days"] = (ref_date - df["last_order_ts"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(df["tenure_days"])

    cols_to_fill = ["order_count", "revenue_sum", "revenue_90d", "orders_90d", "num_complaints_90d"]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    future_customers = ords_future["customer_id"].unique()
    df["is_churned"] = (~df.index.isin(future_customers)).astype(int)
    
    df = df[df["tenure_days"] >= 0]

    feature_cols = [
        "tenure_days", "recency_days", "order_count", 
        "revenue_sum", "revenue_90d", "orders_90d", "num_complaints_90d"
    ]
    
    df_features = df[feature_cols + ["is_churned"]].reset_index()
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
