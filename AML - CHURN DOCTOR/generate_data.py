# generate_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_data(
    n_businesses=5,
    days=365,
    start_date=datetime(2024, 1, 1),
    out_dir="churn_doctor_data"
):
    np.random.seed(42)
    random.seed(42)

    business_rows = []
    customer_rows = []
    order_rows = []
    interaction_rows = []

    business_ids = [f"b_{i+1}" for i in range(n_businesses)]
    industries = ["ecommerce", "retail", "subscription", "beauty", "fitness"]
    countries = ["US", "US", "US", "CA", "UK"]
    currencies = ["USD", "USD", "USD", "CAD", "GBP"]
    product_categories = ["Electronics", "Clothing", "Home", "Beauty", "Fitness", "Grocery"]
    acq_channels = ["ads", "organic", "referral", "offline"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Toronto", "London"]
    
    # Realistic business names by industry
    business_names_by_industry = {
        "ecommerce": ["TechMart", "ShopHub", "QuickCart", "Digital Depot", "MarketPlace Pro", "Online Express", "WebStore Central"],
        "retail": ["Urban Retail Co", "City Market", "Main Street Store", "Retail Express", "The Storefront", "Downtown Retail", "Corner Shop"],
        "subscription": ["SubscribeBox", "Monthly Delights", "The Subscription Co", "Recurring Rewards", "Boxed Bliss", "AutoShip Plus", "Monthly Magic"],
        "beauty": ["Glamour Beauty", "Pure Cosmetics", "Beauty Essentials", "Glow & Co", "The Beauty Bar", "Radiant Skincare", "Luxe Cosmetics"],
        "fitness": ["FitLife Gym", "ActiveZone", "PowerFit", "The Fitness Hub", "Strength & Co", "Elite Fitness", "Peak Performance"]
    }

    customer_id_counter = 1
    order_id_counter = 1
    interaction_id_counter = 1

    for idx, b_id in enumerate(business_ids):
        churn_window_days = random.choice([60, 90])
        base_daily_orders = random.randint(20, 120)
        avg_ticket = random.uniform(20, 120)
        discount_policy = random.choice(["low", "medium", "high"])
        
        # Get industry for this business
        industry = industries[idx % len(industries)]
        # Select a random name from the appropriate industry list
        available_names = business_names_by_industry[industry]
        business_name = random.choice(available_names)

        business_rows.append({
            "business_id": b_id,
            "name": business_name,
            "industry": industries[idx % len(industries)],
            "country": countries[idx % len(countries)],
            "currency": currencies[idx % len(currencies)],
            "churn_window_days": churn_window_days,
            "base_daily_orders": base_daily_orders,
            "avg_ticket_size": round(avg_ticket, 2),
            "discount_policy": discount_policy,
            "created_at": start_date.strftime("%Y-%m-%d")
        })

        customers_for_business = []
        # Track customer behavior profiles for more diverse churn scores
        customer_profiles = {}  # customer_id -> profile dict
        
        n_initial_customers = random.randint(150, 400)
        for _ in range(n_initial_customers):
            signup_offset = random.randint(0, 60)
            signup_date = start_date + timedelta(days=signup_offset)
            c_id = f"c_{customer_id_counter}"
            customer_id_counter += 1
            customers_for_business.append(c_id)
            
            # Assign behavior profile to create diverse churn patterns
            # Profile determines order frequency, recency, and engagement
            profile_type = random.choices(
                ["highly_active", "active", "moderate", "low_activity", "inactive"],
                weights=[0.15, 0.25, 0.30, 0.20, 0.10]  # More moderate customers for uniform distribution
            )[0]
            
            customer_rows.append({
                "customer_id": c_id,
                "business_id": b_id,
                "signup_date": signup_date.strftime("%Y-%m-%d"),
                "segment": random.choice(["value", "standard", "premium"]),
                "city": random.choice(cities),
                "acquisition_channel": random.choice(acq_channels)
            })
            
            # Store profile for this customer
            customer_profiles[c_id] = {
                "type": profile_type,
                "order_probability": {
                    "highly_active": 0.8,
                    "active": 0.6,
                    "moderate": 0.4,
                    "low_activity": 0.2,
                    "inactive": 0.05
                }[profile_type],
                "last_order_day": None,
                "order_count": 0
            }

        issue_profile = {
            "SHIPPING_DELAY": random.uniform(0.15, 0.35),
            "PRODUCT_QUALITY": random.uniform(0.15, 0.3),
            "PRICE_SENSITIVITY": random.uniform(0.1, 0.25),
            "CUSTOMER_SERVICE": random.uniform(0.1, 0.2),
            "LACK_OF_STOCK": random.uniform(0.05, 0.15),
            "USER_EXPERIENCE": random.uniform(0.05, 0.15),
            "COMPETITOR_SWITCH": random.uniform(0.05, 0.1),
            "OTHER": random.uniform(0.02, 0.05)
        }
        total_issue_weight = sum(issue_profile.values())
        for k in issue_profile:
            issue_profile[k] /= total_issue_weight

        def seasonal_factor(day_idx):
            date = start_date + timedelta(days=day_idx)
            month = date.month
            if month in [11, 12]:
                return 1.6
            elif month in [6, 7, 8]:
                return 1.2
            elif month in [1, 2]:
                return 0.8
            else:
                return 1.0

        crisis_start = random.randint(120, 220)
        crisis_end = crisis_start + random.randint(20, 40)

        for day in range(days):
            date = start_date + timedelta(days=day)
            sf = seasonal_factor(day)
            mean_orders = base_daily_orders * sf
            num_orders_today = np.random.poisson(mean_orders)

            new_customers_today = np.random.poisson(1.5)
            for _ in range(new_customers_today):
                signup_date = date
                c_id = f"c_{customer_id_counter}"
                customer_id_counter += 1
                customers_for_business.append(c_id)
                
                # Assign behavior profile for new customers
                profile_type = random.choices(
                    ["highly_active", "active", "moderate", "low_activity", "inactive"],
                    weights=[0.15, 0.25, 0.30, 0.20, 0.10]
                )[0]
                
                customer_rows.append({
                    "customer_id": c_id,
                    "business_id": b_id,
                    "signup_date": signup_date.strftime("%Y-%m-%d"),
                    "segment": random.choice(["value", "standard", "premium"]),
                    "city": random.choice(cities),
                    "acquisition_channel": random.choice(acq_channels)
                })
                
                # Store profile for new customer
                customer_profiles[c_id] = {
                    "type": profile_type,
                    "order_probability": {
                        "highly_active": 0.8,
                        "active": 0.6,
                        "moderate": 0.4,
                        "low_activity": 0.2,
                        "inactive": 0.05
                    }[profile_type],
                    "last_order_day": None,
                    "order_count": 0
                }

            if not customers_for_business:
                continue

            for _ in range(num_orders_today):
                # Select customer based on their activity profile for more diverse patterns
                # Weight selection by activity level
                customer_weights = []
                for cid in customers_for_business:
                    if cid in customer_profiles:
                        # Active customers more likely to order, but not exclusively
                        weight = customer_profiles[cid]["order_probability"]
                        # Add some randomness to prevent pure deterministic behavior
                        weight += random.uniform(-0.1, 0.1)
                        customer_weights.append(max(0.01, weight))
                    else:
                        customer_weights.append(0.5)  # Default weight for customers without profile
                
                # Normalize weights
                total_weight = sum(customer_weights)
                if total_weight > 0:
                    customer_weights = [w / total_weight for w in customer_weights]
                    customer_id = np.random.choice(customers_for_business, p=customer_weights)
                else:
                    customer_id = random.choice(customers_for_business)
                
                # Check if this customer should order based on their profile
                if customer_id in customer_profiles:
                    profile = customer_profiles[customer_id]
                    # Some customers have periods of inactivity
                    if profile["type"] == "inactive" and random.random() > profile["order_probability"]:
                        continue  # Skip this order for inactive customers
                    
                    # Moderate and low activity customers order less frequently
                    if profile["type"] in ["moderate", "low_activity"]:
                        days_since_last = day - profile["last_order_day"] if profile["last_order_day"] is not None else 999
                        min_days_between_orders = {
                            "moderate": 7,
                            "low_activity": 21
                        }.get(profile["type"], 0)
                        if days_since_last < min_days_between_orders:
                            continue  # Skip if too soon since last order
                order_time = date + timedelta(
                    hours=random.randint(9, 21),
                    minutes=random.randint(0, 59)
                )
                category = random.choice(product_categories)
                factor = {
                    "Electronics": 1.5,
                    "Clothing": 1.0,
                    "Home": 1.1,
                    "Beauty": 0.7,
                    "Fitness": 1.2,
                    "Grocery": 0.5
                }[category]
                revenue = np.random.lognormal(mean=np.log(avg_ticket * factor), sigma=0.4)

                if discount_policy == "low":
                    discount_pct = max(0, np.random.normal(5, 3))
                elif discount_policy == "medium":
                    discount_pct = max(0, np.random.normal(12, 5))
                else:
                    discount_pct = max(0, np.random.normal(20, 7))
                discount_pct = min(discount_pct, 50)
                discount_amount = revenue * (discount_pct / 100.0)

                base_delay = np.random.poisson(1.5)
                if crisis_start <= day <= crisis_end:
                    base_delay += np.random.poisson(2.0)
                delivery_delay_days = base_delay

                status = "delivered"
                order_id = f"o_{order_id_counter}"
                order_id_counter += 1

                order_rows.append({
                    "order_id": order_id,
                    "business_id": b_id,
                    "customer_id": customer_id,
                    "order_ts": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status,
                    "revenue": round(float(revenue), 2),
                    "discount_amount": round(float(discount_amount), 2),
                    "product_category": category,
                    "channel": random.choice(["web", "app", "offline"]),
                    "delivery_delay_days": int(delivery_delay_days)
                })
                
                # Update customer profile
                if customer_id in customer_profiles:
                    customer_profiles[customer_id]["last_order_day"] = day
                    customer_profiles[customer_id]["order_count"] += 1

                complaint_prob = 0.03
                if delivery_delay_days > 3:
                    complaint_prob += 0.15
                if discount_pct < 5 and random.random() < 0.5:
                    complaint_prob += 0.05

                if random.random() < complaint_prob:
                    if delivery_delay_days > 3 and random.random() < 0.6:
                        reason = "SHIPPING_DELAY"
                    else:
                        reason = random.choices(
                            list(issue_profile.keys()),
                            weights=list(issue_profile.values())
                        )[0]

                    templates = {
                        "SHIPPING_DELAY": [
                            "My order arrived much later than expected.",
                            "The delivery was delayed by several days, very disappointing.",
                            "Shipping took too long, not happy with the delay."
                        ],
                        "PRODUCT_QUALITY": [
                            "The product quality was worse than described.",
                            "Item broke after a few uses, poor quality.",
                            "Not satisfied with the quality of the product."
                        ],
                        "PRICE_SENSITIVITY": [
                            "The prices seem too high compared to other stores.",
                            "I used to get better discounts here.",
                            "Not worth the price anymore."
                        ],
                        "CUSTOMER_SERVICE": [
                            "Customer service was unhelpful and slow.",
                            "Support did not resolve my issue properly.",
                            "Very bad experience with your support team."
                        ],
                        "LACK_OF_STOCK": [
                            "Items are often out of stock.",
                            "I could not find the products I wanted.",
                            "Too many items unavailable when I try to buy."
                        ],
                        "USER_EXPERIENCE": [
                            "The website is confusing and hard to use.",
                            "Checkout process keeps failing.",
                            "Your app is buggy and frustrating."
                        ],
                        "COMPETITOR_SWITCH": [
                            "I found a better alternative elsewhere.",
                            "Other stores are offering better deals.",
                            "I'll probably buy from another brand next time."
                        ],
                        "OTHER": [
                            "Not happy with the overall experience.",
                            "Something just feels off with recent orders.",
                            "Service has declined over the last few months."
                        ]
                    }
                    text = random.choice(templates[reason])

                    interaction_rows.append({
                        "interaction_id": f"i_{interaction_id_counter}",
                        "business_id": b_id,
                        "customer_id": customer_id,
                        "ts": (order_time + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S"),
                        "channel": random.choice(["email", "chat", "phone", "review"]),
                        "raw_text": text,
                        "sentiment": "negative",
                        "reason_label": reason,
                        "resolved_flag": random.choice([0, 1])
                    })
                    interaction_id_counter += 1

    businesses_df = pd.DataFrame(business_rows)
    customers_df = pd.DataFrame(customer_rows)
    orders_df = pd.DataFrame(order_rows)
    interactions_df = pd.DataFrame(interaction_rows)

    os.makedirs(out_dir, exist_ok=True)
    businesses_df.to_csv(f"{out_dir}/businesses.csv", index=False)
    customers_df.to_csv(f"{out_dir}/customers.csv", index=False)
    orders_df.to_csv(f"{out_dir}/orders.csv", index=False)
    interactions_df.to_csv(f"{out_dir}/interactions.csv", index=False)
    print("Data written to", out_dir)

if __name__ == "__main__":
    generate_synthetic_data()
