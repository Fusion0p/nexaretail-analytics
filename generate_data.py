import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

CATEGORIES = {
    "Electronics": {"avg_price": 320, "std": 180, "margin": 0.22},
    "Clothing":    {"avg_price": 65,  "std": 35,  "margin": 0.55},
    "Home & Kitchen": {"avg_price": 95, "std": 55, "margin": 0.42},
    "Sports":      {"avg_price": 110, "std": 70,  "margin": 0.38},
    "Beauty":      {"avg_price": 45,  "std": 20,  "margin": 0.62},
}

REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["Online", "App", "Retail", "Partner"]

COMPETITOR = {
    "Electronics": {"avg_price": 295, "market_share": 0.38, "rating": 4.3},
    "Clothing":    {"avg_price": 58,  "market_share": 0.29, "rating": 4.5},
    "Home & Kitchen": {"avg_price": 88, "market_share": 0.25, "rating": 4.2},
    "Sports":      {"avg_price": 98,  "market_share": 0.31, "rating": 4.4},
    "Beauty":      {"avg_price": 42,  "market_share": 0.35, "rating": 4.6},
}

def generate_customers(n=2000):
    customers = []
    for i in range(n):
        age = int(np.random.normal(35, 12))
        age = max(18, min(70, age))
        segment = np.random.choice(
            ["Premium", "Regular", "Occasional"],
            p=[0.15, 0.50, 0.35]
        )
        region = np.random.choice(REGIONS, p=[0.22, 0.18, 0.20, 0.25, 0.15])
        channel = np.random.choice(CHANNELS, p=[0.45, 0.30, 0.15, 0.10])
        customers.append({
            "customer_id": f"C{i+1:04d}",
            "age": age,
            "segment": segment,
            "region": region,
            "channel": channel,
            "join_date": datetime(2022, 1, 1) + timedelta(days=random.randint(0, 365))
        })
    return pd.DataFrame(customers)

def generate_orders(customers_df, n=18000):
    orders = []
    start = datetime(2023, 1, 1)
    end   = datetime(2024, 12, 31)

    seg_order_mu = {"Premium": 12, "Regular": 5, "Occasional": 1.5}

    for _, cust in customers_df.iterrows():
        mu = seg_order_mu[cust["segment"]]
        n_orders = max(0, int(np.random.poisson(mu)))
        for _ in range(n_orders):
            cat = np.random.choice(list(CATEGORIES.keys()))
            info = CATEGORIES[cat]
            price = max(5, np.random.normal(info["avg_price"], info["std"]))
            qty = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.4, 0.25, 0.15, 0.1, 0.07, 0.03])

            # seasonal boost: Q4 higher
            order_date = start + timedelta(days=random.randint(0, (end - start).days))
            seasonal_mult = 1.3 if order_date.month in [11, 12] else (0.85 if order_date.month in [1, 2] else 1.0)
            revenue = round(price * qty * seasonal_mult, 2)
            margin = round(revenue * info["margin"], 2)

            # simulate returns
            returned = np.random.random() < 0.07
            discount = round(random.choice([0, 0, 0, 5, 10, 15, 20]) / 100 * revenue, 2)

            orders.append({
                "order_id":    f"O{len(orders)+1:06d}",
                "customer_id": cust["customer_id"],
                "order_date":  order_date,
                "category":    cat,
                "quantity":    qty,
                "unit_price":  round(price, 2),
                "revenue":     revenue,
                "margin":      margin,
                "discount":    discount,
                "returned":    returned,
                "region":      cust["region"],
                "channel":     cust["channel"],
                "segment":     cust["segment"],
            })

    df = pd.DataFrame(orders)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def generate_competitor_data():
    rows = []
    months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    for cat, info in COMPETITOR.items():
        base_share = info["market_share"]
        for i, month in enumerate(months):
            # competitor gains share over time (the WWT problem scenario)
            trend = 0.004 * i
            rows.append({
                "month": month,
                "category": cat,
                "our_avg_price": CATEGORIES[cat]["avg_price"] * (1 + 0.002 * i),
                "comp_avg_price": info["avg_price"] * (1 - 0.001 * i),
                "our_market_share": round(max(0.10, base_share - trend + np.random.normal(0, 0.01)), 3),
                "comp_market_share": round(min(0.65, base_share + trend + np.random.normal(0, 0.01)), 3),
                "our_rating": round(4.1 + np.random.normal(0, 0.05), 2),
                "comp_rating": round(info["rating"] + np.random.normal(0, 0.03), 2),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Generating customers...")
    customers = generate_customers(2000)
    customers.to_csv("data/customers.csv", index=False)

    print("Generating orders...")
    orders = generate_orders(customers)
    orders.to_csv("data/orders.csv", index=False)

    print("Generating competitor data...")
    comp = generate_competitor_data()
    comp.to_csv("data/competitor.csv", index=False)

    print(f"Done. Orders: {len(orders)}, Customers: {len(customers)}")
    print(orders.head())
