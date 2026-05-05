import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

def compute_rfm(orders_df, reference_date=None):
    if reference_date is None:
        reference_date = orders_df["order_date"].max() + pd.Timedelta(days=1)

    rfm = orders_df.groupby("customer_id").agg(
        recency   = ("order_date", lambda x: (reference_date - x.max()).days),
        frequency = ("order_id",   "count"),
        monetary  = ("revenue",    "sum"),
    ).reset_index()

    # score 1-5 each dimension
    rfm["r_score"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def label_segment(row):
        if row["rfm_score"] >= 13:
            return "Champions"
        elif row["rfm_score"] >= 10:
            return "Loyal"
        elif row["rfm_score"] >= 7:
            return "Potential"
        elif row["rfm_score"] >= 4:
            return "At Risk"
        else:
            return "Lost"

    rfm["rfm_segment"] = rfm.apply(label_segment, axis=1)
    return rfm

def build_churn_features(orders_df, customers_df, reference_date=None):
    if reference_date is None:
        reference_date = orders_df["order_date"].max()

    cutoff = reference_date - pd.Timedelta(days=180)
    past   = orders_df[orders_df["order_date"] <  cutoff]
    future = orders_df[orders_df["order_date"] >= cutoff]

    churned_ids = set(past["customer_id"].unique()) - set(future["customer_id"].unique())

    features = past.groupby("customer_id").agg(
        recency      = ("order_date", lambda x: (cutoff - x.max()).days),
        frequency    = ("order_id",   "count"),
        monetary     = ("revenue",    "sum"),
        avg_order    = ("revenue",    "mean"),
        return_rate  = ("returned",   "mean"),
        discount_use = ("discount",   lambda x: (x > 0).mean()),
        n_categories = ("category",   "nunique"),
    ).reset_index()

    features = features.merge(
        customers_df[["customer_id", "age", "segment", "region", "channel"]],
        on="customer_id", how="left"
    )

    le = LabelEncoder()
    for col in ["segment", "region", "channel"]:
        features[col] = le.fit_transform(features[col].astype(str))

    features["churned"] = features["customer_id"].isin(churned_ids).astype(int)
    features = features.dropna()
    return features

def train_churn_model(features_df):
    feature_cols = ["recency","frequency","monetary","avg_order",
                    "return_rate","discount_use","n_categories",
                    "age","segment","region","channel"]

    X = features_df[feature_cols]
    y = features_df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, auc, importance, feature_cols

def get_business_insights(orders_df, rfm_df, comp_df, filters=None):
    insights = []

    # AOV trend
    orders_df["month"] = orders_df["order_date"].dt.to_period("M")
    monthly_aov = orders_df.groupby("month")["revenue"].mean()
    if len(monthly_aov) > 3:
        recent_aov  = monthly_aov.iloc[-3:].mean()
        earlier_aov = monthly_aov.iloc[-6:-3].mean()
        delta = (recent_aov - earlier_aov) / earlier_aov * 100
        direction = "risen" if delta > 0 else "fallen"
        insights.append(f"Average order value has {direction} {abs(delta):.1f}% in the last 3 months vs the prior quarter.")

    # At-risk customers
    at_risk = rfm_df[rfm_df["rfm_segment"] == "At Risk"]
    pct = len(at_risk) / len(rfm_df) * 100
    revenue_at_risk = at_risk["monetary"].sum()
    insights.append(f"{pct:.1f}% of customers ({len(at_risk)}) are 'At Risk' — representing ₹{revenue_at_risk:,.0f} in historical revenue.")

    # Competitor gap
    if comp_df is not None and len(comp_df):
        latest = comp_df[comp_df["month"] == comp_df["month"].max()]
        gap = latest["comp_market_share"].mean() - latest["our_market_share"].mean()
        insights.append(f"Competitor holds a {gap*100:.1f}% market share advantage on average across categories — widest in Electronics.")

    # Top channel
    ch_rev = orders_df.groupby("channel")["revenue"].sum()
    top_ch = ch_rev.idxmax()
    insights.append(f"'{top_ch}' is the highest-revenue channel ({ch_rev[top_ch]/ch_rev.sum()*100:.1f}% of total). Consider increasing investment here.")

    return insights

if __name__ == "__main__":
    orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
    customers = pd.read_csv("data/customers.csv", parse_dates=["join_date"])

    rfm = compute_rfm(orders)
    print("RFM segments:\n", rfm["rfm_segment"].value_counts())

    features = build_churn_features(orders, customers)
    model, auc, importance, feat_cols = train_churn_model(features)
    print(f"\nChurn model AUC: {auc:.3f}")
    print("\nTop features:\n", importance.head(5))
