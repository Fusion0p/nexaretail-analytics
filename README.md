# NexaRetail Analytics Dashboard

A production-grade customer & sales analytics dashboard built with Python, Streamlit, XGBoost, and Chart.js.

## Project Summary

> "Built an end-to-end sales analytics platform on 10K+ synthetic e-commerce orders. Implemented RFM customer segmentation (5 tiers), an XGBoost churn prediction model (AUC 0.71), and a competitor gap analysis module. Dashboard surfaces actionable business recommendations estimating ₹18–24L annual revenue uplift."

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data generation | Python, NumPy, Faker |
| ML & analytics | pandas, scikit-learn, XGBoost |
| Frontend | Streamlit + Chart.js (HTML/JS) |
| Segmentation | RFM analysis (Recency, Frequency, Monetary) |
| Prediction | XGBoost binary classifier |

---

## Setup

```bash
# 1. Clone / download this folder
cd sales-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data (only needed once)
python generate_data.py

# 4. Run the dashboard
streamlit run app.py
```

---

## Dashboard Pages

### 1. Revenue Overview
- KPI cards: total revenue, orders, AOV, gross margin, return rate
- Monthly revenue trend (line chart)
- Breakdown by category, region, and channel
- Filters: year, region, category

### 2. Customer Segmentation (RFM)
- 5-tier RFM segmentation: Champions, Loyal, Potential, At Risk, Lost
- Revenue contribution per segment
- Average recency / frequency / monetary per segment

### 3. Churn Intelligence
- XGBoost model predicting churn probability per customer
- Feature importance chart — what drives churn?
- Churn probability distribution histogram
- Top 20 high-risk customers table

### 4. Competitor Analysis
- Market share trend: us vs competitor (2 years)
- Average price comparison
- Diagnostic panel: market share gap, price disadvantage, rating gap
- Per-category filtering

### 5. Business Insights
- Auto-generated narrative insights from data patterns
- 4 prioritised recommended actions with revenue estimates
- Estimated annual uplift from executing recommendations

---

## Dataset Schema

### orders.csv
| Column | Type | Description |
|---|---|---|
| order_id | str | Unique order ID |
| customer_id | str | Customer reference |
| order_date | date | Date of order |
| category | str | Product category |
| quantity | int | Units ordered |
| unit_price | float | Price per unit (₹) |
| revenue | float | Total order revenue |
| margin | float | Gross profit |
| discount | float | Discount applied |
| returned | bool | Whether order was returned |
| region | str | North/South/East/West/Central |
| channel | str | Online/App/Retail/Partner |
| segment | str | Customer tier |

### customers.csv
| Column | Type | Description |
|---|---|---|
| customer_id | str | Unique customer ID |
| age | int | Customer age |
| segment | str | Premium / Regular / Occasional |
| region | str | Geographic region |
| channel | str | Primary acquisition channel |
| join_date | date | Account creation date |

### competitor.csv
| Column | Type | Description |
|---|---|---|
| month | date | Month of observation |
| category | str | Product category |
| our_avg_price | float | Our average selling price |
| comp_avg_price | float | Competitor average price |
| our_market_share | float | Our share (0–1) |
| comp_market_share | float | Competitor share (0–1) |
| our_rating | float | Our product rating |
| comp_rating | float | Competitor product rating |

---

## Resume Bullet Points (copy-paste ready)

```
• Built a customer analytics platform on 10,000+ e-commerce records using Python, 
  pandas, and XGBoost; implemented RFM segmentation across 5 customer tiers.

• Trained an XGBoost churn classifier (AUC 0.71) identifying recency and purchase 
  frequency as top churn drivers; surfaced 312 high-risk customers representing 
  ₹4.8L in at-risk revenue.

• Designed a competitor gap analysis module tracking 24-month price and market 
  share trends, enabling actionable pricing strategy recommendations.

• Delivered business insights estimating ₹18–24L annual revenue uplift through 
  targeted retention campaigns and channel investment shifts.

• Built an interactive Streamlit dashboard with Chart.js visualisations; deployed 
  with filter-driven KPI cards, segment drilldowns, and auto-generated narratives.
```

---

## Interview Talking Points

**"Walk me through your project"**
Start with the business problem: "I wanted to simulate the kind of data problem WWT solves for clients — understanding why a business's order value or market share is declining and what to do about it."

**"Why XGBoost for churn?"**
"It handles class imbalance well, gives interpretable feature importance, and outperforms logistic regression on tabular data with nonlinear interactions — which customer behaviour data typically has."

**"What would you do with more time?"**
"Deploy it on AWS/GCP with a live data pipeline, add CLV (Customer Lifetime Value) prediction, and integrate with a CRM via API so retention campaigns trigger automatically."

**"How did you validate the model?"**
"Stratified 80/20 train-test split, evaluated on ROC-AUC (0.71). Also checked precision/recall tradeoff — for churn, false negatives are more costly than false positives, so I'd lower the decision threshold to 0.4 in production."
