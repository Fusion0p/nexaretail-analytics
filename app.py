import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ml_engine import compute_rfm, build_churn_features, train_churn_model, get_business_insights

st.set_page_config(
    page_title="NexaRetail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0a0a0f; }
  section[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e2e; }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 100%; }
  h1,h2,h3 { font-family: 'Syne', sans-serif; }
  div[data-testid="stSelectbox"] label,
  div[data-testid="stMultiSelect"] label { color: #888 !important; font-size: 12px; letter-spacing:.06em; text-transform:uppercase; }
  div[data-baseweb="select"] { background: #1a1a2e !important; border-color: #2a2a3e !important; }
  .stButton>button { background:#7c3aed; color:#fff; border:none; border-radius:8px; font-family:'Syne'; font-weight:600; }
  .stButton>button:hover { background:#6d28d9; }
  div[data-testid="stRadio"] label { color: #ccc !important; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme base ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(17,17,40,1)",
    plot_bgcolor="rgba(17,17,40,1)",
    font=dict(family="DM Sans", color="#9ca3af", size=12),
    margin=dict(l=20, r=20, t=44, b=40),
    xaxis=dict(gridcolor="#1e1e3a", linecolor="#1e1e3a", tickfont=dict(color="#6b7280")),
    yaxis=dict(gridcolor="#1e1e3a", linecolor="#1e1e3a", tickfont=dict(color="#6b7280")),
)
LEGEND_DEFAULT = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af"))

COLORS = ["#7c3aed","#6366f1","#0ea5e9","#10b981","#f59e0b","#f97316","#ef4444","#ec4899"]
SEG_COLORS = {"Champions":"#10b981","Loyal":"#6366f1","Potential":"#f59e0b","At Risk":"#f97316","Lost":"#ef4444"}

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    orders    = pd.read_csv("data/orders.csv",     parse_dates=["order_date"])
    customers = pd.read_csv("data/customers.csv",  parse_dates=["join_date"])
    comp      = pd.read_csv("data/competitor.csv", parse_dates=["month"])
    return orders, customers, comp

@st.cache_data
def run_ml(key):
    orders, customers, _ = load_data()
    rfm      = compute_rfm(orders)
    features = build_churn_features(orders, customers)
    model, auc, importance, feat_cols = train_churn_model(features)
    features["churn_prob"] = model.predict_proba(features[feat_cols])[:, 1]
    return rfm, features, auc, importance

orders, customers, comp = load_data()
rfm, churn_df, auc, importance = run_ml("v1")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem;'>
      <div style='font-family:Syne;font-size:22px;font-weight:800;color:#e2e8f0;letter-spacing:-.5px;'>NexaRetail</div>
      <div style='font-family:DM Mono;font-size:11px;color:#6366f1;margin-top:2px;letter-spacing:.1em;'>ANALYTICS INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📈  Overview",
        "👥  Customer Segments",
        "⚠️  Churn Intelligence",
        "🆚  Competitor Analysis",
        "💡  Business Insights"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;'>Filters</div>", unsafe_allow_html=True)
    year_filter     = st.selectbox("Year", ["All", "2023", "2024"])
    region_filter   = st.multiselect("Region",   orders["region"].unique().tolist(),   default=orders["region"].unique().tolist())
    category_filter = st.multiselect("Category", orders["category"].unique().tolist(), default=orders["category"].unique().tolist())

# ── Apply filters ──────────────────────────────────────────────────────────────
df = orders.copy()
if year_filter != "All":
    df = df[df["order_date"].dt.year == int(year_filter)]
if region_filter:
    df = df[df["region"].isin(region_filter)]
if category_filter:
    df = df[df["category"].isin(category_filter)]

# ── Helpers ────────────────────────────────────────────────────────────────────
def html(content): st.markdown(content, unsafe_allow_html=True)

def page_header(title, subtitle):
    html(f"""
    <div style='margin-bottom:1.5rem;'>
      <div style='font-family:Syne;font-size:28px;font-weight:800;color:#f1f5f9;margin-bottom:.25rem;'>{title}</div>
      <div style='font-size:14px;color:#6b7280;'>{subtitle}</div>
    </div>
    """)

def kpi_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value, delta, delta_label) in zip(cols, metrics):
        color = "#34d399" if delta >= 0 else "#f87171"
        arrow = "↑" if delta >= 0 else "↓"
        col.markdown(f"""
        <div style='background:#111128;border:1px solid #1e1e3a;border-radius:14px;padding:1.25rem 1.5rem;height:115px;'>
          <div style='font-family:DM Mono;font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>{label}</div>
          <div style='font-family:Syne;font-size:24px;font-weight:700;color:#f1f5f9;line-height:1.2;'>{value}</div>
          <div style='font-size:12px;color:{color};margin-top:.3rem;'>{arrow} {abs(delta):.1f}% {delta_label}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    page_header("Revenue Overview", "Filtered view across regions and categories")

    total_rev    = df["revenue"].sum()
    total_orders = len(df)
    aov          = df["revenue"].mean()
    margin_pct   = df["margin"].sum() / df["revenue"].sum() * 100
    ret_rate     = df["returned"].mean() * 100

    kpi_row([
        ("Total Revenue",   f"₹{total_rev/1e6:.2f}M",  8.3,  "vs last period"),
        ("Total Orders",    f"{total_orders:,}",         5.1,  "vs last period"),
        ("Avg Order Value", f"₹{aov:.0f}",              -2.4, "vs last period"),
        ("Gross Margin",    f"{margin_pct:.1f}%",        1.2,  "vs last period"),
        ("Return Rate",     f"{ret_rate:.1f}%",         -0.3, "vs last period"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: trend + donut
    col1, col2 = st.columns([3, 2])
    with col1:
        monthly = df.copy()
        monthly["month"] = monthly["order_date"].dt.to_period("M").dt.to_timestamp()
        monthly_rev = monthly.groupby("month")["revenue"].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_rev["month"], y=monthly_rev["revenue"],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2.5),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.10)",
            marker=dict(size=5, color="#7c3aed"),
            hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
            title=dict(text="Monthly Revenue Trend", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with col2:
        cat_rev = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        fig2 = go.Figure(go.Pie(
            labels=cat_rev["category"], values=cat_rev["revenue"],
            hole=0.55,
            marker=dict(colors=COLORS, line=dict(color="#0a0a0f", width=2)),
            textfont=dict(color="#9ca3af", size=11),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300,
            title=dict(text="Revenue by Category", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig2.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11, color="#9ca3af")))

        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

    # Row 2: region + channel
    col3, col4 = st.columns(2)
    with col3:
        reg_rev = df.groupby("region")["revenue"].sum().reset_index().sort_values("revenue")
        fig3 = go.Figure(go.Bar(
            x=reg_rev["revenue"], y=reg_rev["region"],
            orientation="h",
            marker=dict(color=COLORS[:len(reg_rev)], line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=280,
            title=dict(text="Revenue by Region", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig3.update_xaxes(tickprefix="₹", tickformat=",.0f")
        fig3.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig3, width='stretch', config={"displayModeBar": False})

    with col4:
        ch_rev = df.groupby("channel")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        fig4 = go.Figure(go.Bar(
            x=ch_rev["channel"], y=ch_rev["revenue"],
            marker=dict(color="#6366f1", line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=280,
            title=dict(text="Revenue by Channel", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig4.update_yaxes(tickprefix="₹", tickformat=",.0f")
        fig4.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig4, width='stretch', config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Customer" in page:
    page_header("Customer Segmentation", "RFM analysis — Recency · Frequency · Monetary value")

    seg_counts  = rfm["rfm_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment","count"]
    seg_revenue = rfm.groupby("rfm_segment")["monetary"].sum().reset_index()
    seg_revenue.columns = ["segment","revenue"]
    seg = seg_counts.merge(seg_revenue, on="segment")

    col1, col2 = st.columns([1, 2])
    with col1:
        total_custs = seg["count"].sum()
        for _, row in seg.iterrows():
            pct = row["count"] / total_custs * 100
            color = SEG_COLORS.get(row["segment"], "#888")
            html(f"""
            <div style='background:#111128;border:1px solid #1e1e3a;border-left:4px solid {color};border-radius:0 12px 12px 0;padding:.9rem 1.1rem;margin-bottom:.65rem;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                  <div style='font-family:Syne;font-size:14px;font-weight:600;color:#f1f5f9;'>{row["segment"]}</div>
                  <div style='font-size:12px;color:#6b7280;margin-top:2px;'>{row["count"]} customers · {pct:.1f}%</div>
                </div>
                <div style='font-family:DM Mono;font-size:13px;color:{color};font-weight:500;'>₹{row["revenue"]/1000:.0f}K</div>
              </div>
              <div style='margin-top:.5rem;background:#1a1a2e;border-radius:4px;height:3px;'>
                <div style='width:{pct}%;background:{color};height:3px;border-radius:4px;'></div>
              </div>
            </div>
            """)

    with col2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=seg["segment"], y=seg["count"], name="Customers",
            marker=dict(color=[SEG_COLORS.get(s,"#888") for s in seg["segment"]], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y} customers<extra></extra>"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=seg["segment"], y=seg["revenue"], name="Revenue (₹)",
            mode="lines+markers", line=dict(color="#e2e8f0", width=2),
            marker=dict(size=8, color="#e2e8f0"),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"
        ), secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
            title=dict(text="Segment — Customers vs Revenue", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig.update_yaxes(title_text="Customers", gridcolor="#1e1e3a", tickfont=dict(color="#6b7280"), secondary_y=False)
        fig.update_yaxes(title_text="Revenue (₹)", tickprefix="₹", gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#6b7280"), secondary_y=True)
        fig.update_xaxes(tickfont=dict(color="#9ca3af"), gridcolor="#1e1e3a")
        fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    html("<div style='font-family:Syne;font-size:16px;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;'>RFM Statistics by Segment</div>")
    rfm_stats = rfm.groupby("rfm_segment").agg(
        avg_recency=("recency","mean"),
        avg_frequency=("frequency","mean"),
        avg_monetary=("monetary","mean")
    ).round(1).reset_index()
    rfm_stats.columns = ["Segment","Avg Recency (days)","Avg Orders","Avg Revenue (₹)"]
    st.dataframe(rfm_stats, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHURN INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Churn" in page:
    page_header("Churn Intelligence", f"XGBoost model · AUC = {auc:.3f}")

    churn_rate  = churn_df["churned"].mean() * 100
    high_risk   = churn_df[churn_df["churn_prob"] > 0.6]
    rev_at_risk = high_risk["monetary"].sum()

    kpi_row([
        ("Overall Churn Rate",   f"{churn_rate:.1f}%",        -1.2, "vs benchmark"),
        ("High-Risk Customers",  f"{len(high_risk)}",           7.4, "flagged"),
        ("Revenue at Risk",      f"₹{rev_at_risk/1000:.0f}K",  0.0, "this cohort"),
        ("Model AUC",            f"{auc:.3f}",                  0.0, "accuracy"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        feat_top = importance.head(8)
        fig = go.Figure(go.Bar(
            x=feat_top["importance"], y=feat_top["feature"],
            orientation="h",
            marker=dict(color=feat_top["importance"],
                colorscale=[[0,"#3730a3"],[0.5,"#7c3aed"],[1,"#a78bfa"]],
                line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="Feature Importance — What drives churn?", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with col2:
        bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ["0–20%","20–40%","40–60%","60–80%","80–100%"]
        counts = pd.cut(churn_df["churn_prob"], bins=bins).value_counts().sort_index().tolist()
        fig2 = go.Figure(go.Bar(
            x=labels, y=counts,
            marker=dict(color=["#10b981","#6366f1","#f59e0b","#f97316","#ef4444"], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y} customers<extra></extra>"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="Churn Probability Distribution", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig2.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    html("<div style='font-family:Syne;font-size:16px;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;'>Top 20 High-Risk Customers</div>")
    top_risk = churn_df.nlargest(20, "churn_prob")[["customer_id","churn_prob","frequency","monetary","recency"]].copy()
    top_risk["churn_prob"] = (top_risk["churn_prob"] * 100).round(1).astype(str) + "%"
    top_risk["monetary"]   = top_risk["monetary"].round(0).astype(int)
    top_risk.columns = ["Customer ID","Churn Risk","Orders","Revenue (₹)","Days Since Last Order"]
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COMPETITOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Competitor" in page:
    page_header("Competitor Analysis", "The WWT case study scenario — why is our market share declining?")

    comp_cat = st.selectbox("Select Category", comp["category"].unique())
    cat_data = comp[comp["category"] == comp_cat].sort_values("month")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cat_data["month"], y=cat_data["our_market_share"] * 100,
            name="NexaRetail", mode="lines+markers",
            line=dict(color="#7c3aed", width=2.5),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.10)",
            hovertemplate="<b>NexaRetail</b><br>%{x|%b %Y}: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=cat_data["month"], y=cat_data["comp_market_share"] * 100,
            name="Competitor", mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
            hovertemplate="<b>Competitor</b><br>%{x|%b %Y}: %{y:.1f}%<extra></extra>"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="Market Share Trend (%)", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig.update_yaxes(ticksuffix="%")
        fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cat_data["month"], y=cat_data["our_avg_price"],
            name="NexaRetail Price", mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            hovertemplate="<b>NexaRetail</b><br>%{x|%b %Y}: ₹%{y:.0f}<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=cat_data["month"], y=cat_data["comp_avg_price"],
            name="Competitor Price", mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            hovertemplate="<b>Competitor</b><br>%{x|%b %Y}: ₹%{y:.0f}<extra></extra>"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="Average Price Comparison (₹)", font=dict(color="#e2e8f0", size=14, family="Syne"), x=0))
        fig2.update_yaxes(tickprefix="₹")
        fig2.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')))

        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

    latest    = cat_data.iloc[-1]
    share_gap = (latest["comp_market_share"] - latest["our_market_share"]) * 100
    price_gap = latest["our_avg_price"] - latest["comp_avg_price"]
    rating_gap= latest["comp_rating"] - latest["our_rating"]

    html(f"""
    <div style='background:#1a0a2e;border:1px solid #4c1d95;border-radius:14px;padding:1.5rem;margin-top:.5rem;'>
      <div style='font-family:Syne;font-size:15px;font-weight:700;color:#c4b5fd;margin-bottom:1rem;'>⚡ Diagnostic — {comp_cat}</div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;'>
        <div>
          <div style='font-size:11px;color:#7c3aed;text-transform:uppercase;letter-spacing:.08em;'>Market Share Gap</div>
          <div style='font-family:DM Mono;font-size:22px;color:#f87171;margin:.25rem 0;'>+{share_gap:.1f}%</div>
          <div style='font-size:12px;color:#6b7280;'>Competitor advantage</div>
        </div>
        <div>
          <div style='font-size:11px;color:#7c3aed;text-transform:uppercase;letter-spacing:.08em;'>Price Disadvantage</div>
          <div style='font-family:DM Mono;font-size:22px;color:#fbbf24;margin:.25rem 0;'>₹{price_gap:.0f}</div>
          <div style='font-size:12px;color:#6b7280;'>We charge more</div>
        </div>
        <div>
          <div style='font-size:11px;color:#7c3aed;text-transform:uppercase;letter-spacing:.08em;'>Rating Gap</div>
          <div style='font-family:DM Mono;font-size:22px;color:#f87171;margin:.25rem 0;'>{rating_gap:+.2f}</div>
          <div style='font-size:12px;color:#6b7280;'>Competitor rated higher</div>
        </div>
      </div>
    </div>
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:
    page_header("Business Insights", "Auto-generated narrative from data patterns")

    insights = get_business_insights(df, rfm, comp)
    for i, insight in enumerate(insights):
        html(f"""
        <div style='background:#111128;border:1px solid #1e1e3a;border-radius:12px;padding:1rem 1.25rem;margin-bottom:.75rem;display:flex;gap:1rem;align-items:flex-start;'>
          <div style='font-family:DM Mono;font-size:18px;color:#7c3aed;min-width:28px;'>{i+1:02d}</div>
          <div style='font-size:14px;color:#d1d5db;line-height:1.6;'>{insight}</div>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    html("<div style='font-family:Syne;font-size:18px;font-weight:700;color:#e2e8f0;margin-bottom:1rem;'>Recommended Actions</div>")

    actions = [
        ("Reactivation Campaign",      "At Risk & Lost segments",  "Target 398 at-risk customers with a 15% discount on their most-purchased category. Estimated recovery: ₹2.4L revenue.", "#f59e0b"),
        ("Price Review — Electronics", "Competitive pricing",       "Competitor undercuts us by ₹25 on average. A 6–8% price reduction could recover ~4% market share based on elasticity estimates.", "#ef4444"),
        ("App Channel Investment",     "Channel mix optimisation",  "App channel shows 23% higher LTV than web. Doubling app-exclusive offers could shift 15% of web traffic to higher-retention channel.", "#10b981"),
        ("Champion Loyalty Program",   "Retention of top segment",  "Champions (15% of customers) drive 38% of revenue. A tiered loyalty program reduces their churn risk from 12% to an estimated 4%.", "#6366f1"),
    ]
    for title, tag, desc, color in actions:
        html(f"""
        <div style='background:#111128;border:1px solid #1e1e3a;border-left:4px solid {color};border-radius:0 12px 12px 0;padding:1rem 1.25rem;margin-bottom:.75rem;'>
          <div style='display:flex;align-items:center;gap:.75rem;margin-bottom:.4rem;flex-wrap:wrap;'>
            <div style='font-family:Syne;font-size:15px;font-weight:600;color:#f1f5f9;'>{title}</div>
            <div style='font-size:11px;background:#1e1e3a;color:#9ca3af;padding:2px 8px;border-radius:20px;'>{tag}</div>
          </div>
          <div style='font-size:13px;color:#9ca3af;line-height:1.6;'>{desc}</div>
        </div>
        """)

    html(f"""
    <div style='background:linear-gradient(135deg,#1e1b4b,#1a0a2e);border:1px solid #4c1d95;border-radius:14px;padding:1.5rem;margin-top:1rem;text-align:center;'>
      <div style='font-family:Syne;font-size:13px;color:#a78bfa;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;'>Estimated Annual Revenue Uplift</div>
      <div style='font-family:Syne;font-size:36px;font-weight:800;color:#f1f5f9;'>₹18.6L – ₹24.2L</div>
      <div style='font-size:12px;color:#6b7280;margin-top:.25rem;'>If all recommended actions are executed over 6 months</div>
    </div>
    """)
