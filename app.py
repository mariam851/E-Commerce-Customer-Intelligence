import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src import (
    load_and_clean_data,
    compute_rfm,
    scale_rfm,
    apply_kmeans,
    assign_segments,
    segment_statistics
)

st.set_page_config(page_title="E-Commerce Customer Intelligence Dashboard", layout="wide")

@st.cache_data
def get_data():
    df = load_and_clean_data("data/E-commerce data.csv")
    rfm = compute_rfm(df)
    rfm_scaled = scale_rfm(rfm)
    kmeans = apply_kmeans(rfm_scaled)
    rfm = assign_segments(rfm, kmeans.labels_)
    return df, rfm

df, rfm = get_data()

churn_threshold_days = 90
rfm["Churn"] = (rfm["Recency"] > churn_threshold_days).astype(int)

# Churn Model
X = rfm[["Frequency", "Monetary"]]
y = rfm["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
X_all_scaled = scaler.transform(X)
rfm["ChurnRisk (%)"] = (model.predict_proba(X_all_scaled)[:, 1] * 100).round(1)

# KPI Cards
st.title("E-Commerce Customer Intelligence Dashboard")
total_customers = len(rfm)
total_revenue = rfm["Monetary"].sum()
avg_value = rfm["Monetary"].mean()
top_segment = rfm.groupby("Segment")["Monetary"].sum().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{total_customers:,}")
c2.metric("Revenue", f"${total_revenue:,.0f}")
c3.metric("Avg Customer Value", f"${avg_value:,.0f}")
c4.metric("Top Segment", top_segment)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", 
    "Segmentation & KPIs", 
    "Churn Analysis", 
    "Retention Actions",
    "Time-Based Insights",
    "Feature Importance"
])

# ---------------- Tab 1: Overview ----------------
with tab1:
    st.subheader("Customer RFM Data Overview")
    st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)

# ---------------- Tab 2: Segmentation & KPIs ----------------
with tab2:
    st.subheader("Segment KPIs & Revenue")
    stats = segment_statistics(rfm)
    st.dataframe(stats.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

    revenue_df = rfm.groupby("Segment")["Monetary"].sum().reset_index()
    fig = px.bar(revenue_df, x="Segment", y="Monetary", text="Monetary", title="Revenue by Segment", color="Segment")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Risk by Segment")
    churn_seg = rfm.groupby("Segment")["ChurnRisk (%)"].mean().reset_index()
    fig = px.bar(churn_seg, x="Segment", y="ChurnRisk (%)", text="ChurnRisk (%)", color="Segment", title="Average Churn Risk by Segment")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 3: Churn Analysis ----------------
with tab3:
    st.subheader("Churn Prediction Model")
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    st.metric("ROC-AUC Score", f"{auc:.2f}")

    fpr, tpr, _ = roc_curve(y_test, probs)
    fig_roc = px.line(x=fpr, y=tpr, labels={"x":"False Positive Rate", "y":"True Positive Rate"}, title=f"ROC Curve (AUC: {auc:.2f})")
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="red"))
    st.plotly_chart(fig_roc, use_container_width=True)

    top_risk = rfm[rfm["ChurnRisk (%)"] > 50].sort_values("Monetary", ascending=False)
    st.subheader("Top At-Risk Customers")
    st.dataframe(top_risk[["Segment", "Monetary", "ChurnRisk (%)"]].head(10))

# ---------------- Tab 4: Retention Actions ----------------
with tab4:
    st.subheader("Recommended Retention Actions")
    def retention_action(row):
        if row["ChurnRisk (%)"] > 70 and row["Segment"] == "Big Spenders":
            return "Urgent VIP Call"
        elif row["ChurnRisk (%)"] > 70:
            return "Immediate Retention Offer"
        elif row["ChurnRisk (%)"] > 40:
            return "Email Campaign"
        return "No Action Required"

    rfm["RetentionAction"] = rfm.apply(retention_action, axis=1)
    st.dataframe(
        rfm[["Segment", "Recency", "Frequency", "Monetary", "ChurnRisk (%)", "RetentionAction"]]
        .sort_values("ChurnRisk (%)", ascending=False),
        use_container_width=True
    )

# ---------------- Tab 5: Time-Based Insights ----------------
with tab5:
    st.subheader("Recency Distribution")
    fig_recency = px.histogram(rfm, x="Recency", nbins=30, title="Distribution of Recency (Days since Last Purchase)")
    st.plotly_chart(fig_recency, use_container_width=True)

    st.subheader("Monthly Revenue Trend")
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly_revenue = df.groupby("InvoiceMonth")["TotalSpent"].sum().reset_index()
    monthly_revenue["InvoiceMonth"] = monthly_revenue["InvoiceMonth"].dt.to_timestamp()
    fig_trend = px.line(monthly_revenue, x="InvoiceMonth", y="TotalSpent", title="Monthly Revenue Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------------- Tab 6: Feature Importance ----------------
with tab6:
    st.subheader("Feature Importance for Churn Prediction")
    coef_df = pd.DataFrame({
        "Feature": ["Frequency", "Monetary"],
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True)
    fig_feat = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="Feature Impact on Churn")
    st.plotly_chart(fig_feat, use_container_width=True)

st.download_button("Export Full Customer Intelligence Report", rfm.to_csv(index=False), "customer_intelligence.csv", "text/csv")

