# E-Commerce Customer Intelligence Dashboard

An advanced **Business Intelligence (BI)** and **Machine Learning** solution designed to transform raw e-commerce data into strategic growth levers. This dashboard focuses on **RFM Segmentation**, **Churn Prediction**, and **Automated Retention Strategies** to maximize profitability and optimize marketing ROI.

---

## Strategic Business Overview
In the e-commerce sector, the cost of customer acquisition (CAC) is rising. Retaining an existing customer is **5x more cost-effective** than acquiring a new one. 

This project provides a **Profit-Driven Framework** to:
- **Maximize Revenue:** Identifying high-value segments for targeted VIP treatment.
- **Minimize Revenue Leakage:** Predicting customer churn before it happens.
- **Operational Excellence:** Automating decision-making through an "Action Engine" that suggests immediate business interventions.

---

## Tech Stack & Architecture
- **Language:** Python 3.x
- **Dashboard:** Streamlit (Interactive UI)
- **Data Science:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Logistic Regression, KMeans Clustering)
- **Visualization:** Plotly, Seaborn (Dynamic and Publication-quality charts)

---

## Core Business Intelligence & Analysis

### RFM Data Intelligence
We analyze the customer base through three fundamental dimensions to understand the "Health" of our customer relationships:
- **Recency:** Time since last purchase (Engagement indicator).
- **Frequency:** Total transactions (Loyalty indicator).
- **Monetary:** Total revenue (Financial value).

![Customer Data Overview](C:/Users/MASTER/OneDrive/Desktop/projects%20to%20find%20job/E-Commerce%20Customer%20Behavior%20Analysis/reports/figures/customers_data.png)

### Customer Segmentation (Targeting Strategy)
Using **KMeans Clustering**, we categorize customers into 4 personas to enable **Precision Marketing**:
1. **Loyal Customers:** High-frequency assets; focus on cross-selling.
2. **Big Spenders:** High monetary contribution; focus on premium upsells.
3. **At Risk:** Declining engagement; requires immediate reactivation offers.
4. **New / Occasional:** Recently acquired; requires onboarding nurture sequences.

![Segment Risk Analysis](C:/Users/MASTER/OneDrive/Desktop/projects%20to%20find%20job/E-Commerce%20Customer%20Behavior%20Analysis/reports/figures/avg_chunk_risk_by_segment.png)

### Churn Prediction (Risk Mitigation)
We implemented a **Logistic Regression** model to calculate a **Churn Risk Score (%)**.
- **Key Discovery:** Our analysis shows that **Frequency (-2.62)** has a significantly higher impact on retention than **Monetary (-1.67)**. 
- **Business Insight:** Building **shopping habits** is more valuable for long-term stability than isolated high-ticket sales.

![Churn Prediction Model Performance](C:/Users/MASTER/OneDrive/Desktop/projects%20to%20find%20job/E-Commerce%20Customer%20Behavior%20Analysis/reports/figures/churn_prediction_model.png)

### Time-Based Behavioral Insights
Understanding when and how customers interact with the platform helps in timing our interventions.
- **Recency Distribution:** Helps management define the "Defection Point" to trigger recovery campaigns.
- **Seasonality:** Tracks monthly revenue trends to align marketing budgets with high-activity periods.

![Recency Distribution](C:/Users/MASTER/OneDrive/Desktop/projects%20to%20find%20job/E-Commerce%20Customer%20Behavior%20Analysis/reports/figures/recency_distribution.png)

---

## Business Impact & Value Proposition
- **Enhanced ROI:** By targeting only at-risk or high-value customers, we optimize marketing spend and avoid "blanket discounts."
- **Revenue Recovery:** The **Action Engine** identifies VIP customers at risk, triggering "Urgent VIP Calls" to save high-stakes accounts.
- **Data-Driven Culture:** Shifts the organization from "guessing" to "knowing" based on statistical significance.

---

## Project Structure (Modules)
Designed for scalability and clean code maintenance, the project follows a modular architecture:

```graphql
project/
│
├─ app.py                    # Main Business Dashboard (Streamlit UI)
├─ requirements.txt          # Environment dependencies
├─ data/                     # Transactional Datasets (CSV)
├─ reports/
│   └─ figures/              # Visual assets and performance charts
└─ src/                      # Production-grade modular logic
    ├─ load_and_clean_data.py
    ├─ compute_rfm.py
    ├─ scale_rfm.py
    ├─ apply_kmeans.py
    ├─ assign_segments.py
    └─ segment_statistics.py
```

## How to Deploy

Clone the repository:

```graphql
git clone <your-repo-url>
```

Install dependencies:

```graphql
pip install -r requirements.txt
```

Launch the Dashboard:

```graphql
streamlit run app.py
```
