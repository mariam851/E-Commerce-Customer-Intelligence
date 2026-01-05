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

## Project Structure
Designed for scalability and clean code maintenance:
```graphql
project/
│
├─ app.py                    # Main Business Dashboard
├─ requirements.txt          # Environment dependencies
├─ src/                      # Production-grade modular logic
│   ├─ load_and_clean_data.py
│   ├─ compute_rfm.py
│   ├─ scale_rfm.py
│   ├─ apply_kmeans.py
│   ├─ assign_segments.py
│   └─ segment_statistics.py
├─ data/                     # Transactional Datasets
└─ reports/
    └─ figures/              # Visual assets for analysis

## Core Business Intelligence & Analysis

![Customer Data Overview]

### RFM Data Intelligence
We analyze the customer base through three fundamental dimensions to understand the "Health" of our customer relationships:

- **Recency:** Time since last purchase (Engagement indicator)  
- **Frequency:** Total transactions (Loyalty indicator)  
- **Monetary:** Total revenue (Financial value)  

### Customer Segmentation (Targeting Strategy)
Using KMeans Clustering, we categorize customers into 4 personas to enable Precision Marketing:

- **Loyal Customers:** High-frequency assets; focus on cross-selling  
- **Big Spenders:** High monetary contribution; focus on premium upsells  
- **At Risk:** Declining engagement; requires immediate reactivation offers  
- **New / Occasional:** Recently acquired; requires onboarding nurture sequences  

![Segment Risk Analysis]

### Churn Prediction (Risk Mitigation)
We implemented a Logistic Regression model to calculate a **Churn Risk Score (%)**.

- **Key Discovery:** Frequency (-2.62) has a significantly higher impact on retention than Monetary (-1.67)  
- **Business Insight:** Building shopping habits is more valuable for long-term stability than isolated high-ticket sales  

![Churn Prediction Model Performance]

### Time-Based Behavioral Insights
Understanding when and how customers interact with the platform helps in timing our interventions.

- **Recency Distribution:** Helps management define the "Defection Point" to trigger recovery campaigns  
- **Seasonality:** Tracks monthly revenue trends to align marketing budgets with high-activity periods  

---

![Recency Distribution]

## Business Impact & Value Proposition
- **Enhanced ROI:** By targeting only at-risk or high-value customers, we optimize marketing spend and avoid "blanket discounts."  
- **Revenue Recovery:** The Action Engine identifies VIP customers at risk, triggering "Urgent VIP Calls" to save high-stakes accounts  
- **Data-Driven Culture:** Shifts the organization from "guessing" to "knowing" based on statistical significance  

---

## How to Deploy

1. **Clone the repository**
```bash
git clone <your-repo-url>

### Install dependencies

pip install -r requirements.txt


### Launch the Dashboard

streamlit run app.py
