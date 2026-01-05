import pandas as pd
import datetime as dt


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary)
    features for each customer.
    """

    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "count",
        "TotalSpent": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    return rfm
