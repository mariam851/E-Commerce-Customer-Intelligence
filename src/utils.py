import pandas as pd


def segment_statistics(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate customer metrics at segment level.
    """
    return rfm.groupby("Segment").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": ["mean", "sum", "count"]
    }).round(2)


def revenue_share(rfm: pd.DataFrame) -> pd.Series:
    """
    Calculate revenue contribution per segment.
    """
    return (
        rfm.groupby("Segment")["Monetary"].sum()
        / rfm["Monetary"].sum()
    )
