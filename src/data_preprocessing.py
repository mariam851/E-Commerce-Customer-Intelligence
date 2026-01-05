import pandas as pd


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    Load raw e-commerce transaction data and apply
    essential cleaning rules for customer analysis.
    """

    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Remove transactions without customer identifier
    df = df.dropna(subset=["CustomerID"])

    # Remove returns and invalid prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Type casting
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Revenue per transaction
    df["TotalSpent"] = df["Quantity"] * df["UnitPrice"]

    return df
