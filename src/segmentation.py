import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def scale_rfm(rfm: pd.DataFrame) -> np.ndarray:
    """
    Apply log transformation and standard scaling
    to RFM features.
    """
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    return scaler.fit_transform(rfm_log)


def apply_kmeans(
    rfm_scaled: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> KMeans:
    """
    Train KMeans clustering model.
    """
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=random_state
    )
    model.fit(rfm_scaled)
    return model


def assign_segments(rfm: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    """
    Assign business-friendly segment labels.
    """

    rfm = rfm.copy()
    rfm["Cluster"] = clusters

    segment_map = {
        0: "At Risk",
        1: "Loyal Customers",
        2: "Big Spenders",
        3: "New / Occasional"
    }

    rfm["Segment"] = rfm["Cluster"].map(segment_map)

    return rfm
