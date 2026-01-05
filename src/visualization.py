import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_elbow(inertia: list):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(inertia) + 1), inertia, marker="o")
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    return fig

def plot_segment_heatmap(segment_stats: pd.DataFrame):
    mean_values = segment_stats.xs("mean", level=1, axis=1)
    normalized = (mean_values - mean_values.mean()) / mean_values.std()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(normalized, annot=True, cmap="RdYlGn", ax=ax)
    ax.set_title("Customer Segment Characteristics")
    return fig
