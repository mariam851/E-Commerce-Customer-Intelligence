# main.py
from src import (
    load_and_clean_data,
    compute_rfm,
    scale_rfm,
    apply_kmeans,
    assign_segments,
    segment_statistics,
    revenue_share,
    plot_elbow,
    plot_segment_heatmap
)

def main():
    data_path = "data/E-commerce data.csv"

    # Load & Clean Data
    df = load_and_clean_data(data_path)

    # Feature Engineering (RFM)
    rfm = compute_rfm(df)

    # Scaling & Clustering
    rfm_scaled = scale_rfm(rfm)
    model = apply_kmeans(rfm_scaled)
    rfm_segmented = assign_segments(rfm, model.labels_)

    # Business Insights
    stats = segment_statistics(rfm_segmented)
    rev = revenue_share(rfm_segmented)

    # Show Results
    print("=== Segment Statistics ===")
    print(stats)
    print("\n=== Revenue Share ===")
    print(rev)

    # Optional: Visualizations
    plot_segment_heatmap(stats)

if __name__ == "__main__":
    main()
