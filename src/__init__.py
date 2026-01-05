# src/__init__.py
# Package initialization + convenient imports

from .data_preprocessing import load_and_clean_data
from .feature_engineering import compute_rfm
from .segmentation import scale_rfm, apply_kmeans, assign_segments
from .utils import segment_statistics, revenue_share
from .visualization import plot_elbow, plot_segment_heatmap
