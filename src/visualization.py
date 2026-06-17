from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_distance_heatmap(
    distance_matrix: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(distance_matrix) * 0.45), 6))
    image = ax.imshow(distance_matrix.values, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(distance_matrix.columns)))
    ax.set_xticklabels(distance_matrix.columns, rotation=90)
    ax.set_yticks(range(len(distance_matrix.index)))
    ax.set_yticklabels(distance_matrix.index)
    ax.set_title("Ticker Correlation Distance Matrix")
    fig.colorbar(image, ax=ax, label="1 - correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_cluster_sizes(
    cluster_summary: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cluster_summary["cluster"].astype(str), cluster_summary["n_tickers"])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of tickers")
    ax.set_title("Cluster Sizes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

