from __future__ import annotations

import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_clustering(
    distance_matrix: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    labels = assignments.set_index("ticker").loc[distance_matrix.index, "cluster"]
    unique_clusters = labels.nunique()

    metrics: dict[str, float | int] = {
        "n_tickers": int(len(labels)),
        "n_clusters": int(unique_clusters),
    }

    if unique_clusters > 1 and unique_clusters < len(labels):
        metrics["silhouette_distance"] = float(
            silhouette_score(distance_matrix, labels, metric="precomputed")
        )

    cluster_summary = (
        assignments.groupby("cluster")
        .agg(n_tickers=("ticker", "size"), tickers=("ticker", lambda s: ", ".join(s)))
        .reset_index()
        .sort_values("cluster")
    )
    cluster_summary["share_pct"] = (
        cluster_summary["n_tickers"] / cluster_summary["n_tickers"].sum() * 100
    ).round(2)

    return metrics, cluster_summary

