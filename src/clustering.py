from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler


def cluster_from_distance(
    distance_matrix: pd.DataFrame,
    n_clusters: int,
    method: str = "agglomerative",
    random_state: int = 42,
) -> pd.DataFrame:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if n_clusters > len(distance_matrix):
        raise ValueError("n_clusters cannot exceed the number of tickers.")

    tickers = distance_matrix.index.tolist()

    if method == "agglomerative":
        try:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="average",
            )
        labels = model.fit_predict(distance_matrix)
    elif method == "kmeans":
        features = StandardScaler().fit_transform(distance_matrix)
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(features)
    else:
        raise ValueError("clustering_method must be 'agglomerative' or 'kmeans'.")

    return pd.DataFrame({"ticker": tickers, "cluster": labels}).sort_values(
        ["cluster", "ticker"]
    )

