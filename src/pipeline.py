from __future__ import annotations

import json
from pathlib import Path

from .clustering import cluster_from_distance
from .config import ExperimentConfig
from .data import load_stock_data, prepare_price_matrix
from .evaluation import evaluate_clustering
from .features import compute_returns, correlation_distance
from .visualization import save_cluster_sizes, save_distance_heatmap


def ensure_output_dirs(config: ExperimentConfig) -> None:
    for path in [config.tables_dir, config.figures_dir, config.metrics_dir]:
        path.mkdir(parents=True, exist_ok=True)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    raw = load_stock_data(config.input_path, config)
    prices = prepare_price_matrix(raw, config)
    returns = compute_returns(prices, config.return_type)
    distance = correlation_distance(returns)
    assignments = cluster_from_distance(
        distance,
        n_clusters=config.n_clusters,
        method=config.clustering_method,
        random_state=config.random_state,
    )
    metrics, cluster_summary = evaluate_clustering(distance, assignments)

    ensure_output_dirs(config)

    prices.to_csv(config.tables_dir / "price_matrix.csv")
    returns.to_csv(config.tables_dir / "returns.csv")
    distance.to_csv(config.tables_dir / "distance_matrix.csv")
    assignments.to_csv(config.tables_dir / "cluster_assignments.csv", index=False)
    cluster_summary.to_csv(config.tables_dir / "cluster_summary.csv", index=False)

    save_distance_heatmap(
        distance,
        config.figures_dir / "distance_heatmap.png",
        dpi=config.figure_dpi,
    )
    save_cluster_sizes(
        cluster_summary,
        config.figures_dir / "cluster_sizes.png",
        dpi=config.figure_dpi,
    )

    metrics_payload = {
        **metrics,
        "input_path": str(Path(config.input_path)),
        "return_type": config.return_type,
        "clustering_method": config.clustering_method,
    }
    (config.metrics_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.metrics_dir / "run_config.json").write_text(
        json.dumps(config.to_jsonable(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "prices": prices,
        "returns": returns,
        "distance": distance,
        "assignments": assignments,
        "cluster_summary": cluster_summary,
        "metrics": metrics_payload,
    }
