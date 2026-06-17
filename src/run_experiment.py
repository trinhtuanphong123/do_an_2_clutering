from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ExperimentConfig
from .pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Vietnamese stock co-movement clustering experiment."
    )
    parser.add_argument("--input", dest="input_path", type=Path, default=None)
    parser.add_argument("--outputs", dest="outputs_dir", type=Path, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--return-type", choices=["log", "simple"], default=None)
    parser.add_argument(
        "--clustering-method",
        choices=["agglomerative", "kmeans"],
        default=None,
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-observation-ratio", type=float, default=None)
    parser.add_argument("--min-average-volume", type=float, default=None)
    parser.add_argument("--forward-fill-limit", type=int, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig()
    values = base.to_jsonable()

    for key in [
        "input_path",
        "outputs_dir",
        "n_clusters",
        "return_type",
        "clustering_method",
        "start_date",
        "end_date",
        "min_observation_ratio",
        "min_average_volume",
        "forward_fill_limit",
    ]:
        value = getattr(args, key)
        if value is not None:
            values[key] = value

    values["input_path"] = Path(values["input_path"])
    values["outputs_dir"] = Path(values["outputs_dir"])
    return ExperimentConfig(**values)


def main() -> int:
    config = config_from_args(parse_args())
    try:
        result = run_experiment(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1

    print("Experiment completed.")
    print(f"Tickers clustered: {result['metrics']['n_tickers']}")
    print(f"Clusters: {result['metrics']['n_clusters']}")
    print(f"Tables: {config.tables_dir}")
    print(f"Figures: {config.figures_dir}")
    print(f"Metrics: {config.metrics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
