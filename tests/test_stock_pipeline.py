from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import ExperimentConfig
from src.data import load_stock_data, prepare_price_matrix
from src.features import compute_returns, correlation_distance
from src.pipeline import run_experiment


class StockPipelineTests(unittest.TestCase):
    def write_stock_csv(self, path: Path) -> None:
        rows = []
        dates = [f"2026-01-{day:02d}" for day in range(1, 8)]
        series = {
            "AAA": [10, 11, 12, 13, 14, 15, 16],
            "BBB": [20, 22, 24, 26, 28, 30, 32],
            "CCC": [30, 29, 28, 27, 26, 25, 24],
            "DDD": [40, 39, 38, 39, 40, 41, 42],
        }
        for ticker, prices in series.items():
            for date, close in zip(dates, prices):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "close": close,
                        "volume": 1000,
                    }
                )

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "ticker", "close", "volume"])
            writer.writeheader()
            writer.writerows(rows)

    def test_return_and_distance_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stocks.csv"
            self.write_stock_csv(csv_path)
            config = ExperimentConfig(input_path=csv_path, n_clusters=2)

            raw = load_stock_data(csv_path, config)
            prices = prepare_price_matrix(raw, config)
            returns = compute_returns(prices, "log")
            distance = correlation_distance(returns)

            self.assertEqual(set(distance.index), {"AAA", "BBB", "CCC", "DDD"})
            np.testing.assert_allclose(np.diag(distance.values), 0.0)
            self.assertFalse(distance.isna().any().any())

    def test_full_experiment_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "stocks.csv"
            outputs = tmp_path / "outputs"
            self.write_stock_csv(csv_path)

            config = ExperimentConfig(
                input_path=csv_path,
                outputs_dir=outputs,
                n_clusters=2,
                figure_dpi=80,
            )
            result = run_experiment(config)

            self.assertEqual(result["metrics"]["n_tickers"], 4)
            self.assertTrue((outputs / "tables" / "returns.csv").exists())
            self.assertTrue((outputs / "tables" / "distance_matrix.csv").exists())
            self.assertTrue((outputs / "tables" / "cluster_assignments.csv").exists())
            self.assertTrue((outputs / "figures" / "distance_heatmap.png").exists())
            self.assertTrue((outputs / "figures" / "cluster_sizes.png").exists())

            metrics = json.loads((outputs / "metrics" / "metrics.json").read_text())
            self.assertEqual(metrics["n_clusters"], 2)

    def test_missing_input_does_not_create_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outputs = tmp_path / "outputs"
            config = ExperimentConfig(
                input_path=tmp_path / "missing_stocks.csv",
                outputs_dir=outputs,
            )

            with self.assertRaises(FileNotFoundError):
                run_experiment(config)

            self.assertFalse(outputs.exists())


if __name__ == "__main__":
    unittest.main()
