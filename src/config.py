from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    input_path: Path = Path("data/vietnam_stocks.csv")
    outputs_dir: Path = Path("outputs")
    date_col: str = "date"
    ticker_col: str = "ticker"
    price_col: str = "close"
    volume_col: str | None = "volume"
    start_date: str | None = None
    end_date: str | None = None
    return_type: str = "log"
    min_observation_ratio: float = 0.8
    min_average_volume: float | None = None
    forward_fill_limit: int = 0
    n_clusters: int = 4
    clustering_method: str = "agglomerative"
    random_state: int = 42
    figure_dpi: int = 150

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data

