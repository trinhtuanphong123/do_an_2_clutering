# US-001 Reproducible Stock Co-Movement Experiment Pipeline

## Overview

Before US-001, the repository contained notebook-exported credit-card
segmentation code and no reusable stock co-movement pipeline.

Target behavior: the repository provides one reusable, configurable experiment
entry point that can load real Vietnamese stock data, construct returns and a
correlation distance matrix, cluster tickers, evaluate the result, and save
report-ready artifacts under `outputs/`.

Affected users: researcher writing the "Thuc nghiem va ket qua" report section.

Affected product docs: `docs/product/stock-comovement.md`.

Non-goals: delete existing notebooks/report files, fabricate missing stock data,
or generate report conclusions without real experiment data.

## Design

Domain model:

- Stock observation: `(date, ticker, close, optional volume)`.
- Price matrix: rows are dates, columns are tickers.
- Return matrix: rows are dates, columns are tickers.
- Distance matrix: rows and columns are tickers.
- Cluster assignment: ticker-to-cluster mapping.

Application flow:

1. `ExperimentConfig` defines paths and modeling parameters.
2. `load_stock_data` validates the configured CSV.
3. `prepare_price_matrix` filters and pivots prices.
4. `compute_returns` creates simple or log returns.
5. `correlation_distance` builds a ticker distance matrix.
6. `cluster_from_distance` assigns clusters.
7. `evaluate_clustering` computes available internal metrics.
8. `run_experiment` writes tables, metrics, figures, and config.

Interface contract: run with `python -m src.run_experiment --input <stock_csv>`.
Without a valid stock input, the command exits with a clear error and does not
write fabricated outputs.

Data model: CSV-only input and file artifacts only. No database or migration.

Platform impact: command-line research workflow.

Observability: run configuration and metrics JSON are saved under
`outputs/metrics/`.

Alternatives considered:

1. Keep logic inside notebooks. Rejected because notebook state is not a
   reproducible experiment contract.
2. Convert existing credit-card pipeline directly. Rejected because the target
   domain is stock co-movement and requires time-series returns and distances.

## Exec Plan

Goal: create the smallest safe stock experiment pipeline without producing fake
results.

Scope:

- In scope: modules, CLI entry point, deterministic output directories, unit
  tests on synthetic in-memory/temp-file stock data.
- Out of scope: real Vietnamese stock dataset, sector metadata, report writing,
  and deletion of old notebook/report artifacts.

Harness lane: normal.

Risk drivers:

- Existing behavior.
- Weak proof until real stock input data is supplied.

Hard gates:

- Do not invent data or results.
- Do not use future data in preprocessing.

Work phases:

1. Define product contract.
2. Add pipeline modules.
3. Add CLI entry point.
4. Add deterministic tests.
5. Run compile/tests/entry point.
6. Summarize generated artifacts and residual risks.

Stop conditions:

- Real stock data schema is ambiguous.
- A requested validation requires fabricated data.
- Existing notebooks/report files would need deletion.

## Validation

Proof strategy: validate pure functions and full pipeline behavior on temporary
synthetic stock data, then run compile checks and the real entry point against
the default missing stock path to confirm it fails safely.

Test plan:

| Layer | Cases |
| --- | --- |
| Unit | Return construction, distance matrix shape/diagonal, clustering output size |
| Integration | Temporary stock CSV produces tables, figures, and metrics |
| E2E | CLI exits clearly when configured stock input is missing and does not fabricate outputs |
| Platform | Windows PowerShell command invocation |
| Performance | Not in scope for first slice |
| Logs/Audit | Metrics and config JSON saved for successful runs |

Fixtures: tests create temporary stock observations for four synthetic tickers.

Commands:

```text
python -m compileall src
python -m unittest discover -s tests
python -m src.run_experiment
```

Acceptance evidence:

- `python -m compileall src`: passed with escalated system Python after the
  sandbox Python shim failed to spawn.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests`: passed, 3 tests.
- `.\.venv\Scripts\python.exe -m src.run_experiment`: ran the entry point and
  failed safely because `data/vietnam_stocks.csv` is not present. No real stock
  result artifacts were generated. Output directories are created only after
  data has been loaded, transformed, clustered, and evaluated successfully.
