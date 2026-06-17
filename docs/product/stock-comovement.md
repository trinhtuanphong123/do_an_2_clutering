# Vietnamese Stock Co-Movement Experiment Contract

## Objective

Build a reproducible research pipeline for clustering Vietnamese stocks by
co-movement in price returns and, when available, trading behavior.

The pipeline must generate artifacts that can support the report section
"Thuc nghiem va ket qua" without depending on notebook state.

## Input Data Contract

The experiment expects a CSV file containing one row per stock per trading date.
Required columns:

- `date`: trading date, parseable by pandas.
- `ticker`: stock symbol.
- `close`: adjusted or unadjusted close price used for return construction.

Optional columns:

- `volume`: trading volume used for liquidity filtering or interpretation.

The repository currently does not contain a Vietnamese stock dataset. The
pipeline must fail clearly when the configured stock input file is missing or
does not satisfy the schema. It must not fabricate stock data or results.

## Experiment Flow

1. Load and validate stock price data.
2. Filter by optional date range and liquidity settings.
3. Pivot prices into a date-by-ticker matrix.
4. Compute simple returns or log returns.
5. Build a ticker-by-ticker correlation distance matrix.
6. Run clustering with configured parameters.
7. Evaluate clusters with metrics supported by the chosen representation.
8. Save report-ready tables, metrics, figures, and run configuration.

## Output Contract

Generated artifacts live under `outputs/`:

- `outputs/tables/`: returns, distance matrix, cluster assignments, summaries.
- `outputs/figures/`: distance heatmap and cluster-size plots.
- `outputs/metrics/`: metrics and run configuration.

Each run overwrites deterministic filenames for the configured experiment.

## Leakage Rules

- No future values may be used to fill earlier dates.
- Backward fill is forbidden.
- Forward fill, if enabled, may only use past observations and must be bounded
  by `forward_fill_limit`.
- Scaling or feature construction must be fit only on the data used by that
  experiment run.
- If a future train/test or rolling-window protocol is added, preprocessing
  parameters must be fit inside each window.

## Non-Goals

- Do not replace the existing credit-card notebooks in this story.
- Do not invent stock data, sectors, labels, benchmark results, or report
  conclusions.
- Do not claim external validation metrics without ground-truth labels.
