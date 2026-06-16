# Agent Instructions

This repository is a Vietnamese stock co-movement clustering research project.

The goal is not only to make the code run. The goal is to produce a clean, reproducible experimental pipeline whose outputs can be used directly in the report section after the theoretical background.

Work carefully. Do not rewrite the whole project blindly. First understand the current research objective, current notebook logic, available data, and report requirements. Then refactor the code into a maintainable pipeline while preserving scientific traceability.

## Project Goal

The project studies clustering of Vietnamese stocks based on price movement and trading behavior. The core research direction is stock co-movement clustering.

Expected research outputs include:

- Clean input data description.
- Feature engineering pipeline.
- Return or log-return construction.
- Similarity or distance matrix construction.
- Clustering model execution.
- Cluster validation and interpretation.
- Experimental tables and figures suitable for the report.
- A reproducible workflow from raw data to final results.

The final implementation must support writing the experimental and results section of the report immediately after the theoretical background.

## Required First Reading

Before editing code, read the following project files if they exist:

- `README.md`
- `code_notebook.md`
- all files under `report/`
- all notebooks under `notebooks/`
- all source files under `src/`
- all data-loading or preprocessing scripts
- all experiment-output or result folders

If a required file is missing, state that it is missing in the work summary. Do not invent its contents.

## Harness

This repo uses Harness. Before work, read:

- `README.md`
- `docs/HARNESS.md`
- `docs/FEATURE_INTAKE.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_RULES.md`
- `docs/TOOL_REGISTRY.md`
- `docs/TEST_MATRIX.md`
- `scripts/bin/harness-cli query matrix` on macOS/Linux, or `.\scripts\bin\harness-cli.exe query matrix` on Windows

Use the Rust Harness CLI at `scripts/bin/harness-cli` on macOS/Linux or `scripts/bin/harness-cli.exe` on Windows as the main operational tool.

On Windows PowerShell, prefer:

```powershell
.\scripts\bin\harness-cli.exe query matrix
.\scripts\bin\harness-cli.exe tool check