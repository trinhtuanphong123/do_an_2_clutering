# Agent Instructions

This repository is a Vietnamese stock co-movement clustering research project.

The main objective is to redesign the current local research code into a clean, reproducible experimental pipeline. The final outputs must support writing the report section “Thực nghiệm và kết quả” immediately after the theoretical background.

Do not treat this as a normal app-only refactor. This is a research-code refactor. Correctness, traceability, reproducibility, and report alignment are more important than adding features quickly.

## Project-Specific Objective

The project studies clustering of Vietnamese stocks based on price movement and trading behavior.

Expected outputs:

- a reproducible data preprocessing pipeline,
- a return or log-return construction pipeline,
- a similarity or distance matrix construction pipeline,
- a clustering pipeline,
- quantitative evaluation tables,
- cluster interpretation tables,
- figures for the report,
- clear experiment artifacts under `outputs/`,
- report-ready notes for the “Thực nghiệm và kết quả” section.

The implementation should make it possible to rerun the experiment and regenerate the results used in the report.

## Required First Reading

Before editing code, inspect these files and folders if they exist:

- `README.md`
- `code_notebook.md`
- `report/`
- `notebooks/`
- `src/`
- `data/`
- `outputs/`
- existing experiment scripts
- existing result tables or figures

If a file or folder is missing, state that it is missing. Do not invent its content.

## Harness

This repo uses Harness. Before work, read:

- `README.md`
- `docs/HARNESS.md`
- `docs/FEATURE_INTAKE.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_RULES.md`
- `docs/TOOL_REGISTRY.md`
- `docs/TEST_MATRIX.md`
- `docs/TRACE_SPEC.md`
- `scripts/bin/harness-cli query matrix` on macOS/Linux, or `.\scripts\bin\harness-cli.exe query matrix` on Windows

Use the Rust Harness CLI at `scripts/bin/harness-cli` on macOS/Linux or `scripts/bin/harness-cli.exe` on Windows as the main operational tool.

On Windows PowerShell, prefer:

```powershell
.\scripts\bin\harness-cli.exe query matrix
.\scripts\bin\harness-cli.exe tool check