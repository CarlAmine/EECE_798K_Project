# EECE 798K Project Repository Overview

This repository is a Python-based equation-discovery and validation workflow focused on frictional system dynamics across multiple datasets (`lanl`, `pangaea`, `utah_forge`, and `fdem_zenodo`).

## Top-Level Layout

- `src/`: reusable library code for data loading, preprocessing, segmentation, derivatives, sparse model fitting, metrics, and exact-RSF utilities.
- `scripts/`: runnable experiment and packaging scripts, especially a large Utah FORGE analysis pipeline.
- `data/`: raw/source dataset folders and dataset-specific README notes.
- `notebooks/`: generated and/or curated notebooks for load -> preprocess/segment -> SINDy baseline workflows.
- `results/`: generated reports, figures, checkpoints, comparisons, and packaged final artifacts (`Final`, `Finalv2`...`Finalv5`).
- root files: minimal `README.md`, `.gitignore`, and utility scripts.

## Core Source Package (`src`)

### Configuration and Paths

- `src/config.py`: dataset metadata and per-dataset processing defaults.
- `src/utils/paths.py`: repository-relative path helpers and directory creation utilities.

### Data I/O

- `src/io/lanl.py`
- `src/io/pangaea.py`
- `src/io/utah_forge.py`
- `src/io/load_fdem_zenodo.py`
- `src/io/__init__.py`: shared validation and convenience exports.

These modules locate raw files, harmonize columns, and expose consistent loading interfaces.

### Preprocessing

- `src/preprocess/common.py`: cleaning, smoothing, normalization, velocity derivation, canonical frame preparation.
- `src/preprocess/lanl.py`
- `src/preprocess/pangaea.py`
- `src/preprocess/utah_forge.py`

### Segmentation

- `src/segmentation/common.py`: generic segmentation and summary logic.
- `src/segmentation/lanl.py`
- `src/segmentation/pangaea.py`
- `src/segmentation/utah_forge.py`

### Modeling and Metrics

- `src/sindy/library.py`: feature-library construction.
- `src/sindy/models.py`: sparse regression / SINDy model fitting.
- `src/sindy/metrics.py`: fit and rollout metrics.
- `src/derivatives.py`: derivative estimation methods (finite-difference, Savitzky-Golay, spline style helpers).

### Dataset Summaries and Exact RSF

- `src/datasets/*`: dataset summary/packaging helpers.
- `src/exact_rsf.py`: exact RSF inverse-fitting, simulation, checkpointing, and diagnostics.

## Scripts (`scripts/`) by Workflow Stage

### Notebook Workflow

- `build_multidataset_notebooks.py`: creates standard notebook triplets per dataset.
- `run_notebooks.py`: executes notebooks end-to-end.

### Baseline / General Runs

- `run_fdem_zenodo_sindy.py`: baseline SINDy run for FDEM Zenodo data.

### Utah FORGE Main Pipeline

Representative phases include:

- preparation and model refinement (`improve_utah_forge_model.py`, `refine_utah_forge_validation.py`, `utah_forge_memory_refinement.py`)
- proposal equation recovery and robustness (`utah_forge_proposal_equation_recovery.py`, `utah_forge_proposal_equation_robustness.py`)
- exact-RSF inverse and multistart checks (`utah_forge_exact_rsf_inverse_fit.py`, `utah_forge_exact_rsf_multistart_check.py`, `utah_forge_exact_rsf_showcase.py`)
- diagnostics and split studies (`utah_forge_tau_all_splits_assessment.py`, `utah_forge_v_all_splits_assessment.py`, `utah_forge_v_reduced_all_splits.py`, `utah_forge_v_exact_selected_splits.py`, `utah_forge_step_variability_diagnostics.py`, `utah_forge_rollout_metric_explainer.py`)
- conditional/ablation analyses (`utah_forge_conditional_v_diagnostic.py`, `utah_forge_model_bc_tau_fix_comparison.py`, `utah_forge_model_c_velocity_isolation.py`, `utah_forge_theta_equation_consistency.py`)
- final report packaging (`assemble_utah_forge_final_report.py`, `utah_forge_finalize_project_package.py`, `utah_forge_v_package_finalv4.py`, `utah_forge_finalv2_refresh_package.py`)

## Data and Results Organization

- `data/` contains large source artifacts and dataset-scoped subfolders.
- `results/utah_forge/` is the most extensive result tree:
  - checkpoint folders
  - many diagnostics in `.json` / `.md` / `.csv`
  - rollout/phase/scatter galleries in `.png`
  - packaged outputs (`Final`, `Finalv2`, `Finalv3`, `Finalv4`, `Finalv5`) and zip bundles
- `results/lanl/` and `results/pangaea/` contain dataset-specific outputs with a smaller footprint.

## Notebooks

Canonical dataset notebooks exist under:

- `notebooks/lanl/`
- `notebooks/pangaea/`
- `notebooks/utah_forge/`

Each dataset generally follows:

1. `00_load_*.ipynb`
2. `01_preprocess_and_segment_*.ipynb`
3. `02_sindy_baseline_*.ipynb`

Additional root-level notebooks appear to be convenience or copied/generated variants.

## Entry Points

The primary execution entry points are script-level `main()` functions in `scripts/` (invoked via `python scripts/<name>.py`).
Most heavy workflows are Utah FORGE scripts, with notebook generation/execution as a secondary reproducibility route.

## Dependency/Packaging Notes

The repository currently does not expose a formal dependency manifest (`pyproject.toml`, `requirements.txt`, etc.) in root.
Imports indicate reliance on:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- optional use of `scikit-learn` in some diagnostics

For reproducibility, adding an explicit environment specification would be beneficial.
