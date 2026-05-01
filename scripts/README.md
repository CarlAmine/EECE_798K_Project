# Scripts

This folder contains all experiment scripts for the Utah FORGE stick-slip SINDy project.

---

## Quick Start

```bash
# Check environment and list scripts
python scripts/run_final_pipeline.py --list

# Run smoke test (no data required)
python scripts/smoke_test.py

# Run full final pipeline (requires raw data)
python scripts/run_final_pipeline.py
```

---

## Script Catalog

### FINAL Scripts (run these for final results)

| Script | Phase | Description |
|--------|-------|-------------|
| `run_final_pipeline.py` | Entry point | Wrapper that orchestrates the full pipeline |
| `utah_forge_proposal_equation_recovery.py` | 01 Baseline | Main SINDy: polynomial baseline + physics-informed library |
| `utah_forge_reviewer_ablation.py` | 02 Ablation | A/B/C model comparison (observed-only / memory / theta) |
| `utah_forge_tau_all_splits_assessment.py` | 03 Tau CV | Tau equation cross-validation across all train/holdout splits |
| `utah_forge_v_all_splits_assessment.py` | 04 Velocity CV | Velocity equation cross-validation |
| `utah_forge_multistep_rollout_summary.py` | 05 Rollout | Multi-step rollout validation summary |
| `utah_forge_exact_rsf_showcase.py` | 06 Exact RSF | Exact RSF inverse fitting showcase |

### IMPORTANT DIAGNOSTIC Scripts

| Script | Description |
|--------|-------------|
| `utah_forge_regime_analysis.py` | Regime mismatch: why holdout differs from training |
| `utah_forge_regime_balanced_tau_evaluation.py` | Regime-balanced tau evaluation |
| `utah_forge_conditional_v_diagnostic.py` | Velocity conditioned on tau regime |
| `utah_forge_conditional_v_visualize.py` | Visualization companion for conditional V |
| `utah_forge_rollout_metric_explainer.py` | Explains rollout metric interpretation |
| `utah_forge_step_variability_diagnostics.py` | Step-level variability analysis |

### EXPLORATORY Scripts (preserved for history)

| Script | Description |
|--------|-------------|
| `utah_forge_sparsity_frontier.py` | Sparsity vs accuracy tradeoff |
| `utah_forge_memory_refinement.py` | Memory feature (tau_avg, tau_ema) experiments |
| `utah_forge_augmented_theta.py` | Theta surrogate augmentation |
| `utah_forge_model_c_velocity_isolation.py` | Velocity isolation for Model C |
| `utah_forge_model_bc_tau_fix_comparison.py` | B/C model tau comparison |
| `utah_forge_exact_rsf_inverse_fit.py` | Exact RSF inverse fitting exploration |
| `utah_forge_exact_rsf_multistart_check.py` | RSF parameter identifiability check |
| `utah_forge_theta_equation_consistency.py` | Theta ODE consistency check |
| `utah_forge_showcase_fit_visuals.py` | Fit visualization scripts |
| `utah_forge_v_reduced_all_splits.py` | Reduced velocity law across splits |
| `utah_forge_v_exact_selected_splits.py` | Exact velocity law on selected splits |
| `utah_forge_proposal_equation_robustness.py` | Robustness check on proposal equations |
| `run_fdem_zenodo_sindy.py` | FDEM Zenodo dataset SINDy (exploratory) |

### OBSOLETE Scripts (preserved for iteration history)

| Script | Why Obsolete |
|--------|-------------|
| `utah_forge_v_package_finalv4.py` | Superseded by final packaging |
| `utah_forge_finalv2_alternative_holdouts.py` | v2 iteration, superseded |
| `utah_forge_finalv2_refresh_package.py` | v2 packaging, superseded |
| `utah_forge_finalize_project_package.py` | v1 packaging, superseded |
| `assemble_utah_forge_final_report.py` | Assembly iteration, superseded |
| `fix_utah_forge_final_figures.py` | Figure fix iteration, superseded |
| `improve_utah_forge_model.py` | Model improvement iteration, superseded |
| `refine_utah_forge_validation.py` | Validation refinement iteration, superseded |

### Utilities

| Script | Description |
|--------|-------------|
| `smoke_test.py` | Import and directory verification |
| `build_multidataset_notebooks.py` | Generates multi-dataset notebook templates |
| `run_notebooks.py` | Notebook execution runner |
| `export_utah_forge_table.m` | MATLAB table export helper |
| `utah_forge_script_catalog.md` | Legacy script catalog (superseded by this README) |

---

## Outputs

All scripts write outputs to `results/utah_forge/`. Key outputs:

- `*.md` files: human-readable reports
- `*.json` files: machine-readable metrics and equation records
- `*.png` files: figures (fit quality, rollouts, phase plots, heatmaps)
- `*.txt` files: equation strings
- `*.csv` files: preprocessed data and sweep results

---

## Notes

- Run all scripts from the **repository root**, not from within `scripts/`.
- Raw `.mat` data must be present in `data/utah_forge/` for most scripts to run.
- Preprocessed CSVs in `results/utah_forge/` allow some analysis without raw data.
- See [`docs/reproducibility.md`](../docs/reproducibility.md) for full environment setup.
