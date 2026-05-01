# Results: Utah FORGE p5838

This folder contains all generated outputs from the Utah FORGE p5838 stick-slip SINDy analysis.

---

## What is Final?

The following files represent the final, citeable results:

| File | Description |
|------|-------------|
| `p5838_final_report.md` | **Core results report** |
| `best_equations_showcase.md` | **Discovered equations with interpretation** |
| `p5838_paper_section.md` | **Paper-ready section draft** |
| `project_performance_assessment.md` | **Comprehensive performance summary** |
| `p5838_final_equations.txt` | **Best discovered equation strings** |
| `best_equations_showcase_equations.txt` | **Sparse equation strings** |
| `showcase_tau_fit.png` | **Final tau equation fit figure** |
| `showcase_velocity_fit.png` | **Final velocity equation fit figure** |
| `showcase_phaseplot.png` | **Final phase plot** |
| `multistep_tau_rollout_gallery.png` | **Multi-step tau rollout figure** |
| `multistep_velocity_rollout_gallery.png` | **Multi-step velocity rollout figure** |
| `overall_performance_comparison.png` | **Model comparison summary** |

---

## What is Historical / Exploratory?

All other files are preserved for **iteration history** and should not be cited as final results. They document the thought process, model iterations, and diagnostic analyses.

Key historical/exploratory categories:
- `baseline_*` — polynomial baseline iteration
- `augmented_*` — memory-augmented model iteration
- `model_BC_*` — B/C ablation iteration
- `model_C_*` — velocity isolation iteration
- `proposal_equation_*` — proposal-stage results
- `p5838_memory_*` — memory model exploration
- `p5838_refinement_*` — refinement iteration

---

## Which Metrics Should Be Cited?

For the tau equation:
- `project_performance_assessment.json` — primary performance table
- `tau_all_splits_assessment.json` — cross-split metrics
- `multistep_rollout_summary.json` — rollout validation

For the velocity equation:
- `v_reduced_summary.json` — reduced velocity law metrics
- `v_exact_summary.json` — exact velocity model metrics

For RSF identifiability:
- `exact_rsf_multistart_summary.json` — parameter instability evidence
- `exact_rsf_identifiability_summary.md` — narrative summary

---

## Which Figures Should Be Used?

For the final paper/report, use:
- `showcase_tau_fit.png` (tau derivative prediction)
- `showcase_velocity_fit.png` (velocity derivative prediction)
- `showcase_phaseplot.png` (phase plot)
- `multistep_tau_rollout_gallery.png` (rollout validation)
- `overall_performance_comparison.png` (model comparison)
- `tau_leave_two_out_heatmap.png` (cross-validation)
- `conditional_v_diagnostic_report.md` (with figures) for regime analysis

---

## Folder Structure (Generated Plot Galleries)

Subfolders contain plot galleries from specific analysis scripts:

| Folder | Description |
|--------|-------------|
| `p5838_delay_rollouts/` | Delay model rollout plots (exploratory) |
| `p5838_delay_validation_plots/` | Delay validation (exploratory) |
| `p5838_derivative_diagnostics/` | Derivative quality analysis plots |
| `p5838_memory_feature_plots/` | Memory feature visualization |
| `p5838_memory_rollouts/` | Memory model rollout plots |
| `p5838_memory_theta_plots/` | Memory/theta comparison |
| `p5838_rsfit_theta_plots/` | RSF fit theta reconstruction |
| `p5838_theta_comparison_plots/` | Theta comparison |
| `p5838_long_rollouts/` | Long multi-step rollout gallery |
| `derivative_diagnostics/` | Derivative method diagnostics |
| `theta_reconstruction_plots/` | Theta reconstruction attempts |
| `top_model_rollouts/` | Top model rollout gallery |

---

## Large Files Note

- `sindy_sweep_results.csv` (75 MB): Full hyperparameter sweep results. Large but documents the sparsity frontier exploration.
- `selected_cycle_*.csv`: Preprocessed cycle data (1-10 MB each). Committed for use without raw data.
- `conditional_v_diagnostic_report.json` (3 MB): Full diagnostic results.
- `model_BC_tau_fix_comparison.json` (7 MB): Full B/C comparison data.

All other JSON/CSV files are < 1 MB.
