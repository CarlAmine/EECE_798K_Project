# Results Index

This document indexes all result files in `results/utah_forge/`, distinguishing between final, exploratory, and archival outputs.

---

## Final Results (Cite These)

### Core Reports
| File | Description |
|------|-------------|
| `results/utah_forge/p5838_final_report.md` | Main results report |
| `results/utah_forge/best_equations_showcase.md` | Discovered equations with interpretation |
| `results/utah_forge/p5838_paper_section.md` | Paper-ready section draft |
| `results/utah_forge/project_performance_assessment.md` | Comprehensive performance summary |

### Final Equations
| File | Description |
|------|-------------|
| `results/utah_forge/p5838_final_equations.txt` | Best discovered equations |
| `results/utah_forge/best_equations_showcase.json` | Equations with metadata |
| `results/utah_forge/best_sparse_equations.txt` | Sparsest recovered equations |
| `results/utah_forge/best_equations_showcase_equations.txt` | Equation strings |

### Final Figures (Key)
| File | Description |
|------|-------------|
| `results/utah_forge/showcase_tau_fit.png` | Tau equation fit |
| `results/utah_forge/showcase_velocity_fit.png` | Velocity equation fit |
| `results/utah_forge/showcase_phaseplot.png` | Phase plot (τ vs V) |
| `results/utah_forge/showcase_exact_rsf_fit.png` | Exact RSF fit quality |
| `results/utah_forge/showcase_derivative_scatter.png` | Derivative prediction scatter |
| `results/utah_forge/multistep_tau_rollout_gallery.png` | Multi-step tau rollout |
| `results/utah_forge/multistep_velocity_rollout_gallery.png` | Multi-step velocity rollout |
| `results/utah_forge/multistep_phaseplots.png` | Phase plots across cycles |
| `results/utah_forge/overall_performance_comparison.png` | Model comparison summary |
| `results/utah_forge/tau_leave_two_out_heatmap.png` | Tau cross-validation heatmap |

### Final Metrics
| File | Description |
|------|-------------|
| `results/utah_forge/project_performance_assessment.json` | Numerical performance metrics |
| `results/utah_forge/multistep_rollout_summary.json` | Rollout validation metrics |
| `results/utah_forge/tau_all_splits_assessment.json` | Tau cross-split metrics |
| `results/utah_forge/v_reduced_summary.json` | Reduced velocity model metrics |
| `results/utah_forge/exact_rsf_multistart_summary.json` | Exact RSF identifiability metrics |

---

## Important Diagnostic Results

| File | Description |
|------|-------------|
| `results/utah_forge/conditional_v_diagnostic_HONEST_INTERPRETATION.md` | Honest analysis of velocity results |
| `results/utah_forge/conditional_v_diagnostic_report.md` | Conditional velocity detailed report |
| `results/utah_forge/regime_balanced_tau_evaluation.md` | Regime mismatch analysis |
| `results/utah_forge/holdout_shift_assessment.md` | Train/holdout regime shift |
| `results/utah_forge/equation1_vs_equation2_forensic_audit.md` | Equation variant comparison |
| `results/utah_forge/exact_rsf_identifiability_summary.md` | RSF parameter identifiability |
| `results/utah_forge/theta_equation_consistency_report.md` | Theta consistency check |
| `results/utah_forge/sparsity_frontier_report.md` | Sparsity vs accuracy tradeoff |

---

## Preprocessed Data (Committed for Convenience)

| File | Size | Description |
|------|------|-------------|
| `results/utah_forge/selected_cycle_short.csv` | ~1.1 MB | Short stick-slip cycle |
| `results/utah_forge/selected_cycle_medium.csv` | ~6.1 MB | Medium stick-slip cycle |
| `results/utah_forge/selected_cycle_long.csv` | ~10.3 MB | Long stick-slip cycle |

**Note:** `sindy_sweep_results.csv` (75 MB) is large and borderline for git. It documents the full hyperparameter sweep.

---

## Historical/Exploratory Results (Preserved)

The following files document the iteration history. They are preserved but should not be cited as final results.

### Earlier Model Iterations
| File | Description |
|------|-------------|
| `results/utah_forge/baseline_summary.json` | Polynomial baseline metrics |
| `results/utah_forge/baseline_rollout.png` | Polynomial baseline rollout |
| `results/utah_forge/augmented_model_summary.json` | Memory-augmented model metrics |
| `results/utah_forge/augmented_vs_baseline_report.md` | Memory vs baseline comparison |
| `results/utah_forge/model_BC_tau_fix_comparison.json` | B/C comparison data |
| `results/utah_forge/model_C_velocity_isolation_comparison.json` | Model C velocity isolation |
| `results/utah_forge/p5838_memory_model_summary.json` | Memory model metrics |
| `results/utah_forge/p5838_physics_informed_summary.json` | Physics-informed model metrics |

### Proposal-Stage Results
| File | Description |
|------|-------------|
| `results/utah_forge/proposal_equation_recovery.json` | Proposal-stage equation recovery |
| `results/utah_forge/proposal_equation_recovery_report.md` | Proposal-stage report |
| `results/utah_forge/proposal_equation_identifiability.json` | Identifiability analysis |
| `results/utah_forge/proposal_equation_robustness_check.md` | Robustness check |

### Plot Galleries (Historical)
| Folder | Description |
|--------|-------------|
| `results/utah_forge/p5838_delay_rollouts/` | Delay model rollout plots |
| `results/utah_forge/p5838_delay_validation_plots/` | Delay model validation |
| `results/utah_forge/p5838_derivative_diagnostics/` | Derivative quality plots |
| `results/utah_forge/p5838_memory_feature_plots/` | Memory feature visualizations |
| `results/utah_forge/p5838_memory_rollouts/` | Memory model rollouts |
| `results/utah_forge/p5838_memory_theta_plots/` | Memory/theta comparison |
| `results/utah_forge/p5838_rsfit_theta_plots/` | RSF fit theta plots |
| `results/utah_forge/p5838_theta_comparison_plots/` | Theta comparison plots |
| `results/utah_forge/derivative_diagnostics/` | Derivative diagnostics |
| `results/utah_forge/theta_reconstruction_plots/` | Theta reconstruction |
| `results/utah_forge/top_model_rollouts/` | Top model rollout plots |
| `results/utah_forge/p5838_long_rollouts/` | Long multi-step rollout plots |
| `results/utah_forge/p5838_delay_validation_plots/` | Delay validation plots |

---

## Results in Other Datasets

| Folder | Description |
|--------|-------------|
| `results/lanl/` | LANL dataset results (secondary) |
| `results/pangaea/` | PANGAEA dataset results (secondary) |
| `results/fdem_zenodo/` | FDEM Zenodo results (exploratory) |
| `results/multidataset_status.md` | Cross-dataset status summary |
