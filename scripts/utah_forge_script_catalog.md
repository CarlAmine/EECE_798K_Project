# Utah FORGE Script Catalog

This catalog groups Utah FORGE scripts by role so we can migrate safely without breaking existing paths.

## Keep as primary entrypoints (packaging/final outputs)
- `utah_forge_finalize_project_package.py`
- `utah_forge_finalv2_refresh_package.py`
- `utah_forge_v_package_finalv4.py`

## Evaluation / assessment
- `utah_forge_project_performance_assessment.py`
- `utah_forge_tau_all_splits_assessment.py`
- `utah_forge_v_all_splits_assessment.py`
- `utah_forge_v_reduced_all_splits.py`
- `utah_forge_v_exact_selected_splits.py`
- `utah_forge_regime_balanced_tau_evaluation.py`

## Model and equation analysis
- `utah_forge_proposal_equation_recovery.py`
- `utah_forge_proposal_equation_robustness.py`
- `utah_forge_theta_equation_consistency.py`
- `utah_forge_sparsity_frontier.py`
- `utah_forge_augmented_theta.py`
- `utah_forge_model_bc_tau_fix_comparison.py`
- `utah_forge_model_c_velocity_isolation.py`

## Diagnostics / visualization
- `utah_forge_conditional_v_diagnostic.py`
- `utah_forge_conditional_v_visualize.py`
- `utah_forge_rollout_metric_explainer.py`
- `utah_forge_step_variability_diagnostics.py`
- `utah_forge_showcase_fit_visuals.py`
- `utah_forge_multistep_rollout_summary.py`

## RSF focused experiments
- `utah_forge_exact_rsf_inverse_fit.py`
- `utah_forge_exact_rsf_multistart_check.py`
- `utah_forge_exact_rsf_showcase.py`

## One-off / historical candidates for archive
- `utah_forge_reviewer_ablation.py`
- `utah_forge_memory_refinement.py`
- `utah_forge_finalv2_alternative_holdouts.py`
- `utah_forge_regime_analysis.py`

## Recommended next migration step
1. Add `scripts/utah_forge/` subfolders (`assessment`, `analysis`, `diagnostics`, `rsf`, `legacy`).
2. Move one category at a time.
3. Leave wrapper scripts at old paths that call new locations and print deprecation notices.
4. Add each moved entrypoint to `scripts/tools/check_script_entrypoints.py` in small batches.
