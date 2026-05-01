# Project Iteration History

This document narrates the project progression from proposal to final report. It is intended to help a grader or future researcher understand the thought process behind the model choices.

---

## Phase 0: Proposal

**Goal:** Identify a feasible data-driven governing equation recovery problem.

**Decision:** Utah FORGE p5838 stick-slip data is the best candidate because:
- Clear ODE structure (RSF) is known from physics
- Data is well-instrumented
- Natural segmentation into cycles allows cross-validation

**Output:** Project proposal; initial data exploration (`explore_data.py`).

---

## Phase 1: Dataset Survey and Baseline

**Scripts:** `utah_forge_proposal_equation_recovery.py` (early versions), initial notebooks

**What was done:**
- Loaded and parsed Utah FORGE p5838 `.mat` files
- Implemented stick-slip cycle segmentation (`src/segmentation/`)
- Implemented derivative estimation (`src/derivatives.py`)
- Built polynomial SINDy baseline
- Established evaluation protocol: leave-one-out on cycles, train/holdout split

**Key finding:** Polynomial SINDy recovers the tau equation reasonably but the velocity equation is noisy.

---

## Phase 2: Physics-Informed Library Construction

**Scripts:** `utah_forge_proposal_equation_recovery.py` (final version), `utah_forge_proposal_equation_robustness.py`

**What was done:**
- Replaced polynomial library with RSF-motivated library (τ, V, V_drive, V_drive−V, log terms)
- Applied STLSQ sparse regression with hyperparameter sweep
- Validated robustness across splits

**Key finding:** **dτ/dt ≈ k(V_drive − V)** is robustly recovered. k ≈ physical machine stiffness.

---

## Phase 3: Memory and State Augmentation (Model B and C Ablation)

**Scripts:** `utah_forge_memory_refinement.py`, `utah_forge_augmented_theta.py`, `utah_forge_reviewer_ablation.py`

**Motivation:** RSF state θ might be approximated by memory surrogates (τ_avg, τ_ema).

**What was done:**
- Added rolling mean and exponential moving average of τ as candidate features
- Constructed θ surrogate from integrated RSF aging law
- Ran full A/B/C ablation study (Model A = observed only, B = +memory, C = +theta)

**Key finding:** Memory improves rollout slightly but is not physically meaningful. θ surrogates are too correlated with τ to be independently identified.

---

## Phase 4: Tau Isolation and Spring Law Confirmation

**Scripts:** `utah_forge_model_bc_tau_fix_comparison.py`, `utah_forge_tau_all_splits_assessment.py`

**What was done:**
- Isolated the tau equation and tested specifically for the spring-slider form
- Cross-validated across all train/holdout splits
- Generated the leave-two-out heatmap

**Key finding (confirmed):** dτ/dt = k(V_drive − V) is the most robustly identified result. R² > 0.99 across all splits.

---

## Phase 5: Reduced Velocity Law and Conditional Diagnostics

**Scripts:** `utah_forge_v_reduced_all_splits.py`, `utah_forge_v_all_splits_assessment.py`, `utah_forge_conditional_v_diagnostic.py`

**What was done:**
- Focused on velocity equation using RSF-motivated features: log(V), τ/V, etc.
- Ran conditional (regime-split) diagnostics to test if different regimes need different models
- Generated velocity cross-validation heatmaps

**Key finding:** Velocity equation shows log(V) structure in some splits but is not stable across holdouts. Regime mismatch is a primary explanation.

---

## Phase 6: Exact RSF Inverse Fitting

**Scripts:** `utah_forge_exact_rsf_inverse_fit.py`, `utah_forge_exact_rsf_showcase.py`, `utah_forge_exact_rsf_multistart_check.py`

**What was done:**
- Implemented full nonlinear RSF parameter estimation (a, b, D_c, µ₀, k)
- Multi-start optimization to explore parameter space
- Analyzed identifiability and parameter stability

**Key finding (negative):** Exact RSF fitting is parameter unstable. Multiple local optima exist. This is expected from theory — the problem is ill-posed without independent θ measurements.

---

## Phase 7: Regime Analysis and Diagnostics

**Scripts:** `utah_forge_regime_analysis.py`, `utah_forge_regime_balanced_tau_evaluation.py`, `utah_forge_step_variability_diagnostics.py`

**What was done:**
- Statistical analysis of τ and V distributions across cycles
- Comparison of train vs holdout feature spaces
- Step difficulty analysis (which cycles are hardest to predict)

**Key finding:** Training and holdout cycles occupy different regions of the (τ, V) phase space. This regime heterogeneity is the primary explanation for reduced holdout performance — not model failure.

---

## Phase 8: Final Report Assembly

**Scripts:** `assemble_utah_forge_final_report.py`, `utah_forge_finalize_project_package.py`, `fix_utah_forge_final_figures.py` (all now obsolete)

**What was done:**
- Multiple packaging attempts (Finalv1 through Finalv5 folders — now cleaned up/archived)
- Final results consolidated in `results/utah_forge/p5838_final_report.md`
- Final equations in `results/utah_forge/best_equations_showcase.md`

---

## Phase 9: Repository Cleanup (This PR)

**Branch:** `repo-cleanup/readability-pass`

**What was done:**
- Created comprehensive `docs/` folder with 9 documentation files
- Replaced minimal README with full project landing page
- Added `requirements.txt`
- Added `scripts/run_final_pipeline.py` wrapper
- Added `scripts/smoke_test.py`
- Updated `results/utah_forge/README.md`
- Added `reports/README.md`
- No source code was changed
- No files were deleted
- `malek-utah-forge` was not touched
