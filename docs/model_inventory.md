# Model Inventory

This document catalogs all model families explored in the project, with scripts, features, target equations, main results, and rationale.

All models are applied to the Utah FORGE p5838 stick-slip dataset unless otherwise noted.

---

## Model Summary Table

| # | Model Name | Script(s) | Library/Features | Equation Target | Main Result | Kept / Superseded |
|---|-----------|-----------|-----------------|-----------------|-------------|-------------------|
| 1 | Polynomial SINDy baseline | `utah_forge_proposal_equation_recovery.py` | Polynomial (degree 1-3) | dτ/dt, dV/dt | Tau law recovered; velocity poor | Superseded by physics library |
| 2 | Observed-only RSF (Model A) | `utah_forge_proposal_equation_recovery.py` | τ, V, V_drive, τ², V², τV | dτ/dt, dV/dt | Tau: excellent (R²>0.99); V: partial | **Kept as primary baseline** |
| 3 | Memory-augmented (Model B) | `utah_forge_memory_refinement.py`, `utah_forge_reviewer_ablation.py` | + τ_avg, τ_ema, rolling std | dτ/dt, dV/dt | Improved rollout; memory not physically interpretable | **Kept as ablation** |
| 4 | Theta-proxy (Model C) | `utah_forge_augmented_theta.py`, `utah_forge_reviewer_ablation.py` | + θ surrogate, dθ/dt estimate | dτ/dt, dV/dt | Marginal improvement; θ not identifiable | **Kept as ablation** |
| 5 | Tau-spring law | `utah_forge_model_bc_tau_fix_comparison.py` | τ, V_drive-V specifically | dτ/dt | dτ/dt = k(V_drive-V) cleanly recovered | **Key finding** |
| 6 | Reduced velocity law | `utah_forge_v_reduced_all_splits.py`, `utah_forge_v_all_splits_assessment.py` | log(V), τ, V | dV/dt | Partial log(V) structure recovered | **Kept as main velocity result** |
| 7 | Conditional velocity variants | `utah_forge_conditional_v_diagnostic.py`, `utah_forge_conditional_v_visualize.py` | V conditioned on τ regime | dV/dt | Regime-specific equations differ significantly | **Important diagnostic** |
| 8 | Exact RSF inverse fit | `utah_forge_exact_rsf_inverse_fit.py`, `utah_forge_exact_rsf_showcase.py`, `utah_forge_exact_rsf_multistart_check.py` | Full RSF: a, b, D_c, µ₀ | τ, V, θ jointly | Parameter unstable; multiple optima | **Key negative result** |
| 9 | Theta consistency check | `utah_forge_theta_equation_consistency.py` | RSF dθ/dt from estimated θ | dθ/dt | Inconsistent; θ not recoverable | **Key negative result** |
| 10 | Regime diagnostics | `utah_forge_regime_analysis.py`, `utah_forge_regime_balanced_tau_evaluation.py` | Train/holdout regime statistics | N/A (diagnostic) | Regime mismatch explains holdout degradation | **Key explanation** |

---

## Detailed Model Descriptions

### Model 1: Polynomial SINDy Baseline

**Why tried:** Establish a non-physics-informed baseline. SINDy with polynomial libraries is a standard starting point.

**Library:** {1, τ, V, V_drive, τ², V², τV, τ²V, ...} (polynomial combinations up to degree 2-3)

**Equation target:** Both dτ/dt and dV/dt simultaneously.

**Result:** Tau equation converges quickly to something interpretable. Velocity equation is noisy and coefficients are unstable across splits.

**Lesson:** Physics-informed library construction is necessary.

---

### Model 2: Observed-Only RSF-Informed SINDy (Model A)

**Why tried:** Include physics-motivated features (V_drive − V for the stress law; log terms for velocity) without requiring unobserved θ.

**Library:** {τ, V, V_drive, V_drive−V, τ·V, log(V), log(V_drive), ...}

**Results:**
- **Tau equation:** dτ/dt ≈ k(V_drive − V) recovered robustly. R² > 0.99 on in-sample derivative prediction.
- **Velocity equation:** Partial log(V) structure; coefficient stability is split-dependent.

**Status:** This is the primary baseline model. All later models compare to this.

---

### Model 3: Memory-Augmented Model (Model B)

**Why tried:** RSF state θ captures contact age/memory. Since θ is unobserved, use rolling averages (τ_avg, τ_ema) as proxies.

**Additional features:** τ_avg (rolling mean of τ), τ_ema (exponential moving average), rolling standard deviation.

**Results:**
- Rollout duration slightly improved on training cycles.
- Memory features selected by STLSQ but with variable coefficients.
- Not physically interpretable as true θ.

**Lesson:** Memory improves numeric rollout without being scientifically meaningful.

---

### Model 4: Theta-Proxy Model (Model C)

**Why tried:** Construct an explicit θ surrogate from the integrated RSF aging law: θ(t) ≈ t − S/D_c + θ₀.

**Additional features:** θ surrogate, dθ/dt estimate.

**Results:** Marginal improvement over Model B. θ_surrogate is highly correlated with τ, so the regression cannot distinguish whether it is selecting θ or just a redundant τ feature.

**Lesson:** θ is not independently identifiable from τ and V observations alone.

---

### Model 5: Tau-Isolated Spring Law

**Why tried:** Isolate the stress equation to specifically test whether the spring-slider law is recoverable.

**Equation target:** Only dτ/dt.

**Result:** **dτ/dt = k(V_drive − V)** is cleanly and robustly recovered. k estimates are physically reasonable (≈ machine stiffness). This is the **most robust finding** of the project.

---

### Model 6: Reduced Velocity Law

**Why tried:** The RSF velocity equation has the form dV/dt ∝ (1/a·V)[dτ/dt − σ_N·b·dθ/dt]. A reduced form dV/dt ≈ f(τ, V) can be attempted without θ.

**Library:** {log(V), τ/V, V·log(V), 1/V, ...} (RSF-motivated terms)

**Result:** Partial recovery of logarithmic structure. Best-performing splits show physically plausible equations. Holdout splits show higher variance.

---

### Model 7: Conditional Velocity Variants

**Why tried:** Different dynamic regimes (sticking vs. slipping) may follow different governing equations. Split the data by τ regime and fit separate models.

**Result:** Equations differ significantly between regimes. This confirms that a single global SINDy model for V is insufficient. **Regime heterogeneity is a key explanation for holdout failure.**

---

### Model 8: Exact RSF Inverse Fit

**Why tried:** Directly fit the full RSF parameter set (a, b, D_c, µ₀, k) using nonlinear least squares on the exact RSF system.

**Method:** Multi-start nonlinear optimization (scipy.optimize) on the full RSF ODE system.

**Result:** Parameter estimation is unstable. Multiple local optima exist with similar residuals but very different parameter values. This is a **key negative result:** exact RSF fitting from noisy data is non-identifiable with standard optimization.

---

### Model 9: Theta Equation Consistency

**Why tried:** Check whether the estimated θ surrogate is consistent with the RSF aging law dθ/dt = 1 − θ·V/D_c.

**Result:** Inconsistent. The θ surrogate does not satisfy the RSF aging law when evaluated against the data. This confirms that θ cannot be recovered reliably.

---

### Model 10: Regime Diagnostics

**Why tried:** Explain the gap between training and holdout performance.

**Method:** Statistical comparison of τ, V distributions across training and holdout cycles. Feature space visualization.

**Result:** Training and holdout cycles occupy different regions of (τ, V) space. This regime mismatch is the primary explanation for holdout degradation. **Not a model — a diagnostic.**
