# Equation (1) vs Equation (2) Forensic Audit

## Scope and current git state

This audit is a diagnosis of workflow evolution, not a new modeling pass.

- `git status --short` shows one modified tracked file, [`src/io/utah_forge.py`](/abs/path/placeholder), plus many untracked proposal/exact-RSF scripts and output artifacts under `results/utah_forge/`.
- `git status --short` shows one modified tracked file, `src/io/utah_forge.py`, plus many untracked proposal/exact-RSF scripts and output artifacts under `results/utah_forge/`.
- `git log --oneline -n 20` is available and shows the relevant recent repo milestones:
  - `76467f2 Updating new dataset`
  - `0401118 Add Utah FORGE p5838 refinement results`
  - `0578244 Restructure datasets and add validation outputs`
- The key forensic trail lives in:
  - `scripts/utah_forge_reviewer_ablation.py`
  - `scripts/utah_forge_memory_refinement.py`
  - `scripts/utah_forge_proposal_equation_recovery.py`
  - `src/exact_rsf.py`
  - `scripts/utah_forge_exact_rsf_inverse_fit.py`
  - `scripts/utah_forge_exact_rsf_multistart_check.py`
  - `results/utah_forge/p5838_final_report.md`
  - `results/utah_forge/p5838_memory_model_report.md`
  - `results/utah_forge/proposal_equation_recovery_report.md`
  - `results/utah_forge/exact_rsf_inverse_fit_report.md`
  - `results/utah_forge/exact_rsf_multistart_check.md`

## Short timeline

1. Generic sparse regression phase:
   - `src/sindy/models.py` provides a basic iterative thresholded least-squares SINDy model.
   - This layer is generic: no sign constraints, no special handling for spring loading, no latent-state treatment, and no distinction between equations with and without hidden variables.

2. Reviewer ablation phase with shared libraries:
   - `scripts/utah_forge_reviewer_ablation.py` fit Models A, B, and C by giving both `dtau/dt` and `dV/dt` the same broad candidate library within each model.
   - This produced the messy early tau equations in `results/utah_forge/p5838_final_report.md`.
   - Model C also changed the usable subset by skipping steps when reconstructed theta was invalid or heavily clipped.

3. Memory-refinement phase:
   - `scripts/utah_forge_memory_refinement.py` was the first clear turning point for equation (1).
   - It fit tau and velocity with different libraries, and the tau library was reduced to `["1", "V", "V_drive_minus_V"]`.
   - This is where tau first became compact and visibly spring-loading dominated.

4. Proposal-equation recovery phase:
   - `scripts/utah_forge_proposal_equation_recovery.py` made the separation explicit.
   - Equation (1) was fit in its own dedicated function, `fit_tau_recovery()`, with a tiny physical library, positive sign constraint on `V_drive_minus_V`, threshold sweep, and coefficient de-normalization back to physical units.
   - Equation (2) moved to a separate model ladder with proposal-specific RSF features and identifiability diagnostics.

5. Exact latent-state RSF phase:
   - `src/exact_rsf.py` and `scripts/utah_forge_exact_rsf_inverse_fit.py` stopped treating theta as just another regression column and fit the coupled RSF system directly by simulation.
   - `scripts/utah_forge_exact_rsf_multistart_check.py` then checked whether weak convergence was the real issue.
   - Convergence improved, but identifiability did not.

## Direct answer first

Equation (1) improved mainly because the workflow stopped making it compete inside a large shared library and instead fit it separately with a tiny physically targeted structure centered on `V_drive - V`.

Equation (2) did not improve in the same way because its difficulty was not mostly library clutter; it was the combination of latent theta, near-constant `sigmaN`, and parameter confounding. Even after the workflow became more proposal-faithful and moved from regression to direct latent-state inverse fitting, the data still did not separate `mu0`, `a`, `b`, and `Dc` cleanly.

## Comparison table: old vs new for equation (1)

| Stage | Main file(s) | Candidate terms for tau | Fit strategy | Constraints / sparsity | Data / subset effect | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Early generic / reviewer ablation | `scripts/utah_forge_reviewer_ablation.py` | Model A: `1, tau, V, logV, tau*logV, V_drive_minus_V`; Model B adds `tau_avg, tau_ema, S`; Model C adds `logTheta, tau*logTheta` | `fit_two_equation_model()` fits tau and V separately but from the same broad library per model | Generic thresholded regression via `base.fit_sparse_equation()`; only mandatory term for tau was `V_drive_minus_V`; no sign constraints | Model C also skips steps when theta reconstruction is clipped/invalid, so the subset changes | Tau equations stay dense and pick up nuisance terms |
| First real improvement | `scripts/utah_forge_memory_refinement.py` | Tau library reduced to `1, V, V_drive_minus_V` | Tau and V fit with different libraries in `fit_memory_configuration()` | Same sparse thresholding backend, but nuisance tau terms were removed from the library | Same RSFit-aligned step split as later work | Tau becomes compact: `dtau/dt = - 2.590e-02*1 + 3.470e-04*V + 8.961e-03*V_drive_minus_V` |
| Final proposal-specific tau path | `scripts/utah_forge_proposal_equation_recovery.py` | `1, V, V_drive_minus_V` only | Dedicated `fit_tau_recovery()` separate from any velocity fit | Positive lower bound on `V_drive_minus_V`; threshold sweep; simplest physically consistent model selected; coefficients de-normalized to physical units | Train/holdout fixed to RSFit-aligned steps | Final compact tau equation: `dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V`; closest one-term form `dtau/dt ~= 8.476432e-03*(V_drive - V)` |

## Comparison table: old vs new for equation (2)

| Stage | Main file(s) | Velocity structure | Fit strategy | What improved | What still failed |
| --- | --- | --- | --- | --- | --- |
| Reviewer ablation A/B/C | `scripts/utah_forge_reviewer_ablation.py` | Broad observed or surrogate libraries; Model C adds `logTheta` and `tau*logTheta` | Shared-library sparse regression on derivatives | Revealed that `logV` and memory/state-like terms matter | Dense equations, unstable interpretation, theta proxy quality issues |
| Memory surrogate phase | `scripts/utah_forge_memory_refinement.py` | `tau, V, logV, tau*logV, tau_avg, tau_ema, S` | Separate V library with memory surrogates | Better practical surrogate behavior | Still not the exact RSF equation, and still not a clean theta recovery |
| Proposal model ladder | `scripts/utah_forge_proposal_equation_recovery.py` | Exact-RSF features `1, tau, sigmaN_logV, sigmaN_logTheta`; reduced fallback `1, tau, sigmaN, sigmaN_logV` | Constrained sparse linear fitting with sign bounds, sample-level theta masking, identifiability report | Confirmed that `log(V)` survives in a physics-informed reduced law | Exact theta term collapsed to ~0; `sigmaN` nearly constant; condition number and redundancy diagnostics stayed poor |
| Exact latent-state inverse fit | `src/exact_rsf.py`, `scripts/utah_forge_exact_rsf_inverse_fit.py` | Full proposal system with latent theta and `dtheta/dt = 1 - V theta / Dc` | Trajectory-based constrained inverse fitting by simulation | Rules out the objection that derivative regression alone caused the failure | Exact equation (2) still not identifiable; blocker remains `near-constant sigmaN, parameter confounding` |
| Bounded multistart convergence check | `scripts/utah_forge_exact_rsf_multistart_check.py` | Same exact latent-state model | Same model, more evaluations and 4 starts | Convergence got cleaner; theta coefficient could become numerically active | Parameters did not stabilize across starts; conclusion unchanged |

## Reconstructing the evolution of equation (1)

### Earliest bad or messy versions

The earliest clearly documented messy tau equations are in `results/utah_forge/p5838_final_report.md` from the reviewer-ablation workflow:

- Model A:
  - `dtau/dt = 3.129e+00*1 - 2.193e-01*tau + 9.743e-04*V - 1.183e+00*logV + 8.300e-02*tau*logV + 9.229e-03*V_drive_minus_V`
- Model B:
  - `dtau/dt = 1.081e+01*1 + 1.082e-03*V + 8.479e-03*V_drive_minus_V + 7.072e+00*tau - 2.706e+00*logV + 1.941e-01*tau*logV - 4.686e+00*tau_avg - 3.169e+00*tau_ema + 6.116e-05*S`
- Model C:
  - `dtau/dt = - 3.755e+00*1 + 2.852e-01*tau + 1.727e-04*V + 1.102e+00*logV + 8.855e-02*logTheta - 8.525e-02*tau*logV - 6.983e-03*tau*logTheta + 1.056e-02*V_drive_minus_V`

These are not compact spring-loading laws. They are what you would expect when the tau fit is allowed to soak up whatever terms are available in a large shared library.

### Later good versions

The first clean version appears in `results/utah_forge/p5838_memory_model_report.md`:

- `dtau/dt = - 2.590e-02*1 + 3.470e-04*V + 8.961e-03*V_drive_minus_V`

The final proposal-specific version appears in `results/utah_forge/proposal_equation_recovery_report.md`:

- `dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V`
- One-term approximation: `dtau/dt ~= 8.476432e-03*(V_drive - V)`

### Exact differences that made equation (1) better

- Candidate terms:
  - Early tau fit used the same broad library as velocity.
  - Later tau fit used only `1`, `V`, and `V_drive_minus_V`.

- Thresholding:
  - Early tau used generic sparse thresholding inside a large library.
  - Final tau still used thresholding, but only inside a tiny targeted library, so thresholding became a cleanup step rather than the main source of structure.

- Fitting strategy:
  - Early: `fit_two_equation_model()` in `scripts/utah_forge_reviewer_ablation.py`.
  - Later: tau isolated in `fit_memory_configuration()` and finally in `fit_tau_recovery()`.

- Validation criterion:
  - Early reports emphasized broad derivative fit and holdout divergence across joint tau/V models.
  - Final tau path explicitly ranked candidates by physical sign consistency, presence of `V_drive_minus_V`, low term count, and holdout derivative MSE.

- Physical constraints:
  - Early tau path had no sign constraints.
  - Final tau path forced the `V_drive_minus_V` coefficient to be nonnegative.

- Physical units:
  - Final tau path explicitly z-scored features for fitting and then de-normalized coefficients back to physical units using `denormalize_coefficients()`.

- Data subset:
  - The train/holdout step split stayed broadly consistent in the RSFit-aligned workflows.
  - The main improvement did not come from a radically different subset; it came from isolating tau and simplifying its library.

### Plain conclusion for equation (1)

Equation (1) got better mostly because it was isolated and fit separately with nuisance terms removed. The biggest practical change was not a smarter optimizer; it was reducing the tau identification problem to the physically relevant spring-loading structure and then enforcing the expected sign on the drive term.

## Reconstructing the evolution of equation (2)

### Earlier bad versions

The reviewer-ablation equations in `results/utah_forge/p5838_final_report.md` show the early problem clearly:

- Model B was practical but surrogate-heavy:
  - `dV/dt = - 1.245e+03*1 + 5.023e-01*V + 5.504e+00*V_drive_minus_V + 5.823e+02*tau + 3.072e+02*logV - 2.281e+01*tau*logV - 1.960e+02*tau_avg - 2.934e+02*tau_ema - 1.294e-01*S`
- Model C tried to make theta explicit:
  - `dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V`

Those equations are informative as surrogate models, but they do not cleanly identify the proposal RSF structure.

### Later reduced-RSF and exact-RSF versions

The proposal-specific reduced fallback in `results/utah_forge/proposal_equation_recovery_report.md` is:

- `dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV`

The exact feature-regression attempt in the same report is:

- `dV/dt = -8.983782e+01 + 6.996144e+00*tau - 5.382505e-02*sigmaN_logV`

The exact theta coefficient collapsed numerically to approximately zero, and the report says:

- `Exact equation (2) not identifiable from current data; best fallback model is B_reduced_rsf`

Then the direct latent-state inverse fit in `results/utah_forge/exact_rsf_inverse_fit_report.md` implemented the proposal literally:

- `dtau/dt = 2.124098e-03*(V_drive - V)`
- `dV/dt = (1/4.948323e-03) * [tau - sigmaN*(3.339335e-01 + 1.065980e-02*log(V/V0) + 2.843257e-02*log(theta*V0/9.964518e+02))]`
- `dtheta/dt = 1 - V*theta/9.964518e+02`

But the same report still concludes:

- `Equation (1) recovered; exact equation (2) implemented and tested directly, but still not identifiable from current Utah FORGE data`

The bounded multistart check in `results/utah_forge/exact_rsf_multistart_check.md` improved convergence but not identifiability:

- Best run success: `True`
- Parameter estimates stabilize across starts: `False`
- Final scientific conclusion changes: `False`

### Why equation (2) did not improve to the same degree

The blocker is a combination, not just one issue:

- Hidden-state theta:
  - Equation (2) contains a latent state that equation (1) does not.
  - Early theta was only available through reconstruction or proxy paths, which were noisy and sometimes clipped.

- Near-constant `sigmaN`:
  - The proposal and exact-RSF reports both document that `sigmaN` is nearly constant on the usable subset.
  - That makes intercept, `mu0`, and the `sigmaN`-weighted RSF coefficients hard to separate.

- Parameter confounding and multicollinearity:
  - Proposal report: exact-design condition number `4.917298e+04`, `sigma_cv = 6.864859e-04`, intercept and `sigmaN` effectively redundant.
  - Exact inverse fit report: `sigmaN` too constant for clean `mu0/a/b` separation and `parameter confounding flag = True`.
  - Multistart check: condition number improved, but estimates still drifted strongly across starts.

- Weak convergence was only a secondary issue:
  - Multistart showed the optimizer was part of the story, but not the main story.
  - Once convergence improved, identifiability still did not.

### Plain conclusion for equation (2)

Equation (2) did not fail because we never tried the right model. It failed because once we did try the right model, the data still did not provide enough independent information to separate the theta-bearing RSF parameters cleanly.

## Model B vs Model C for equation (1)

Model B looked cleaner than Model C for tau largely because Model C forced tau to share a theta-bearing library that equation (1) does not physically need.

Three things happened in Model C:

- Different feature library:
  - Model B tau library included memory surrogates.
  - Model C tau library included `logTheta` and `tau*logTheta`.
  - That let theta-related terms leak into the tau equation even though the proposal tau law is only spring loading.

- Different usable subset:
  - `prepare_model_segments()` in `scripts/utah_forge_reviewer_ablation.py` reconstructs theta for Model C and skips a step entirely if theta is invalid or too clipped.
  - So Model C was not fit on exactly the same rows as Model B.

- Reconstructed and clipped theta:
  - `reconstruct_theta()` computes theta from RSFit inversion and clips it.
  - If clipping fraction exceeds `0.25`, the step is dropped.
  - That is enough to alter both feature quality and the effective training subset.

So the extra terms in Model C’s tau equation are much more likely to be fitting artifacts from a shared theta-bearing library and altered subset than evidence that tau dynamics truly depend on theta.

## Direct answers to the five main questions

### 1. What exact code or workflow changes made equation (1) improve?

The decisive changes were:

- `scripts/utah_forge_memory_refinement.py`
  - `build_memory_libraries()` reduced tau to `1`, `V`, `V_drive_minus_V`.
  - `fit_memory_configuration()` fit tau separately from velocity.

- `scripts/utah_forge_proposal_equation_recovery.py`
  - `fit_tau_recovery()` fit tau in a dedicated path.
  - Added positive sign constraint on `V_drive_minus_V`.
  - Added threshold sweep over a tiny tau library.
  - De-normalized coefficients back to physical units.
  - Ranked models by physical consistency and simplicity, not just residual size.

### 2. Why did equation (2) not improve in the same way?

Because equation (2) has a fundamentally harder identification problem:

- It depends on latent theta.
- Its physically correct terms are multiplied by `sigmaN`, which is nearly constant on the usable subset.
- That makes `mu0`, `a`, `b`, and `Dc` strongly confounded.
- Even direct latent-state inverse fitting plus multistart did not stabilize those parameters.

### 3. Did we explicitly change any model, library, fitting path, thresholding, validation, or data subset that helped equation (1) but not equation (2)?

Yes.

- Equation (1):
  - library was drastically reduced
  - fitted in isolation
  - positive drive sign was enforced
  - model choice favored sparse physically correct structure

- Equation (2):
  - also received a more proposal-faithful path later, but the main challenge was not library clutter alone
  - it needed theta and `sigmaN` variation that the data did not provide
  - the later move from regression to latent-state inverse fitting improved honesty, not identifiability

### 4. Did we ever fit equation (1) separately from equation (2), and if so where and why?

Yes, in two important places.

- `scripts/utah_forge_memory_refinement.py`
  - tau and velocity used different candidate libraries because the tau law is much simpler physically.

- `scripts/utah_forge_proposal_equation_recovery.py`
  - `fit_tau_recovery()` is a completely separate tau-specific workflow.
  - This was done specifically to recover the proposal spring-loading law cleanly instead of letting tau inherit velocity-library clutter.

### 5. Was the improvement due to changing the candidate library, threshold logic, sign constraints, de-normalization, subset, a proposal-specific path, moving to latent-state inverse fitting, or something else?

For equation (1), the improvement was mainly due to:

- changing the candidate library
- isolating the tau fit from the velocity fit
- adding sign constraints
- choosing the simplest physically consistent model
- de-normalizing to physical units for interpretation

For equation (2), the later changes did include:

- a proposal-specific path
- sign-aware constrained fitting
- sample-level theta masking
- a direct latent-state inverse fit
- bounded multistart convergence checking

But those changes mainly improved the identification strategy and the honesty of the diagnosis. They did not change the underlying data limitations enough to make the full theta-bearing equation identifiable.

## Single most important change

- Single most important reason equation (1) became good:
  - it was isolated into its own tiny physically targeted identification problem centered on `V_drive - V`.

- Single most important reason equation (2) did not:
  - near-constant `sigmaN` left the theta-bearing RSF parameters structurally confounded, so better fitting strategy could not create missing identifiability.

- Did we change “the model” or mainly change “the identification strategy”?
  - For equation (1), we mainly changed the identification strategy by stripping the problem down to the right physics.
  - For equation (2), we changed both the representation and the identification strategy, up to a direct latent-state inverse fit, but the scientific outcome stayed limited by identifiability rather than by model-form choice alone.

## Student Q/A version

If someone asks why equation (1) cleaned up while equation (2) did not, the short answer is that equation (1) is observable and simple, while equation (2) contains a hidden state and confounded parameters. Early in the project we were fitting tau with the same broad libraries used for velocity, so tau picked up lots of nuisance terms. Once tau was isolated and fit only against `V_drive - V` and a minimal companion term, it consistently collapsed to the expected spring-loading form. Equation (2) did not respond the same way because the main obstacle was not library clutter but the lack of independent information needed to separate `mu0`, `a`, `b`, `Dc`, and theta effects when `sigmaN` barely changes. That is why the reduced RSF fallback is scientifically defensible, while the full theta-bearing equation remains weakly identifiable on this dataset.
