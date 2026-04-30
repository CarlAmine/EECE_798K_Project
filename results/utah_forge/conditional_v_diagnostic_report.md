# Conditional V Diagnostic Report

## Goal
Test whether equation (2) for `V` looks cleaner when isolated and evaluated conditionally, in the same spirit as the semi-observed tau rollout, while keeping the interpretation honest.

## Honesty framing
- The tau rollout used elsewhere in this project is semi-observed because `V(t)` and `V_drive(t)` are supplied.
- This new V diagnostic is also conditional / semi-observed because `tau(t)` and `sigmaN(t)` are supplied from observations, and `theta(t)` is supplied only when RSFit-aligned theta is considered valid enough to use diagnostically.
- These conditional tests are not the same evidentiary level as a full dynamic rollout.
- The current reduced RSF model remains the main usable dynamic V model unless this diagnostic clearly overturns it on a fair comparison, which requires caution because the conditional task is easier.

## Setup
- Train steps: `p5838_step3, p5838_step4, p5838_step5, p5838_step8, p5838_step9, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`
- Representative steps for visualization: `p5838_step2, p5838_step9, p5838_step5`
- Representative steps were chosen from the saved reduced-RSF step-difficulty summary to span one easy, one medium, and one hard case under the existing V analysis.

## Theta availability
| step_name | theta_event_valid | theta_sample_valid | theta_log_correlation | theta_keep_fraction | theta_reason |
| --- | --- | --- | --- | --- | --- |
| p5838_step2 | True | True | 0.468 | 1.000 | ok |
| p5838_step9 | True | True | 0.955 | 1.000 | ok |
| p5838_step5 | False | False | n/a | 0.000 | theta_direct_invalid,theta_direct_low_variation,theta_alignment_undefined,theta_clipped,too_few_high_quality_samples,short_high_quality_run |

## Tested equations

### Conditional reduced-style V fit
- Library: `1, tau, sigmaN, sigmaN_logV`
- Equation: `dV/dt = 1.577175e+03 - 2.326718e+01*tau - 6.418459e+01*sigmaN - 9.170580e-01*sigmaN_logV`
- Eligible train steps: `p5838_step3, p5838_step4, p5838_step5, p5838_step8, p5838_step9, p5838_step10`
- Eligible holdout steps: `p5838_step2, p5838_step7`
- Mean holdout derivative RMSE: `24.507`
- Mean representative rollout RMSE: `65.093`

### Conditional theta-augmented V fit
- Library: `1, tau, sigmaN, sigmaN_logV, sigmaN_logTheta`
- Equation: `dV/dt = 9.817605e+02 + 1.817337e+00*tau - 5.269288e+01*sigmaN + 2.476670e+00*sigmaN_logV + 2.516514e+00*sigmaN_logTheta`
- Eligible train steps: `p5838_step3, p5838_step8, p5838_step9`
- Eligible holdout steps: `p5838_step2, p5838_step7`
- Mean holdout derivative RMSE: `102.482`
- Mean representative rollout RMSE: `9.189e+03`

### Conditional no-tau ablation
- Library: `1, sigmaN, sigmaN_logV, sigmaN_logTheta`
- Equation: `dV/dt = 1.046393e+03 - 5.479768e+01*sigmaN + 2.478207e+00*sigmaN_logV + 2.522139e+00*sigmaN_logTheta`
- Eligible train steps: `p5838_step3, p5838_step8, p5838_step9`
- Eligible holdout steps: `p5838_step2, p5838_step7`
- Mean holdout derivative RMSE: `102.653`
- Mean representative rollout RMSE: `9.082e+03`

## Conditional vs dynamic comparison
| display_name | equation_form | derivative_rmse | rollout_rmse | timing_error_s | stable_fraction | graph_looks_better | still_physically_trustworthy | evaluation_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current reduced RSF dynamic rollout | dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV | 121.736 | 60.569 | 9.363 | 0.669 | baseline | yes | dynamic |
| Conditional reduced-style V fit | dV/dt = 1.577175e+03 - 2.326718e+01*tau - 6.418459e+01*sigmaN - 9.170580e-01*sigmaN_logV | 24.507 | 65.093 | 5.196 | 0.670 | no | limited | conditional |
| Conditional theta-augmented V fit | dV/dt = 9.817605e+02 + 1.817337e+00*tau - 5.269288e+01*sigmaN + 2.476670e+00*sigmaN_logV + 2.516514e+00*sigmaN_logTheta | 102.482 | 9.189e+03 | 15.807 | 0.260 | no | limited | conditional |
| Conditional no-tau ablation | dV/dt = 1.046393e+03 - 5.479768e+01*sigmaN + 2.478207e+00*sigmaN_logV + 2.522139e+00*sigmaN_logTheta | 102.653 | 9.082e+03 | 15.809 | 0.289 | no | limited | conditional |

## Direct answers
- 1. Does isolating V make the graphs look nicer? On the representative steps, the best conditional variant is `Conditional reduced-style V fit` and the graph-quality judgment is `no` relative to the dynamic baseline.
- 2. Does it improve derivative fit? Best conditional derivative RMSE is `24.507` versus `121.736` for the representative-step dynamic baseline; the saved project holdout derivative RMSE for reduced RSF is `27.116`.
- 3. Does it improve conditional rollout? Best conditional rollout RMSE is `65.093` versus `60.569` for the same representative steps.
- 4. Does tau remain active or collapse? In the tau-including reduced-style fit, tau stays in the equation. Compared with the no-tau ablation, rollout RMSE changes from `65.093` to `9.082e+03`.
- 5. Does theta help visually or numerically? Theta-augmented conditional rollout RMSE is `9.189e+03` versus `65.093` for the reduced-style conditional fit.
- 6. Does this overturn the main conclusion about V? No automatic overturning is warranted just because a conditional test is cleaner; the evidentiary standard remains lower than a full dynamic rollout.
- 7. What is the honest way to present this result? Present it as a conditional / semi-observed velocity diagnostic that asks how much of the V difficulty comes from coupled rollout difficulty, while keeping the reduced RSF dynamic model as the main usable V result unless the conditional evidence is compelling on a fair comparison.

## Fairness notes
- Saved project-wide reduced-RSF holdout pair (`p5838_step2, p5838_step7`) metrics remain: derivative RMSE `27.116`, rollout MSE `449.303`, stable fraction `6.167e-03`.
- The representative-step dynamic row above was recomputed on the same easy/medium/hard steps used for the new conditional figures so the plots are visually comparable.
- That side-by-side comparison is still not fully fair to the dynamic model because the conditional variant consumes observed auxiliary signals.

## How to present this honestly
- This diagnostic shows whether poor V graphs are partly due to coupled rollout difficulty rather than only poor instantaneous equation structure.
- Cleaner conditional V fits do not automatically mean full equation recovery.
- A conditional / semi-observed V rollout should be described as a diagnostic using observed forcing/context inputs, not as a replacement for the main dynamic result.
- The reduced RSF dynamic model remains the main usable V result unless the diagnostic clearly beats it in a scientifically fair way.

## Generated files
- `results/utah_forge/conditional_v_diagnostic_report.md`
- `results/utah_forge/conditional_v_diagnostic_report.json`
- `results/utah_forge/conditional_v_diagnostic_table.csv`
- `results/utah_forge/conditional_v_rollout_examples.png`
- `results/utah_forge/reduced_dynamic_vs_conditional_v.png`
- `results/utah_forge/conditional_v_error_maps.png`
- `results/utah_forge/conditional_v_derivative_scatter.png`
- `results/utah_forge/conditional_v_equation_table.png`
- `results/utah_forge/Finalv5/` created: `True`

