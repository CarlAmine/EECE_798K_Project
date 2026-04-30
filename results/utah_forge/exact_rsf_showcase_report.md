# Exact RSF Showcase Report

## 1. Exact equation set
- `dtau/dt = 2.852472e-03*(V_drive - V)`
- `dV/dt = (1/2.519965e-02) * [tau - sigmaN*(9.042632e-12 + 7.146679e-01*log(V/V0) + 1.671089e-01*log(theta*V0/1.518258e+01))]`
- `dtheta/dt = 1 - V*theta/1.518258e+01`

## 2. Where it came from
- Workflow source: `scripts/utah_forge_exact_rsf_multistart_check.py`
- Core fitter: `fit_exact_rsf_inverse_model()` in `src/exact_rsf.py`
- Winning run: `start_index = 2`
- Winning checkpoint: `results/utah_forge/exact_rsf_multistart_checkpoints/exact_fit_multistart_2.pkl`
- It came from the bounded multistart refinement, not from the original base exact fit.

## 3. How it was obtained
- Data subset: train steps `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`; holdout steps `p5838_step2, p5838_step7`.
- Events were selected by the exact-RSF workflow from the RSFit-aligned Utah FORGE step windows, then downsampled/smoothed into `ExactRSFSegment` objects.
- Theta was treated as a latent dynamical state during fitting, with one event-specific `delta_log_theta0` offset per training event.
- The optimized global parameters were `k, m, mu0, a, b, Dc`.
- The loss was mixed: trajectory matching on `tau(t)` and `V(t)`, plus a small derivative-consistency penalty on `dV/dt`, plus a penalty on theta offsets.
- Constraints enforced were `k > 0`, `m > 0`, `a > 0`, `b > 0`, `Dc > 0`, with positive `theta(t)` and positive `V(t)` maintained during simulation.
- Multistart generated four nearby initializations and chose the best exact-form fit by lowest final cost.

## 4. Parameter table
| parameter | value | mean_across_starts | std_across_starts | cv_across_starts |
| --- | --- | --- | --- | --- |
| k | 2.852472e-03 | 2.671697e-03 | 3.132796e-04 | 1.172587e-01 |
| m | 2.519965e-02 | 2.013600e-02 | 8.770678e-03 | 4.355721e-01 |
| mu0 | 9.042632e-12 | 8.348592e-02 | 1.446018e-01 | 1.732051e+00 |
| a | 7.146679e-01 | 5.386812e-01 | 3.048620e-01 | 5.659415e-01 |
| b | 1.671089e-01 | 1.324547e-01 | 6.006350e-02 | 4.534645e-01 |
| Dc | 1.518258e+01 | 2.605005e+02 | 4.249017e+02 | 1.631098e+00 |

Standard RSF form:
- `m dV/dt = tau - sigmaN [ mu0 + a log(V/V0) + b log(theta*V0/Dc) ]`
- `m = 2.519965e-02`
- `mu0 = 9.042632e-12`
- `a = 7.146679e-01`
- `b = 1.671089e-01`
- `Dc = 1.518258e+01`

Event-specific theta0 values from the winning run:
- `p5838_step3`: `5.072713e+02`
- `p5838_step8`: `5.670663e+02`
- `p5838_step9`: `1.381797e+02`
- `p5838_step4`: `1.227481e+02`
- `p5838_step5`: `2.438402e+01`
- `p5838_step10`: `3.434744e+00`
- `p5838_step2`: `5.841272e+02`
- `p5838_step7`: `5.547572e+02`

## 5. Metrics and rollout
- Optimization success: `True`
- Status / message: `2` / ``ftol` termination condition is satisfied.`
- `nfev = 37`
- Final cost: `2.483117e+03`
- Mean holdout rollout error: `3.499610e+02`
- Mean holdout stable fraction: `0.004444`
- Mean holdout onset timing error: `4.622` s
- Mean holdout peak timing error: `7.754` s
- Holdout event metrics from the winning run:
- `p5838_step2`: tau RMSE `3.968328e+01`, V RMSE `4.311719e+02`, combined rollout error `4.494918e+02`, stable fraction `0.004444`, onset timing error `0.469` s, peak timing error `14.722` s
- `p5838_step7`: tau RMSE `2.380539e+01`, V RMSE `3.537155e+02`, combined rollout error `2.504303e+02`, stable fraction `0.004444`, onset timing error `8.774` s, peak timing error `0.785` s

## 6. Why it is attractive
- It is the closest exact-form RSF-looking system found in the repo.
- The tau equation stays in the expected spring-loading form.
- The velocity law keeps explicit `tau`, `log(V/V0)`, and `log(theta*V0/Dc)` structure.
- After multistart, the fit converged cleanly and the theta term remained meaningfully active.

## 7. Why it is not the final trusted model
- It was not selected as the final trusted model because identifiability remained poor even after multistart.
- Near-constant sigmaN remained true: `True`.
- Parameter confounding remained true: `True`.
- JTJ condition number stayed very large: `8.904456e+08`.
- Parameter stability across starts was weak, especially for `m`, `mu0`, `a`, `b`, and `Dc`.
- In plain English: the exact form can be fit, but the parameters do not lock down uniquely enough to treat it as a reliable recovered governing law.

## 8. Comparison to the reduced fallback
model,workflow_source,equation,optimization_success,nfev,cost,derivative_rmse,rollout_error,stable_fraction,peak_timing_error_s,onset_timing_error_s,identifiability_status,judgment
exact_rsf_closest_exact_fit,exact_rsf_multistart_check -> fit_exact_rsf_inverse_model,dV/dt = (1/2.519965e-02) * [tau - sigmaN*(9.042632e-12 + 7.146679e-01*log(V/V0) + 1.671089e-01*log(theta*V0/1.518258e+01))],True,37.0,2483.116569174679,,349.96100786151817,0.0044444444444444444,7.753500000000713,4.621500000001106,non-identifiable,closest exact fit but non-identifiable
reduced_rsf_final_fallback,proposal_equation_recovery,dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV,True,,,27.116409341814872,449.30345694564386,0.006166666666666667,16.555999999999585,15.378000000000611,usable reduced fallback,best final usable equation


Key comparison points:
- Exact RSF-looking fit mean rollout error: `3.499610e+02` versus reduced fallback rollout MSE `4.493035e+02`.
- Exact RSF-looking fit stable fraction: `0.004444` versus reduced fallback `0.006167`.
- Exact RSF-looking fit peak timing error: `7.754` s versus reduced fallback `16.556` s.
- The reduced fallback stayed the best final usable equation because it was more defensible under the identifiability diagnostics, even though the exact-form fit looked more physically complete on paper.

## 9. Short presentation-ready explanation
The equation set came from the bounded multistart exact-RSF refinement, specifically winning start index 2 in `exact_rsf_multistart_check.py`. It is attractive because it is the closest exact-form RSF-looking system we found: spring-loading tau, explicit log-rate friction, and an active theta term. It was obtained by directly simulating the coupled tau-V-theta system and fitting global RSF parameters plus per-event theta offsets under positivity constraints. However, it was not selected as the final trusted model because sigmaN stayed nearly constant, the parameters remained confounded, and the estimates were unstable across starts even after cleaner convergence. So this is the best exact-looking fit to showcase, but not the best final usable equation. The reduced RSF fallback remains the final reportable velocity law because it is more scientifically defensible under the identifiability diagnostics.