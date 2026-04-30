# Proposal Equation Recovery Report

## Best insertion point
- new standalone proposal-equation script using RSFit-aligned step segments
- Train steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`
- Why the current models miss the proposal target cleanly: Current Utah FORGE paths either use delay proxies (tau_lag), memory surrogates (tau_avg, tau_ema, S), or theta-informed libraries with extra interaction terms, and many reported equations stay in normalized coordinates or without sign constraints. That makes them useful surrogates, but not clean recoveries of the two proposal equations.

## Data inclusion / exclusion
- Total RSFit-aligned steps inspected: `8`
- Theta-valid steps under strict event screening: `5`
- Theta-usable steps under sample masking: `5`
- Theta-usable samples under sample masking: `15000`

## Equation (1) recovery
- Exact fit: `dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V`
- Closest one-term fit: `dtau/dt ~= 8.476432e-03*(V_drive - V)`
- Holdout derivative MSE: `6.139149e-03`

## Equation (2) model ladder
| model | equation | n_terms | holdout_derivative_mse | holdout_derivative_rmse | holdout_derivative_mae | holdout_derivative_r2 | rollout_mse | stable_fraction | peak_timing_error_s | onset_timing_error_s | timing_error_s | physics_consistent | theta_term_active | acoustic_term_active | acoustic_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_exact_rsf | dV/dt = -8.983782e+01 + 6.996144e+00*tau - 5.382505e-02*sigmaN_logV | 3 | 4.259328e+01 | 6.303842e+00 | 5.050176e+00 | -2.767153 | 8.624414e+03 | 0.029 | 47.706 | 10.101 | 28.903 | False | False | False | avg_timeshift |
| B_reduced_rsf | dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV | 3 | 7.386781e+02 | 2.711641e+01 | 2.538210e+01 | -98.203969 | 4.493035e+02 | 0.006 | 16.556 | 15.378 | 15.967 | True | False | False | avg_timeshift |
| C_local_memory | dV/dt = 4.137464e+01 - 9.507421e-01*sigmaN_logV - 3.137850e-01*deltaS_orth | 3 | 2.039873e+03 | 4.514662e+01 | 3.890302e+01 | -322.110480 | 1.759258e+09 | 0.002 | 13.677 | 15.505 | 14.591 | True | False | False | avg_timeshift |
| D_acoustic_augmented | dV/dt = 8.303750e+02 + 7.141577e+01*tau - 9.316535e+01*sigmaN - 1.057892e+00*sigmaN_logV + 3.526699e+02*acoustic_feature | 5 | 8.864101e+02 | 2.901520e+01 | 2.687087e+01 | -181.687560 | 6.945357e+02 | 0.010 | 0.398 | 15.307 | 7.852 | True | False | True | avg_timeshift |

## Exact RSF attempt
- `dV/dt = -8.983782e+01 + 6.996144e+00*tau - 5.382505e-02*sigmaN_logV`
- Theta coefficient active: `False`
- Exact-model holdout derivative MSE: `4.259328e+01`
- Exact-model holdout derivative RMSE: `6.303842e+00`
- Exact-model holdout derivative MAE: `5.050176e+00`
- Exact-model holdout derivative R^2: `-2.767153`
- Exact-model mean stable rollout fraction: `0.029`
- Exact-model mean peak timing error: `47.706` s
- Exact-model mean onset timing error: `10.101` s

## Final selected velocity model
- Selected model: `B_reduced_rsf`
- `dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV`
- Holdout derivative MSE: `7.386781e+02`
- Holdout derivative RMSE: `2.711641e+01`
- Holdout derivative MAE: `2.538210e+01`
- Holdout derivative R^2: `-98.203969`
- Mean stable rollout fraction: `0.006`
- Mean peak timing error: `16.556` s
- Mean onset timing error: `15.378` s

## Sign checks
| Check | Expected | Result | Status |
| --- | --- | --- | --- |
| tau_drive_positive | positive | 8.812200e-03 | ✅ |
| tau_positive | positive | 6.996144e+00 | ✅ |
| sigmaN_logV_negative | negative | -5.382505e-02 | ✅ |
| sigmaN_logTheta_negative | negative | -4.758896e-18 | ❌ |

## De-normalized coefficients
- Tau coefficients: `{"1": -0.02557013762458702, "V": 0.0003455056134759978, "V_drive_minus_V": 0.008812200358494162}`
- Final velocity coefficients: `{"1": 793.7417517685274, "tau": 1.737169796070437e-18, "sigmaN": -39.526398499222616, "sigmaN_logV": -0.9497243654872072}`

## Parameter mapping
- `m = 1 / beta_tau = 5.756490e+17`
- `mu0 = -beta_sigma / beta_tau = 2.275333e+19`
- `a = -beta_sigmaV / beta_tau = 5.467079e+17`
- `b = -beta_sigmaTheta / beta_tau = nan`
- For reduced or surrogate models, only the parameters attached to present coefficients are interpretable.

## Why the theta term collapsed
- Exact-model theta coefficient: `-4.758896e-18`
- Theta residual fraction after removing [1, tau, sigmaN, sigmaN_logV]: `4.087314e-01`
- Partial correlation of residualized theta term with residualized dV/dt: `2.338966e-01`
- Exact-design condition number: `4.917298e+04`
- SigmaN coefficient of variation in the exact design: `6.864859e-04`

## Validation
- Semi-observed tau rollout rows: `[{"step_name": "p5838_step2", "tau_rollout_mse": 0.9795507278396812}, {"step_name": "p5838_step7", "tau_rollout_mse": 0.48986632548254055}]`
- Exact-RSF rollout rows: `[{"step_name": "p5838_step2", "rollout_mse": 10855.979873900409, "stable_fraction": 0.035333333333333335, "peak_timing_error_s": 48.26499999999942, "onset_timing_error_s": 15.118000000000393}, {"step_name": "p5838_step7", "rollout_mse": 6392.847440663348, "stable_fraction": 0.023666666666666666, "peak_timing_error_s": 47.146999999999025, "onset_timing_error_s": 5.083000000000538}]`
- Reduced-RSF rollout rows: `[{"step_name": "p5838_step2", "rollout_mse": 446.2289938853352, "stable_fraction": 0.007666666666666666, "peak_timing_error_s": 26.949000000000524, "onset_timing_error_s": 21.433000000000902}, {"step_name": "p5838_step7", "rollout_mse": 452.3779200059525, "stable_fraction": 0.004666666666666667, "peak_timing_error_s": 6.162999999998647, "onset_timing_error_s": 9.32300000000032}]`
- Local-memory rollout rows: `[{"step_name": "p5838_step2", "rollout_mse": 2117429942.7857895, "stable_fraction": 0.002, "peak_timing_error_s": 18.873999999999796, "onset_timing_error_s": 21.55000000000109}, {"step_name": "p5838_step7", "rollout_mse": 1401086424.9062774, "stable_fraction": 0.0016666666666666668, "peak_timing_error_s": 8.480000000001382, "onset_timing_error_s": 9.460000000000946}]`
- Acoustic rollout rows: `[{"step_name": "p5838_step2", "rollout_mse": 103.87950298803374, "stable_fraction": 0.01633333333333333, "peak_timing_error_s": 0.18699999999989814, "onset_timing_error_s": 21.409999999999854}, {"step_name": "p5838_step7", "rollout_mse": 1285.1919408881563, "stable_fraction": 0.0033333333333333335, "peak_timing_error_s": 0.6080000000001746, "onset_timing_error_s": 9.204999999999927}]`

## Comparison against previous surrogate-heavy paths
- Previous memory-model divergence: `13.866` s
- Previous delay-model theta correlation: `-0.0224`
- Previous ablation Model B divergence: `7.809` s
- Previous ablation Model C divergence: `3.559` s

## Acoustic feature test
- Acoustic feature tested: `avg_timeshift`
- Acoustic model derivative MSE changed from `7.386781e+02` to `8.864101e+02` and peak timing error from `16.556` s to `0.398` s.
- Acoustic residual std fraction after removing [1, tau, sigmaN, sigmaN*log(V/V0)]: `3.718864e-01`; residual partial correlation with dV/dt: `5.199874e-02`.

## Conclusion
- `Exact equation (2) not identifiable from current data; best fallback model is B_reduced_rsf`
