# Model C Velocity Isolation Comparison

## Experimental setup
- Controlled comparison only: original outputs were left unchanged.
- `theta` from RSFit is treated as time-varying in the main isolated-velocity experiment.
- The constant-theta branch is an ablation only, using per-event median `theta` as the event-constant approximation.
- Equation (2) was isolated from equation (1); the original Model C tau equation was left in place for rollout fairness.
- Tiny isolated RSF-style library: `[1, tau, sigmaN*log(V/V0), sigmaN*log(theta*V0/Dc)]`.

## Original Model C velocity equation
- `dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V`

## Isolated velocity with time-varying RSFit theta
- `dV/dt = 4.833180e+00 - 7.520195e-01*sigmaN_logV - 1.434962e-01*sigmaN_logTheta`

## Isolated velocity with event-constant theta ablation
- `dV/dt = -2.781219e+00 - 5.830604e-01*sigmaN_logV - 1.650363e-01*sigmaN_logTheta`

## Theta variation on the usable subset
- Mean theta coefficient of variation across theta-valid events: `7.399186e-01`
- Mean std of `log(theta)` across theta-valid events: `6.687145e-01`
- Mean std of `sigmaN*log(theta*V0/Dc)` with time-varying theta: `1.274264e+01`
- Mean std of `sigmaN*log(theta*V0/Dc)` with constant-theta ablation: `1.222532e-01`
- Time-varying theta effectively constant on usable subset: `False`

## Comparison table
| variant | equation | active_terms | theta_term_active | holdout_derivative_rmse | holdout_derivative_r2 | holdout_rollout_velocity_rmse | holdout_rollout_combined_rmse | holdout_mean_divergence_s | holdout_stable_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_model_c | dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V | 1|tau|V|logV|logTheta|tau*logV|tau*logTheta|V_drive_minus_V | True | 14.379536170666185 | -7.606171378663172 | 2.284830646100389 | 0.5358454839109198 | 3.558500000001004 | 0.0 |
| isolated_theta_timevarying | dV/dt = 4.833180e+00 - 7.520195e-01*sigmaN_logV - 1.434962e-01*sigmaN_logTheta | 1|sigmaN_logV|sigmaN_logTheta | True | 14.222772624840498 | -7.419547846794597 | 10.336266391336828 | 1.1736205728224953 | 51.22100000000046 | 0.0 |
| isolated_theta_constant_ablation | dV/dt = -2.781219e+00 - 5.830604e-01*sigmaN_logV - 1.650363e-01*sigmaN_logTheta | 1|sigmaN_logV|sigmaN_logTheta | True | 14.082797954796085 | -7.254639893667429 | 15.81873701120831 | 1.3353022828805672 | 18.88550000000032 | 0.5 |

## Interpretation
- Time-varying isolated fit active terms: `['1', 'sigmaN_logV', 'sigmaN_logTheta']`
- Constant-theta ablation active terms: `['1', 'sigmaN_logV', 'sigmaN_logTheta']`
- Time-varying theta term active: `True`
- Constant-theta theta term active: `True`
- The time-varying experiment uses RSFit theta as an external signal, not as a latent state being fit.
- The constant-theta experiment tests whether event-level averaging stabilizes the isolated RSF-style regression or instead removes useful variation.

## Bottom line
- Isolating equation (2) helped like equation (1): `False`
- RSFit theta behaves as a useful time-varying signal: `False`
- Constant-theta ablation helped numerically: `True`
- Remaining bottleneck still near-constant sigmaN and parameter confounding: `True`
