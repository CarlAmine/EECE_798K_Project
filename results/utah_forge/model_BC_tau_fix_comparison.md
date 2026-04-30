# Model B/C Tau-Fix Comparison

## Experimental setup
- Controlled comparison only: the original reviewer-ablation outputs were left unchanged.
- Train steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`
- Tau fix: fit `dtau/dt` separately on `[1, V, V_drive_minus_V]` using the proposal-style tau recovery path with a positive drive-term constraint/check.
- Velocity equations were kept on their original Model B / Model C formulations.

## Original Model B equations
- Tau: `dtau/dt = 1.081e+01*1 + 1.082e-03*V + 8.479e-03*V_drive_minus_V + 7.072e+00*tau - 2.706e+00*logV + 1.941e-01*tau*logV - 4.686e+00*tau_avg - 3.169e+00*tau_ema + 6.116e-05*S`
- Velocity: `dV/dt = - 1.245e+03*1 + 5.023e-01*V + 5.504e+00*V_drive_minus_V + 5.823e+02*tau + 3.072e+02*logV - 2.281e+01*tau*logV - 1.960e+02*tau_avg - 2.934e+02*tau_ema - 1.294e-01*S`

## Tau-fixed Model B equations
- Tau: `dtau/dt = -2.590115e-02 + 3.469985e-04*V + 8.961217e-03*V_drive_minus_V`
- Tau one-term approximation: `dtau/dt ~= 8.623992e-03*(V_drive - V)`
- Velocity: `dV/dt = - 1.245e+03*1 + 5.023e-01*V + 5.504e+00*V_drive_minus_V + 5.823e+02*tau + 3.072e+02*logV - 2.281e+01*tau*logV - 1.960e+02*tau_avg - 2.934e+02*tau_ema - 1.294e-01*S`

## Original Model C equations
- Tau: `dtau/dt = - 3.755e+00*1 + 2.852e-01*tau + 1.727e-04*V + 1.102e+00*logV + 8.855e-02*logTheta - 8.525e-02*tau*logV - 6.983e-03*tau*logTheta + 1.056e-02*V_drive_minus_V`
- Velocity: `dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V`

## Tau-fixed Model C equations
- Tau: `dtau/dt = -9.953424e-03 + 1.051457e-02*V_drive_minus_V`
- Tau one-term approximation: `dtau/dt ~= 1.051541e-02*(V_drive - V)`
- Velocity: `dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V`

## Model C theta-valid subset
- Train steps kept after theta validity screening: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step10`
- Holdout steps kept after theta validity screening: `p5838_step2, p5838_step7`

## Comparison table
| family | variant | tau_equation | tau_one_term_equation | V_equation | tau_n_active_terms | tau_drive_positive | tau_drive_abs_share | tau_extra_terms | holdout_tau_derivative_rmse | holdout_tau_derivative_r2 | holdout_velocity_derivative_rmse | holdout_velocity_derivative_r2 | holdout_rollout_combined_rmse | holdout_tau_rollout_rmse | holdout_velocity_rollout_rmse | holdout_mean_divergence_s | holdout_min_divergence_s | holdout_stable_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | original | dtau/dt = 1.081e+01*1 + 1.082e-03*V + 8.479e-03*V_drive_minus_V + 7.072e+00*tau - 2.706e+00*logV + 1.941e-01*tau*logV - 4.686e+00*tau_avg - 3.169e+00*tau_ema + 6.116e-05*S |  | dV/dt = - 1.245e+03*1 + 5.023e-01*V + 5.504e+00*V_drive_minus_V + 5.823e+02*tau + 3.072e+02*logV - 2.281e+01*tau*logV - 1.960e+02*tau_avg - 2.934e+02*tau_ema - 1.294e-01*S | 9 | True | 0.0002959542001872079 | tau|logV|tau*logV|tau_avg|tau_ema|S | 0.07421896206055043 | 0.15284620862822162 | 26.801502536558694 | -28.897750885713837 | 0.5161193171781875 | 0.9262194543778419 | 21.82691402197595 | 7.809000000001106 | 3.802000000001499 | 0.0 |
| B | tau_fixed | dtau/dt = -2.590115e-02 + 3.469985e-04*V + 8.961217e-03*V_drive_minus_V | dtau/dt ~= 8.623992e-03*(V_drive - V) | dV/dt = - 1.245e+03*1 + 5.023e-01*V + 5.504e+00*V_drive_minus_V + 5.823e+02*tau + 3.072e+02*logV - 2.281e+01*tau*logV - 1.960e+02*tau_avg - 2.934e+02*tau_ema - 1.294e-01*S | 3 | True | 0.25451230572014744 |  | 0.07984225599027445 | 0.019611732797954695 | 26.801502536558694 | -28.897750885713837 | 0.10063634907787462 | 0.25242783596651697 | 3.5105233438536247 | 15.748500000000604 | 9.853000000000975 | 1.0 |
| C | original | dtau/dt = - 3.755e+00*1 + 2.852e-01*tau + 1.727e-04*V + 1.102e+00*logV + 8.855e-02*logTheta - 8.525e-02*tau*logV - 6.983e-03*tau*logTheta + 1.056e-02*V_drive_minus_V |  | dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V | 8 | True | 0.0019802893558977883 | tau|logV|logTheta|tau*logV|tau*logTheta | 0.09843411654697573 | -0.49012877580493575 | 14.379536170666185 | -7.606171378663172 | 0.5358454839109198 | 13.156601386889152 | 2.284830646100389 | 3.558500000001004 | 2.610000000000582 | 0.0 |
| C | tau_fixed | dtau/dt = -9.953424e-03 + 1.051457e-02*V_drive_minus_V | dtau/dt ~= 1.051541e-02*(V_drive - V) | dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V | 2 | True | 0.5137079012526794 |  | 0.0766059956561934 | 0.09747751849761477 | 14.379536170666185 | -7.606171378663172 | 0.1060214251446808 | 0.9975527684554231 | 2.956280823544476 | 11.343000000000757 | 11.325000000000728 | 1.0 |

## What changed
- Model B tau fix removed the extra tau terms `['tau', 'logV', 'tau*logV', 'tau_avg', 'tau_ema', 'S']` and replaced them with a compact spring-loading form.
- Model C tau fix removed the extra tau terms `['tau', 'logV', 'logTheta', 'tau*logV', 'tau*logTheta']` and replaced them with the same compact spring-loading form on the theta-valid subset.
- Model B rollout change: combined holdout RMSE delta `-4.154830e-01`, mean divergence delta `7.939500e+00` s.
- Model C rollout change: combined holdout RMSE delta `-4.298241e-01`, mean divergence delta `7.784500e+00` s.
- Model C velocity equation remained theta-informed after the tau fix: `dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V`.

## Scientific interpretation
- The tau fix is transferable as a methodological cleanup: it consistently removes nonphysical tau-side clutter in both Model B and Model C.
- The main effect is on equation (1), not equation (2).
- Model C still inherits the same theta-side limitations on the velocity equation because the tau fix does not create new information about theta or sigmaN variability.

## Presentation Q/A version
Isolating tau with the compact spring-loading library improved both Model B and Model C in the same way: it removed extra tau-side terms without needing a new model family. That tells us the earlier messy tau equations were partly an identification artifact from shared libraries rather than true physics. The velocity equations did not receive the same cleanup because they were intentionally left on their original B/C logic for a fair comparison. Model C therefore still shows the same theta-side weakness in its velocity law after the tau fix. This makes tau isolation a useful methodological lesson to mention in the presentation.
