# Regime-Balanced Tau Evaluation

## Why add a balanced split
- The original `p5838_step2 + p5838_step7` holdout is intentionally retained here as a harsh low-motion stress test baseline.
- In the saved leave-two-out diagnostics, that original pair ranked `28` out of `28` by rollout RMSE, making it the harshest pair tested.
- Its feature-space shift distance was `1.808`, versus `0.488` for the balanced primary pair.

## Regime classification
- Flatter / lower-motion steps: `p5838_step2, p5838_step7, p5838_step4, p5838_step9`
- More typical / higher-motion steps: `p5838_step3, p5838_step5, p5838_step8, p5838_step10`

## Primary balanced split
- Train steps: `p5838_step3, p5838_step4, p5838_step7, p5838_step8, p5838_step9, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step5`
- Why it is more representative: the holdout contains one flatter low-motion step (`p5838_step2`) and one higher-motion step (`p5838_step5`), while training still contains both flatter and more variable regimes.

## Compact tau law
- Equation class kept fixed: `[1, V, V_drive_minus_V]`
- Original-stress-test fit: `dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V`
- Balanced-split fit: `dtau/dt = 1.935726e-04 + 1.027164e-02*V_drive_minus_V`
- Balanced one-term approximation: `dtau/dt ~= 1.027239e-02*(V_drive - V)`

## Split comparison
| split | holdout_steps | equation | mean_derivative_rmse | mean_tau_rollout_rmse | mean_abs_tau_error | max_abs_tau_error |
| --- | --- | --- | --- | --- | --- | --- |
| original_stress_test | p5838_step2, p5838_step7 | dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V | 0.069869 | 0.844826 | 0.734353 | 1.700841 |
| balanced_primary | p5838_step2, p5838_step5 | dtau/dt = 1.935726e-04 + 1.027164e-02*V_drive_minus_V | 0.289449 | 0.092383 | 0.068347 | 0.292021 |

## Per-step holdout rollout errors
| split | step_name | tau_rollout_rmse | mean_abs_tau_error | max_abs_tau_error |
| --- | --- | --- | --- | --- |
| original_stress_test | p5838_step2 | 0.989733 | 0.851697 | 1.700841 |
| original_stress_test | p5838_step7 | 0.699918 | 0.617009 | 1.172269 |
| balanced_primary | p5838_step2 | 0.066734 | 0.044066 | 0.292021 |
| balanced_primary | p5838_step5 | 0.118033 | 0.092628 | 0.255958 |

## Balanced train examples
| step_name | tau_rollout_rmse | mean_abs_tau_error | tau_rollout_r2 |
| --- | --- | --- | --- |
| p5838_step7 | 0.08727770218435868 | 0.07064873055216844 | -0.7120372446688938 |
| p5838_step10 | 0.08767577522883127 | 0.07060196120571709 | -2.2232130692345944 |
| p5838_step3 | 0.09999847425100088 | 0.06951741830881712 | -0.3967845666854375 |
| p5838_step8 | 0.10036225957879281 | 0.07017612721915055 | -0.20858715072873713 |

## Interpretation
- The balanced split improves mean rollout RMSE from `0.845` to `0.092`.
- Mean absolute tau error improves from `0.734` to `0.068`.
- Mean derivative RMSE changes from `0.070` to `0.289`.
- The graph fit improves visibly on the balanced holdout because the new holdout is no longer composed entirely of low-motion steps.
- The original `step2 + step7` split remains useful because it still shows how the compact tau law behaves under a difficult low-motion stress test.

## How to present this result honestly
- Report the original `p5838_step2 + p5838_step7` pair as a low-motion stress test, not as the only estimate of general performance.
- Report the balanced split as a more representative estimate of typical semi-observed tau-law performance across mixed regimes.
- Keep the compact tau equation highlighted as the strongest recovered tau equation, while stating clearly that its hardest failures occur on flatter, low-motion holdouts.
