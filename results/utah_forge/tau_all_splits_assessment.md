# Tau All-Splits Assessment

## Scope
- The compact tau law class is unchanged: `[1, V, V_drive_minus_V]`.
- Tau rollout is semi-observed: `tau(t)` is integrated while observed `V(t)` and `V_drive(t)` are supplied.
- This report broadens evaluation across all usable RSFit-aligned p5838 steps rather than relying on one holdout pair.

## Global summary
| metric | value |
| --- | --- |
| single-step median RMSE | 0.113 |
| leave-two-out median RMSE | 0.145 |
| leave-two-out mean RMSE | 0.266 |
| current pair rank | 28 / 28 |
| balanced pair rank | 1 / 28 |

- Easiest steps overall: p5838_step3, p5838_step9, p5838_step4.
- Hardest steps overall: p5838_step10, p5838_step7, p5838_step2.
- Best leave-two-out pairs: p5838_step2 + p5838_step5, p5838_step3 + p5838_step9, p5838_step3 + p5838_step8.
- Worst leave-two-out pairs: p5838_step7 + p5838_step10, p5838_step2 + p5838_step10, p5838_step2 + p5838_step7.
- A representative single-step holdout by median difficulty is `p5838_step8`.

## Step2 vs Step5
- `step2` is not easier overall: mean holdout RMSE `0.797` vs `0.142` for `step5`.
- `step2` has lower motion roughness after preparation: total-variation-per-second `0.034` vs `0.421`, and `dtau` std `0.107` vs `0.487`.
- `step5` has stronger forcing/context variation: `V_drive - V` std `35.155` vs `2.016`.
- `step2` is not easy because it has large amplitude; it is easier because its prepared trace is smoother and more monotone. Smoothing ratios: total variation prepared/raw `0.063` for `step2` and `0.039` for `step5`.
- `step5` fits worse visually mainly through shape mismatch and stronger curvature, not simply shorter duration. Peak-time-fraction error: `0.312` for `step2` vs `0.664` for `step5`.

## Holdout fairness
- The original stress-test pair `step2 + step7` ranks `28` out of `28` by leave-two-out mean rollout RMSE, so it remains unusually harsh.
- The balanced example `step2 + step5` ranks `1` out of `28`. It is useful as a mixed-regime example, but it is optimistic rather than median-representative for overall pair difficulty.
- Median leave-two-out RMSE is `0.145`; the original pair is `0.845` and the balanced pair is `0.092`.

## Presentation guidance
- Keep the original `step2 + step7` result as a low-motion stress-test reference.
- Present the all-single-step and all-leave-two-out summaries as the main evidence for broad tau-law behavior.
- Use one representative mixed-regime holdout example, but do not let one easy pair stand in for the whole story.
- The strongest honest framing is that the compact tau law works broadly on many prepared steps, but performance is regime-dependent and degrades most on harsher multi-step low-motion or mismatch combinations.

## Selected Leave-Three-Out Checks
| holdout_steps | mean_tau_rollout_rmse | mean_derivative_rmse |
| --- | --- | --- |
| p5838_step2 + p5838_step7 + p5838_step9 | 0.4453099482688514 | 0.1535544022770248 |
| p5838_step2 + p5838_step5 + p5838_step8 | 0.09483253642067273 | 0.24750055983319877 |
| p5838_step3 + p5838_step8 + p5838_step10 | 0.14293678043717598 | 0.44940738444002376 |
| p5838_step4 + p5838_step5 + p5838_step7 | 0.21010677769720357 | 0.28760745408416716 |

## Step Difficulty Table
| difficulty_rank | step_name | mean_holdout_tau_rmse | median_holdout_tau_rmse | difficulty_label |
| --- | --- | --- | --- | --- |
| 1 | p5838_step3 | 0.10025187098044015 | 0.09905405789426264 | easy |
| 2 | p5838_step9 | 0.11288854203557683 | 0.11338459661901187 | easy |
| 3 | p5838_step4 | 0.11588346600409546 | 0.1130074321449824 | easy |
| 4 | p5838_step8 | 0.12013150330144928 | 0.11650900978496817 | medium |
| 5 | p5838_step5 | 0.14248052342773387 | 0.11805488978679682 | medium |
| 6 | p5838_step10 | 0.1846427846539807 | 0.15762256095044874 | hard |
| 7 | p5838_step7 | 0.5269054567287744 | 0.5141265453170688 | hard |
| 8 | p5838_step2 | 0.7968636242741747 | 0.8268269364093729 | hard |

## Pair Difficulty Table
| pair_rank | pair_name | mean_pair_tau_rollout_rmse | mean_pair_derivative_rmse |
| --- | --- | --- | --- |
| 1 | p5838_step2 + p5838_step5 | 0.0923833896817703 | 0.28944891410974655 |
| 2 | p5838_step3 + p5838_step9 | 0.09408985586157534 | 0.24829758575353575 |
| 3 | p5838_step3 + p5838_step8 | 0.09650543183741567 | 0.16795692106121024 |
| 4 | p5838_step5 + p5838_step9 | 0.1014957443294568 | 0.39212889784224525 |
| 5 | p5838_step5 + p5838_step7 | 0.10295559961333897 | 0.2523476663932627 |
| 6 | p5838_step3 + p5838_step4 | 0.10416228669556199 | 0.2701549862179724 |
| 7 | p5838_step4 + p5838_step9 | 0.1043823001962573 | 0.34965819019551064 |
| 8 | p5838_step8 + p5838_step9 | 0.10603031312269152 | 0.24598102494145196 |
| 9 | p5838_step3 + p5838_step5 | 0.10886588860790629 | 0.32159737374229286 |
| 10 | p5838_step5 + p5838_step8 | 0.10912478940724873 | 0.32044274981994125 |
