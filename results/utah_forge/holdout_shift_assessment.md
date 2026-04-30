# Holdout Shift Assessment

## Holdout z-scores against current training set
| tau_std | tau_range | tau_total_variation_per_s | dtau_std | V_std | V_drive_minus_V_std | step_name | distance_to_train_centroid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.5585441949160125 | 1.8126380685292054 | -1.2208727358210554 | -1.1409614203612988 | -1.20577839171247 | -0.8846356745558286 | p5838_step2 | 2.937079057771051 |
| -0.4596635469399624 | -0.4379989858103106 | -1.256972342135708 | -1.394942660089375 | -1.2413912309820208 | -0.960797714973273 | p5838_step7 | 2.52847041098854 |

## PCA coordinates
| step_name | pc1 | pc2 |
| --- | --- | --- |
| p5838_step5 | 2.0801630478363933 | -1.6374722333318583 |
| p5838_step10 | 4.4757722762508365 | 0.49664191492735904 |
| p5838_step3 | -1.6350662874189659 | -0.5250674397417083 |
| p5838_step8 | -1.9633597052825846 | -1.104609173125835 |
| p5838_step9 | -0.26063650095418084 | -0.047987617302278116 |
| p5838_step4 | 0.5736689677943626 | 1.1924077603367456 |
| p5838_step2 | -1.9251759045736552 | 0.6084493789888054 |
| p5838_step7 | -1.3453658936522062 | 1.0176374092487679 |

## Notes
- Current holdout pair rank by leave-two-out rollout RMSE: `28` / `28`
- Current holdout pair rank by shift distance: `26` / `28`
- Most representative leave-two-out pair by feature-space distance: `p5838_step5, p5838_step2`
- Regime-balanced holdout picked one `flatter` and one `more_variable` step.

