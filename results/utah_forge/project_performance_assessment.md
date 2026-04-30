# Utah FORGE Project Performance Assessment

## Final Ranking
- 1. Best tau equation: strongest recovered equation and most defensible compact law.
- 2. Reduced RSF fallback velocity law: best final usable velocity model.
- 3. Closest exact RSF-looking fit: attractive structurally but non-identifiable and weak on holdout rollout.
- 4. Theta consistency check: weak conditional consistency only, not independent recovery.

## Strongest Success
- The compact spring-loading tau law is the strongest result: it is physically correct, stable under alternate splits, and remains the cleanest recovered governing equation.

## Biggest Weakness
- The theta-bearing exact RSF branch remains scientifically weak because near-constant sigmaN and parameter confounding prevent trustworthy identification even after multistart convergence improves.

## Master Comparison
| result_label | workflow_source | equation_text | training_split | holdout_split | derivative_mse | derivative_rmse | derivative_mae | derivative_r2 | rollout_rmse | rollout_rmse_train | stable_fraction | onset_timing_error_s | peak_timing_error_s | identifiability_status | interpretability_status | scientific_judgment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best tau equation | scripts/utah_forge_proposal_equation_recovery.py | dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V | p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10 | p5838_step2, p5838_step7 | 0.00613915 | 0.0783527 |  |  | 0.844814 | 0.129251 | 1 |  |  | strongly identifiable compact spring-loading law | strongest recovered equation | strongest recovered equation |
| Reduced RSF fallback velocity | scripts/utah_forge_proposal_equation_recovery.py | dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV | p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10 | p5838_step2, p5838_step7 | 738.678 | 27.1164 | 25.3821 | -98.204 | 21.1967 | 70.3048 | 0.00616667 | 15.378 | 16.556 | usable reduced law; theta removed | best final usable velocity law | best final usable model |
| Closest exact RSF-looking fit | scripts/utah_forge_exact_rsf_multistart_check.py | dtau/dt = 2.852472e-03*(V_drive - V); dV/dt = (1/2.519965e-02)*[tau - sigmaN*(9.042632e-12 + 7.146679e-01*log(V/V0) + 1.671089e-01*log(theta*V0/1.518258e+01))]; dtheta/dt = 1 - V*theta/1.518258e+01 | p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10 | p5838_step2, p5838_step7 |  |  |  |  | 392.444 | 21.6041 | 0.00444444 | 4.6215 | 7.7535 | non-identifiable exact-form fit | closest exact fit but non-identifiable | closest exact fit but non-identifiable |
| Theta consistency check | scripts/utah_forge_theta_equation_consistency.py | dtheta/dt = 4.084008e-05 - 5.703972e-03*Vtheta | p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step10 | p5838_step2, p5838_step7 | 0.000781273 | 0.0279513 | 0.00708776 | 0.000373827 |  |  |  |  |  | weak conditional consistency only | weak consistency check on supplied theta | weak consistency check |
| Tau-fixed Model B | scripts/utah_forge_model_bc_tau_fix_comparison.py | dtau/dt = -2.590115e-02 + 3.469985e-04*V + 8.961217e-03*V_drive_minus_V | original B/C split | original B/C holdout | 0.00637479 | 0.0798423 | 0.036101 | 0.0196117 | 0.100636 | 0.035582 | 1 |  |  | supporting ablation | cleaner tau after isolation | supporting ablation result |
| Tau-fixed Model C | scripts/utah_forge_model_bc_tau_fix_comparison.py | dtau/dt = -9.953424e-03 + 1.051457e-02*V_drive_minus_V | theta-screened Model C subset | theta-screened Model C holdout | 0.00586848 | 0.076606 | 0.02903 | 0.0974775 | 0.106021 | 0.0336038 | 1 |  |  | supporting ablation | tau cleaned but velocity still theta-limited | supporting ablation result |

## Train vs Holdout Generalization
- Tau train mean RMSE: `0.1293`; holdout mean RMSE: `0.8448`
- Reduced velocity train mean RMSE: `70.3048`; holdout mean RMSE: `21.1967`
- Exact RSF train mean RMSE: `21.6041`; holdout mean RMSE: `392.4437`
- Holdout representativeness: tau `p5838_step2 is worse than the upper-quartile threshold for this metric. p5838_step7 is worse than the upper-quartile threshold for this metric.`
- Holdout representativeness: exact RSF `p5838_step2 is worse than the upper-quartile threshold for this metric. p5838_step7 is worse than the upper-quartile threshold for this metric.`

## Step Difficulty
| step_name | split | n_samples | duration_s | tau_rollout_rmse | velocity_rollout_rmse | exact_velocity_rollout_rmse | theta_quality | theta_reason | tau_difficulty | reduced_difficulty | exact_difficulty | difficulty_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p5838_step2 | holdout | 3000 | 70.402 | 0.989723 | 21.1241 | 431.172 | high-quality | ok | hard | easy | hard | holdout step; exact RSF unstable early; tau harder than median |
| p5838_step7 | holdout | 3000 | 58.865 | 0.699905 | 21.2692 | 353.715 | high-quality | ok | hard | easy | hard | holdout step; exact RSF unstable early; tau harder than median |
| p5838_step10 | train | 1763 | 1.762 | 0.109793 | 121.302 | 37.4 | low-quality | theta_direct_low_variation,theta_alignment_undefined,too_few_high_quality_samples,short_high_quality_run | medium | hard | medium | theta low-quality; reduced velocity harder than median |
| p5838_step3 | train | 3000 | 23.205 | 0.174618 | 56.9027 | 5.44648 | high-quality | ok | medium | medium | easy | reduced velocity harder than median; tau harder than median |
| p5838_step4 | train | 3000 | 5.558 | 0.123118 | 25.6127 | 21.5241 | low-quality | theta_direct_low_variation,theta_alignment_undefined,too_few_high_quality_samples,short_high_quality_run | medium | medium | medium | theta low-quality |
| p5838_step5 | train | 2151 | 2.15 | 0.079068 | 119.639 | 47.5962 | invalid | theta_direct_invalid,theta_direct_low_variation,theta_alignment_undefined,theta_clipped,too_few_high_quality_samples,short_high_quality_run | easy | hard | hard | theta invalid; reduced velocity harder than median |
| p5838_step8 | train | 3000 | 23.093 | 0.203441 | 57.4292 | 5.54508 | high-quality | ok | hard | hard | easy | reduced velocity harder than median; tau harder than median |
| p5838_step9 | train | 3000 | 7.351 | 0.0854696 | 40.9429 | 12.1124 | high-quality | ok | easy | medium | medium | no major warning |

## Error Decomposition
- The tau law is comparatively strong because it is evaluated in a semi-observed way and only needs to map the observed loading and slip-rate path into a stress evolution; that is much easier than predicting the full velocity trajectory.
- The reduced fallback keeps the most stable local RSF-like ingredient, the negative `sigmaN*log(V/V0)` term, so it performs best as a usable velocity law even though its rollout quality varies a lot across steps.
- The exact RSF fit is structurally attractive because it keeps the full coupled form and can achieve good timing on the original holdout pair, but its rollout error blows up on the hardest holdout steps and its stability remains poor.
- Theta consistency remains weak because the supplied theta signal gives the right sign on the `V*theta` term but not the expected intercept or implied `Dc`, so it is not a strong independent validation of equation (3).

## Identifiability
- `sigmaN` too constant: `True`
- Best JTJ condition number after multistart: `8.904456e+08`
- Best JTJ rank after multistart: `12`
- Parameter estimates stabilized across starts: `False`
- Theta became numerically active: `True`
- Scientific interpretation: multistart solved convergence much better than the baseline exact fit, but it did not solve trustworthiness because the parameter set remains confounded and unstable.

## Final Scientific Judgment
- Strongest result: The compact spring-loading tau law is the strongest result: it is physically correct, stable under alternate splits, and remains the cleanest recovered governing equation.
- Best final usable model: the reduced RSF fallback velocity law.
- Best exact-form result: the multistart exact RSF-looking fit, but it remains non-identifiable.
- Unresolved issue: trustworthy theta-bearing exact RSF recovery on the current Utah FORGE subset.
- Confident claim: Equation (1) is recovered compactly and equation (2) supports a reduced RSF-like form with persistent log-rate structure.
- Cautious claim: the full exact theta-bearing RSF system is implemented and tested directly, but still not scientifically identifiable from the present data.

## Abstract-Style Conclusion
The Utah FORGE p5838 equation-discovery project now supports a compact, physically sensible recovery of the spring-loading stress law and a reduced RSF-style fallback for the velocity law, but not a scientifically trustworthy recovery of the full theta-bearing RSF system. Across saved derivative, rollout, multistep, and multistart diagnostics, the tau equation remains the strongest and most robust result, while the reduced velocity law is the most usable final model despite limited multistep generalization. The closest exact RSF-looking fit is valuable as a structural near-hit because it preserves the full form and achieves decent timing on the original holdout pair, but it fails the trustworthiness test because parameter estimates remain confounded under near-constant sigmaN and holdout rollout degrades sharply outside the training-like steps. Conditional theta checks likewise remain weak, indicating that externally reconstructed theta carries suggestive structure but does not validate a clean standalone RSF state recovery on the current subset.

## Professor-Facing Summary
- Equation (1) is the strongest success: compact, physically interpretable, and robust under bounded split changes.
- The reduced RSF fallback is the best final usable velocity law, not the exact theta-bearing form.
- The original holdout pair was not representative for tau or exact RSF; both holdout steps were unusually hard there.
- The original holdout pair was unusually favorable for the reduced velocity rollout, so that branch should be presented as usable rather than universally strong.
- Multistart improved convergence for exact RSF but did not fix identifiability or parameter trustworthiness.
- The project’s defensible claim is compact stress-law recovery plus reduced RSF structure, not credible exact latent-state RSF recovery.

## Oral Explanation
- The strongest result is the tau equation, because it became compact, physically correct, and stayed robust under bounded split checks.
- The best final usable velocity model is the reduced RSF fallback, not the full exact theta-bearing form.
- The exact RSF-looking fit is still worth showing because it is the closest structural match to the proposal, but it is not trustworthy enough to be the final model.
- The main reason is identifiability, not just optimization: sigmaN is nearly constant and the exact RSF parameters stay confounded across starts.
- When we expanded to all usable steps, the original holdout pair turned out to be unusually hard for tau and exact RSF, but unusually favorable for the reduced velocity rollout.
- So the honest project claim is strong recovery of equation (1), support for a reduced RSF-like equation (2), and no credible full recovery of the theta-bearing exact RSF system.