# Utah FORGE p5838 final reviewer ablation report

## System description
- RSFit-aligned Penn State/Utah FORGE biaxial stick-slip steps were used.
- Training steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`

## What each model learned
### Model A
- Tau equation: `dtau/dt = 3.129e+00*1 - 2.193e-01*tau + 9.743e-04*V - 1.183e+00*logV + 8.300e-02*tau*logV + 9.229e-03*V_drive_minus_V`
- V equation: `dV/dt = - 2.320e+03*1 + 1.728e+02*tau + 5.763e-01*V + 5.165e+02*logV - 3.904e+01*tau*logV + 5.751e+00*V_drive_minus_V`
- Mean holdout divergence: `10.662` s
- AIC/BIC: `278275.326` / `278375.743`
- Structural criteria: tau-drive=`True`, V-tau=`True`, V-log=`True`
### Model B
- Tau equation: `dtau/dt = 1.081e+01*1 + 1.082e-03*V + 8.479e-03*V_drive_minus_V + 7.072e+00*tau - 2.706e+00*logV + 1.941e-01*tau*logV - 4.686e+00*tau_avg - 3.169e+00*tau_ema + 6.116e-05*S`
- V equation: `dV/dt = - 1.245e+03*1 + 5.023e-01*V + 5.504e+00*V_drive_minus_V + 5.823e+02*tau + 3.072e+02*logV - 2.281e+01*tau*logV - 1.960e+02*tau_avg - 2.934e+02*tau_ema - 1.294e-01*S`
- Mean holdout divergence: `7.809` s
- AIC/BIC: `276937.172` / `277087.797`
- Structural criteria: tau-drive=`True`, V-tau=`True`, V-log=`True`
### Model C
- Tau equation: `dtau/dt = - 3.755e+00*1 + 2.852e-01*tau + 1.727e-04*V + 1.102e+00*logV + 8.855e-02*logTheta - 8.525e-02*tau*logV - 6.983e-03*tau*logTheta + 1.056e-02*V_drive_minus_V`
- V equation: `dV/dt = 2.166e+03*1 - 1.561e+02*tau + 1.142e+00*V - 8.632e+02*logV - 2.680e+00*logTheta + 6.253e+01*tau*logV + 4.868e-01*tau*logTheta + 5.940e+00*V_drive_minus_V`
- Mean holdout divergence: `3.559` s
- AIC/BIC: `228559.860` / `228691.426`
- Structural criteria: tau-drive=`True`, V-tau=`True`, V-log=`True`

## Structural comparison to RSF target
- The spring-loading structure `k(V_drive - V)` persists across the tau equations.
- At least one model retains `ln(V)` in the V equation.
- Model C tests whether explicit `theta_approx` from RSFit inversion adds value beyond memory surrogates.
- Mean theta correlations: tau_avg=`0.176`, tau_ema=`0.176`, S=`-0.713`.
- Model C log(theta) coefficient: `-2.680` versus mean `b*sigma_n ~ 0.525`.

## Honest assessment
- Model A validates whether observed-only terms are enough: step2 divergence `20.892` s, step7 divergence `0.431` s.
- Model B memory-augmented result under the stricter >10% deviation metric: step2 divergence `3.802` s, step7 divergence `11.816` s. Its older baseline reference was step2=`17.96` s and step7=`9.77` s under the previous metric.
- Model C theta-informed result: step2 divergence `4.507` s, step7 divergence `2.610` s.
- Model B is the most balanced holdout model by worst-case divergence, but it does not dominate Model A on every held-out step.
- Model C retains `ln(theta)` in the discovered V equation, but it does not outperform Model B and its `ln(theta)` coefficient does not line up cleanly with the RSF scale `b*sigma_n`.
- The data therefore support memory-augmented SINDy as a practical surrogate, while explicit theta recovery remains limited by the available RSFit products.

## Reviewer ablations
- Sparsity frontier rows saved: `9`
- Derivative comparison rows saved: `3`
- Theta validation rows saved: `2`

## Regime Dependence and Generalization Limits

### Step Regimes
| Step | Mean tau | Mean V | Dominant f [Hz] | Cycle-period CV |
| --- | --- | --- | --- | --- |
| p5838_step2 | 13.7 | 7.735 | 0.0142 | NA |
| p5838_step3 | 13.7 | 23.49 | 0.04309 | NA |
| p5838_step4 | 13.68 | 79.93 | 0.1799 | NA |
| p5838_step5 | 13.72 | 236.2 | 0.4649 | NA |
| p5838_step7 | 13.6 | 8.63 | 0.01699 | NA |
| p5838_step8 | 13.48 | 23.62 | 0.0433 | NA |
| p5838_step9 | 13.28 | 74.21 | 0.136 | NA |
| p5838_step10 | 13.18 | 230.5 | 0.5672 | NA |

### Holdout Performance vs Regime
| Holdout | Best model | Best divergence [s] | Nearest train step | Distance | Regime match |
| --- | --- | --- | --- | --- | --- |
| p5838_step2 | A | 20.89 | p5838_step3 | 0.8372 | False |
| p5838_step7 | B | 11.82 | p5838_step3 | 0.8988 | False |

The step-regime table shows a clear velocity-frequency ladder: `step2/step7` form the slowest regime, `step3/step8` form an intermediate regime, `step4/step9` form a fast regime, and `step5/step10` form the fastest regime. The holdout steps therefore sit below the training regime envelope in both mean slip velocity and dominant stress-oscillation frequency, so every holdout evaluation is at least partly an extrapolation problem.

Model performance correlates with regime similarity in the sense that generalization is least reliable when the holdout regime is not directly represented in training. Both holdouts are nearest to `step3`/`step8`, but neither one is actually inside the training velocity-frequency range, which explains why different models win on different holdouts.

This suggests that a single global model is insufficient for this dataset, and regime-aware SINDy (training separate models per regime) may be required.

This is physically consistent with RSF theory: different effective `(a-b)` balances can produce qualitatively different frictional regimes, so regime-dependence is expected rather than anomalous.

### Physical Sparsity Window
No physical sparsity window was found in the tested threshold range. The structural criteria stayed satisfied, but every tested threshold kept at least five active terms in each equation, so the current memory-augmented model does not yet admit a genuinely sparse physical summary.
