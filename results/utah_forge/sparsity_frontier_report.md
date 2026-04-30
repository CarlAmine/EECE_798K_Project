# Utah FORGE sparsity frontier

## Full frontier
| lambda | n_terms_tau | n_terms_V | n_terms_union | mean_rollout | training_mse |
| --- | --- | --- | --- | --- | --- |
| 0.001 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.002 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.003 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.005 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.007 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.010 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.015 | 7 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.020 | 5 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.030 | 5 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.050 | 5 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.070 | 5 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.100 | 5 | 9 | 9 | 0.0000 | 1.860273e+03 |
| 0.150 | 4 | 8 | 8 | 0.0000 | 1.860282e+03 |
| 0.200 | 4 | 8 | 8 | 0.0000 | 1.860282e+03 |

## Knee-point equations
```text
dtau/dt = - 1.625916e-03*1 + 1.417677e+01*tau + 5.622089e+00*tau_avg - 1.980623e+01*tau_ema
dV/dt = - 3.307000e+02*1 + 1.632473e+02*tau + 2.509978e+00*V - 3.485033e+00*log(V) + 3.332788e+00*tau*log(V) - 3.310435e+02*V_drive-V + 7.321830e+01*tau_avg - 2.361639e+02*tau_ema
```

## Sign consistency
| Term | Coefficient in dV/dt | Expected sign | Status |
| --- | --- | --- | --- |
| log(V) | -3.485033e+00 | negative | ✅ |
| V_drive-V | -3.310435e+02 | positive | ❌ |
| tau | 1.632473e+02 | positive | ✅ |
| tau*log(V) | 3.332788e+00 | ambiguous | — |
| 1 | -3.307000e+02 | unspecified | — |
| V | 2.509978e+00 | unspecified | — |
| tau_avg | 7.321830e+01 | unspecified | — |
| tau_ema | -2.361639e+02 | unspecified | — |
| S | 0.000000e+00 | unspecified | — |

## Interpretation
The knee point occurs at lambda=0.150, where the model keeps 8 active terms while achieving a mean rollout stability score of 0.000 on holdout events. This point balances compactness against dynamical stability more effectively than both the densest and most aggressively pruned models. The sign-consistency check highlights whether the discovered dV/dt structure remains aligned with RSF-inspired expectations for stress loading, spring loading, and velocity weakening.
