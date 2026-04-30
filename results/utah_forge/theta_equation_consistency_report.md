# Theta Equation Consistency Report

## Scope
- This is a conditional consistency check, not independent hidden-state discovery.
- `theta(t)` is supplied from externally reconstructed RSFit theta on the theta-valid Model C subset.
- The question is whether that supplied `theta(t)` is itself reasonably consistent with the RSF state law.

## Usable subset
- Train steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`
- Mean theta coefficient of variation: `7.399186e-01`
- Mean std of log(theta): `6.687145e-01`
- Median reference Dc across usable events: `5.941425e+00`

## Tiny-library fit
- Equation: `dtheta/dt = 4.084008e-05 - 5.703972e-03*Vtheta`
- Implied Dc: `1.753164e+02`
- `c0` absolute error from 1: `9.999592e-01`
- `c1` negative: `True`

## Expanded-library ablation
- Equation: `dtheta/dt = 7.929644e-05 - 5.675455e-03*Vtheta - 3.943165e-07*V - 5.248856e-03*theta`
- Implied Dc from `Vtheta` term: `1.761973e+02`
- `c0` absolute error from 1: `9.999207e-01`
- `c1` negative: `True`

## Comparison table
| variant | equation | active_terms | train_rmse | holdout_rmse | holdout_r2 | c0 | c1_Vtheta | Dc_hat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny_library | dtheta/dt = 4.084008e-05 - 5.703972e-03*Vtheta | 1|Vtheta | 0.0034993821775864266 | 0.027951266488802216 | 0.0003738272971269829 | 4.084007884227115e-05 | -0.005703971964262181 | 175.31642972045213 |
| expanded_library | dtheta/dt = 7.929644e-05 - 5.675455e-03*Vtheta - 3.943165e-07*V - 5.248856e-03*theta | 1|Vtheta|V|theta | 0.003499269782681323 | 0.027950795773575742 | 0.00040749556030839074 | 7.929643963629825e-05 | -0.005675455111827374 | 176.197323438617 |

## Interpretation
- If the tiny library already works well, that supports internal self-consistency of RSFit theta with the RSF state law.
- If the expanded library is required to fit well, then the supplied theta trajectory is only weakly consistent with the expected state dynamics after alignment/filtering.
- Expanded library changed the conclusion materially: `False`
- Overall consistency-check strength: `weak`

## Presentation Q/A paragraph
This test does not discover theta from raw data; it only asks whether externally reconstructed RSFit theta behaves the way the third RSF equation says it should. We fit `dtheta/dt` using the minimal physical form `1 - (V*theta)/Dc` and compared it against a slightly expanded nuisance-term ablation. If the tiny law already matches well, that supports RSFit theta as a self-consistent state signal. If it needs the expanded library or gives an implausible `Dc`, then the theta reconstruction is only weakly consistent with the expected state dynamics after alignment and filtering.
