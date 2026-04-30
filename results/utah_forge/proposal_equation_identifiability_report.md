# Proposal Equation Identifiability Report

## Usable sample counts
| step_name | usable_samples | time_start | time_end |
| --- | --- | --- | --- |
| p5838_step3 | 3000 | 11634.704 | 11657.909 |
| p5838_step8 | 3000 | 12573.598 | 12596.691 |
| p5838_step9 | 3000 | 12594.214 | 12601.565 |

Total usable theta-mask training samples: `9000`

## Feature dynamic range
| feature | mean | variance | std | min | max | range |
| --- | --- | --- | --- | --- | --- | --- |
| tau | 13.486182898963188 | 0.035609149626477155 | 0.18870386754509605 | 13.140642581138485 | 14.073509883579398 | 0.9328673024409131 |
| sigmaN | 19.034353656641603 | 0.0001707416688385924 | 0.013066815558451584 | 18.986840599672643 | 19.07884438921031 | 0.0920037895376673 |
| logV_over_V0 | 0.8033470178556704 | 0.2952533792288554 | 0.5433722289819893 | -0.14730041029600902 | 1.7917047746816188 | 1.9390051849776277 |
| logThetaV0_over_Dc | -0.7823628131968325 | 0.36246120822927475 | 0.602047513265585 | -1.6859963826286628 | 18.45733515300824 | 20.143331535636904 |
| sigmaN_logV | 15.29106398763239 | 106.9724507594669 | 10.342748704259758 | -2.8050671475353908 | 34.123432961833075 | 36.928500109368464 |
| sigmaN_logTheta | -14.89162880857483 | 131.3390027499271 | 11.460322977557269 | -32.13331334685603 | 351.53183963123166 | 383.6651529780877 |

## Pairwise correlations
| index | tau | sigmaN | logV_over_V0 | logThetaV0_over_Dc | sigmaN_logV | sigmaN_logTheta |
| --- | --- | --- | --- | --- | --- | --- |
| tau | 1.0 | -0.077765 | -0.123633 | 0.144228 | -0.123743 | 0.144326 |
| sigmaN | -0.077765 | 1.0 | -0.017923 | 0.018008 | -0.01691 | 0.017113 |
| logV_over_V0 | -0.123633 | -0.017923 | 1.0 | -0.912149 | 0.999999 | -0.912093 |
| logThetaV0_over_Dc | 0.144228 | 0.018008 | -0.912149 | 1.0 | -0.912149 | 0.999999 |
| sigmaN_logV | -0.123743 | -0.01691 | 0.999999 | -0.912149 | 1.0 | -0.912094 |
| sigmaN_logTheta | 0.144326 | 0.017113 | -0.912093 | 0.999999 | -0.912094 | 1.0 |

## Multicollinearity diagnostics
- Condition number for [1, tau, sigmaN*log(V/V0), sigmaN*log(theta*V0/Dc)]: `2.072709e+03`
- Singular values: `[2707.245102017601, 685.6934193396112, 303.0837027690928, 1.3061385075903915]`
- Default numerical rank: `4` / `4` columns
- Tight-tolerance numerical rank: `4` / `4` columns
- Near-rank-deficient: `False`
| feature | vif | r2_against_others |
| --- | --- | --- |
| tau | 1.028415649964109 | 0.027630511034230976 |
| sigmaN | 1.0069175223197468 | 0.006869998948682676 |
| sigmaN_logV | 5.951656045416262 | 0.8319795377338444 |
| sigmaN_logTheta | 5.985826062450212 | 0.8329386805485183 |

## Intercept and sigmaN redundancy
- Intercept + sigmaN condition number: `2.780377e+04`
- Intercept + sigmaN rank: `2`
- Intercept + sigmaN redundant: `True`

## Theta screening comparison
- Event-valid training steps: `3`
- Sample-valid training steps: `3`
- Event-valid training rows: `9000`
- Sample-valid training rows: `9000`
- Event-to-sample row gap fraction: `0.000000e+00`
- Event-level rejection too coarse: `False`

## Interpretation
- SigmaN coefficient of variation within the theta-usable training matrix: `6.864859e-04`
- Residual sigmaN variation after projecting onto the intercept: `1.000000e+00` of original std
- Residual theta-feature variation after projecting onto the intercept: `1.000000e+00` of original std
- Residual theta-feature variation after projecting onto [1, tau, sigmaN, sigmaN_logV]: `4.087314e-01` of original std
- Partial correlation between residualized theta feature and residualized dV/dt: `2.338966e-01`
- Theta variation too weak after filtering: `False`

## Hard diagnosis
- Summary: `multicollinearity_or_structural_non_identifiability`
- Implementation bug: `False`
- Alignment / units bug: `False`
- Over-filtering: `False`
- Insufficient theta variation: `False`
- Multicollinearity / structural non-identifiability: `True`

## Diagnosis notes
- The theta term collapses when its residual variation, after removing the baseline RSF predictors, is too small to support a stable independent coefficient.
- If sigmaN is nearly constant, then sigmaN behaves almost like an intercept and cannot be separated cleanly from mu0-driven terms.
- Large VIF, large condition number, or an effectively low-rank exact-RSF matrix indicate structural non-identifiability on the current Utah FORGE subsets.
