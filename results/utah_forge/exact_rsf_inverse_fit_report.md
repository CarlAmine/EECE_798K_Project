# Exact RSF Inverse Fit Report

## Proposal-faithful workflow
- Stage A: sparse structure confirmation using physics-guided candidate libraries
- Stage B: constrained inverse fitting of the coupled RSF system with latent theta

## Train / holdout split
- Train steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`

## SINDy-confirmed structure
- Equation (1) spring-loading confirmed: `True`
- Velocity tau term confirmed: `True`
- Velocity log(V) term confirmed: `True`
- Hidden-state evidence from theta proxy: `False`

## Exact fitted equations
- `dtau/dt = 2.124098e-03*(V_drive - V)`
- `dV/dt = (1/4.948323e-03) * [tau - sigmaN*(3.339335e-01 + 1.065980e-02*log(V/V0) + 2.843257e-02*log(theta*V0/9.964518e+02))]`
- `dtheta/dt = 1 - V*theta/9.964518e+02`

## Parameter estimates
- `{"k": 0.0021240976593914474, "m": 0.004948323305971728, "mu0": 0.33393345418811876, "a": 0.010659795193027904, "b": 0.028432571489141276, "Dc": 996.451816103912, "acoustic_gamma": 0.0}`
- Event-specific theta0 values: `{"p5838_step3": 72156806.07993074, "p5838_step8": 46584185.95339019, "p5838_step9": 9227913.938489342, "p5838_step4": 17755827.21451709, "p5838_step5": 2775187.9439179422, "p5838_step10": 644115.4983014024, "p5838_step2": 100000000.0, "p5838_step7": 100000000.0}`
- Event-specific theta offsets: `{"p5838_step3": -0.32126602743618465, "p5838_step8": -0.23873529058173254, "p5838_step9": -0.4035250247301394, "p5838_step4": -0.20044768148043965, "p5838_step5": -0.9179538372573491, "p5838_step10": -1.4999999999999998}`

## Constraints used
- `k > 0`, `m > 0`, `a > 0`, `b > 0`, `Dc > 0`
- `theta(t) > 0` enforced through positive initialization and clipped forward integration
- `V(t) > 0` enforced through positive forward integration

## Validation
- Mean holdout tau RMSE: `4.292544e-01`
- Mean holdout V RMSE: `7.044738e+00`
- Mean holdout combined rollout error: `5.024655e+00`
- Mean holdout stable fraction: `0.002`
- Mean holdout onset timing error: `4.590` s
- Mean holdout peak timing error: `16.033` s

## Identifiability findings
- SigmaN too constant for clean mu0/a/b separation: `True`
- Weak theta observability: `False`
- Parameter confounding flag: `True`
- Main blocker(s): `near-constant sigmaN, parameter confounding`

## Comparison to current reduced RSF fallback
- Reduced fallback equation: `dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV`
- Reduced fallback stable fraction: `0.006`
- Exact RSF stable fraction: `0.002`
- Reduced fallback peak timing error: `16.556` s
- Exact RSF peak timing error: `16.033` s

## Acoustic branch
- Acoustic branch used: `True`
- Acoustic feature: `avg_timeshift`
- Acoustic holdout error: `3.938228e+00`

## Final conclusion
- `Equation (1) recovered; exact equation (2) implemented and tested directly, but still not identifiable from current Utah FORGE data`
- Blocker(s): `near-constant sigmaN, parameter confounding`
