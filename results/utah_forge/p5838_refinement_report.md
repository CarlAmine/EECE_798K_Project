# Utah FORGE p5838 refinement report

## Best single-delay model
- Lags: `1`
- Smoothing: `strong`
- Derivative: `savgol`
- Threshold: `0.01`
- Stable across training cycles: `True`
- Rollout error: `0.4149`

## Physical-unit equations
- `dtau/dt = -1.185399e+01 + 8.554958e-01*tau + 9.460545e-04*V`
- `dV/dt = -4.111009e+02 + 1.934997e+05*tau + 1.211850e-01*V + -3.674913e+01*ln(V) + -1.934550e+05*tau(t-1)`

## RSFit validation
- Mean theta correlation: `-0.0224`
- Mean max cross-correlation: `-0.1535`

## Long unseen rollouts
- Mean unseen rollout error: `nan`
- Mean divergence time: `7.0030` s