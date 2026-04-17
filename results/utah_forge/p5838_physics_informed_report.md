# Utah FORGE p5838 physics-informed SINDy report

## Best configuration
- Smoothing: `strong`
- Derivative method: `savgol`
- Delay delta: `1` samples
- Sparsity threshold: `0.01`

## Recovered equations
- `dtau/dt = - 4.017e-01*1 + 2.917e-01*tau_z + 9.417e-04*V`
- `dV/dt = 6.598e+04*tau_z + 1.211e-01*V - 3.123e+01*logV_z - 6.596e+04*tau_lag_z`

## Success criteria
- `tau_dot` includes `V`: `True`
- `V_dot` includes `tau`: `True`
- `V_dot` includes `log(V)` or delayed terms: `True`
- Stable rollout across selected cycles: `True`
- Pipeline successful under requested criteria: `True`

## Errors
- `tau` residual: `0.9124`
- `V` residual: `0.8259`
- Mean rollout relative error: `0.4148`

## Notes
- `tau_z` and `logV_z` denote zero-mean, unit-variance transformed variables used for conditioning.
- Delay terms are sample delays over the downsampled cycle grid and serve as hidden-state surrogates for missing rate-and-state memory.
- Rollouts are evaluated on multiple selected clean cycles using the same fitted model.