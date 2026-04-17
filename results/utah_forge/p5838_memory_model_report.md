# Utah FORGE p5838 memory-model refinement

## Data split
- Training RSFit-aligned steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`
- Holdout long steps: `p5838_step2, p5838_step7`

## Best model
- Smoothing: `moderate`
- Derivative method: `spline`
- Sparsity threshold: `0.002`
- Rolling window: `20` samples
- EMA span: `20` samples
- Optional average features: `False`
- Cancellation detected: `False`
- Mean holdout divergence time: `13.866` s
- Long-rollout improvement beyond 8 s: `True`

## Equations
```text
dtau/dt = - 2.590e-02*1 + 3.470e-04*V + 8.961e-03*V_drive_minus_V
dV/dt = 1.082e+03*tau - 2.120e-01*V + 8.588e+01*logV - 2.598e+00*tau*logV - 5.354e+02*tau_avg - 5.539e+02*tau_ema - 2.586e-01*S
```

## Theta alignment
- Mean corr(tau_avg, theta): `-0.0437`
- Mean corr(tau_ema, theta): `-0.0431`
- Mean corr(S, theta): `-0.4213`

## Holdout rollout
- Mean holdout rollout error: `0.4392`
- Mean holdout divergence time: `13.8660` s

## Notes
- Raw `tau(t-Δ)` lag terms were removed from the feature library.
- Memory is represented through `tau_avg`, `tau_ema`, and cumulative slip `S`.
- `log(V)` was retained as a required term in the `V` equation.
