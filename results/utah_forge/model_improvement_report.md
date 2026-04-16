# Utah FORGE model improvement report

## Data coverage
- Available local experiments: p5838
- Missing expected experiments: p5848, p5897, p5905, p5912

## Selected modeling events
- `short`: `p5838_event_017` from `p5838` (duration 11.37 s, score 0.815)
- `medium`: `p5838_event_004` from `p5838` (duration 63.97 s, score 0.616)
- `long`: `p5838_event_008` from `p5838` (duration 106.75 s, score 0.799)

## Best current model
- Best experiment: `p5838`
- Best cycle/event: `medium` (`medium`)
- Preprocessing choice: `strong` smoothing
- Derivative method: `savgol`
- Scaling choice: `none`
- Library degree: `2`
- Sparsity threshold: `1.0e-02`
- Final discovered equations:
  - `dtau/dt = 6.164e+02*1 - 8.993e+01*tau + 3.280e+00*tau^2`
  - `dV/dt = 1.825e+04*1 - 2.674e+03*tau - 1.431e+00*V + 9.797e+01*tau^2 + 3.894e-02*V^2`
- Interpretation quality: now scientifically interpretable at baseline level
- Next bottleneck: missing nonlinear terms

## Notes
- The current local Utah FORGE folder contains only `p5838_datatable.mat`, so the cross-experiment comparison is limited to the locally available raw file set.
- The best equations are written in physical `tau` and `V` coordinates after converting the fitted scaled model back into original units.
