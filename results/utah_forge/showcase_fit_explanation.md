# Showcase Fit Explanation

## What rollout means
- A rollout means integrating the discovered differential equation forward in time and comparing the predicted trajectory to the observed holdout event.
- For Equation (1), the rollout shown here is semi-observed: `tau(t)` is predicted, while observed `V(t)` and `V_drive(t)` are supplied as inputs.
- For the reduced fallback Equation (2) and the exact RSF-looking fit, `V(t)` is rolled forward dynamically and compared against the observed velocity trajectory.

## What derivative fit means
- Derivative fit compares the observed numerical derivative, such as `dtau/dt` or `dV/dt`, against the derivative predicted by the equation at the same samples.
- A good derivative fit means the equation matches instantaneous slopes well, even before full rollout is considered.
- Rollout is stricter because small derivative errors can accumulate over time into larger trajectory errors.

## Which plots correspond to which equations
- `showcase_tau_fit.png`: best Equation (1), the compact spring-loading tau law.
- `showcase_velocity_fit.png`: best final usable Equation (2), the reduced RSF fallback.
- `showcase_exact_rsf_fit.png`: closest exact RSF-looking fit on holdout events.
- `showcase_phaseplot.png`: tau-V phase portrait for the exact RSF-looking fit.
- `showcase_derivative_scatter.png`: derivative-fit scatter for Equation (1), the reduced fallback, and the exact RSF-looking fit.

## What a good fit looks like
- In rollout plots, a good fit keeps predicted and observed curves close over most of the event and avoids early divergence.
- In absolute-error plots, a good fit keeps the error low and prevents it from growing rapidly over time.
- In derivative scatter plots, a good fit places points close to the diagonal line.

## Where the exact RSF fit succeeds and fails
- The exact RSF-looking fit succeeds by preserving the full RSF form and achieving reasonably good timing metrics, especially peak timing.
- It fails as a final trusted model because its holdout trajectories are still unstable, parameter estimates are not robust across starts, and the identifiability diagnostics remain poor.
- So it is strongest as a scientific showcase of the closest exact form, not as the final usable governing law.
