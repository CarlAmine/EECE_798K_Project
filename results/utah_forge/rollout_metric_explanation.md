# Rollout Metric Explanation

## Exact code definition
- Metric source file: `scripts/utah_forge_proposal_equation_recovery.py`
- Metric source function: `semi_observed_tau_rollout()`
- Exact computation: for each holdout segment, the code integrates the tau ODE forward with `odeint`, starting from the observed initial tau, and then computes `mean((tau_roll - observed_tau)^2)`.

## What semi-observed means here
- `semi_observed` means tau is rolled forward while `V(t)` and `V_drive(t)` are not predicted; they are supplied from the observed holdout event as time-varying inputs.
- So this is a tau-only rollout check for Equation (1), not a full joint tau-V forecast.

## What step2 and step7 mean
- `step2` means the holdout event `p5838_step2`.
- `step7` means the holdout event `p5838_step7`.
- In this codebase they are holdout RSFit-aligned Utah FORGE step events, not forecast horizons or checkpoints.

## Which split they are computed on
- Train steps: `p5838_step3, p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10`
- Holdout steps: `p5838_step2, p5838_step7`
- The semi-observed tau rollout metrics are computed only on the holdout events `p5838_step2` and `p5838_step7`.

## Units and scaling
- `tau` is used in the same physical units as the prepared Utah FORGE state in the proposal-recovery workflow.
- `time` is in seconds.
- `tau_rollout_mse` is plain mean squared tau error, so its units are tau-units squared.
- No extra normalization is applied inside `semi_observed_tau_rollout()`.

## Plain-English meaning
- `semi_observed_tau_rollout_mse_step2 = 0.979550727840` means: on holdout event `p5838_step2`, if we drive the recovered tau law using the observed velocity path from that event, the average squared tau error over the event is about `0.9796`.
- `semi_observed_tau_rollout_mse_step7 = 0.489866325483` means: on holdout event `p5838_step7`, the same tau-only rollout check gives an average squared tau error of about `0.4899`.
- Since `0.4899` is smaller than `0.9796`, the recovered tau law tracks `p5838_step7` better than `p5838_step2` under this semi-observed rollout definition.
