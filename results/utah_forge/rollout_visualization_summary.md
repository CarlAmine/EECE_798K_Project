# Rollout Visualization Summary

## Figures created
- `tau_rollout_examples.png`: observed vs predicted tau and absolute tau error for the two holdout events.
- `tau_v_rollout_examples.png`: observed vs predicted tau plus the observed `V(t)` and supplied `V_drive(t)` used by the semi-observed rollout.
- `phaseplot_rollout_examples.png`: phase plots comparing observed `tau(V)` against predicted tau traced along the observed velocity path.

## Interpretation
- These plots visualize the same semi-observed rollout metric reported in the proposal-recovery report.
- The predicted tau curves come from integrating Equation (1) forward with observed `V(t)` and `V_drive(t)` supplied from the holdout event.
- No separate `V(t)` prediction is available in this metric because the rollout is intentionally semi-observed.
- The event titles `p5838_step2` and `p5838_step7` should be read as holdout event identifiers, not forecast horizons.
