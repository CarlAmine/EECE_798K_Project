# Multistep Rollout Summary

## Usable steps
- All evaluated RSFit-aligned `p5838` steps: `p5838_step2, p5838_step3, p5838_step4, p5838_step5, p5838_step7, p5838_step8, p5838_step9, p5838_step10`
- Train steps: `p5838_step10, p5838_step3, p5838_step4, p5838_step5, p5838_step8, p5838_step9`
- Holdout steps: `p5838_step2, p5838_step7`

## Exclusions / caveats
- `p5838_step4` is excluded from theta-usable proposal-identifiability subsets because `theta_direct_low_variation,theta_alignment_undefined,too_few_high_quality_samples,short_high_quality_run`.
- `p5838_step5` is excluded from theta-usable proposal-identifiability subsets because `theta_direct_invalid,theta_direct_low_variation,theta_alignment_undefined,theta_clipped,too_few_high_quality_samples,short_high_quality_run`.
- `p5838_step10` is excluded from theta-usable proposal-identifiability subsets because `theta_direct_low_variation,theta_alignment_undefined,too_few_high_quality_samples,short_high_quality_run`.

## Best and worst tau steps
- Best tau rollout steps by RMSE: `[{"step_name": "p5838_step5", "split": "train", "tau_rollout_rmse": 0.07906796070293531}, {"step_name": "p5838_step9", "split": "train", "tau_rollout_rmse": 0.0854695512493597}, {"step_name": "p5838_step10", "split": "train", "tau_rollout_rmse": 0.10979287439698847}]`
- Worst tau rollout steps by RMSE: `[{"step_name": "p5838_step8", "split": "train", "tau_rollout_rmse": 0.2034405160477594}, {"step_name": "p5838_step7", "split": "holdout", "tau_rollout_rmse": 0.699904511688945}, {"step_name": "p5838_step2", "split": "holdout", "tau_rollout_rmse": 0.9897225509402527}]`

## Best and worst reduced-velocity steps
- Best reduced-velocity rollout steps by RMSE: `[{"step_name": "p5838_step2", "split": "holdout", "velocity_rollout_rmse": 21.12413297357634}, {"step_name": "p5838_step7", "split": "holdout", "velocity_rollout_rmse": 21.269177699336485}, {"step_name": "p5838_step4", "split": "train", "velocity_rollout_rmse": 25.61269934735153}]`
- Worst reduced-velocity rollout steps by RMSE: `[{"step_name": "p5838_step8", "split": "train", "velocity_rollout_rmse": 57.429179278837005}, {"step_name": "p5838_step5", "split": "train", "velocity_rollout_rmse": 119.63906958848114}, {"step_name": "p5838_step10", "split": "train", "velocity_rollout_rmse": 121.30231536882653}]`

## Best and worst exact-RSF steps
- Best exact-RSF steps by velocity RMSE: `[{"step_name": "p5838_step3", "split": "train", "velocity_rollout_rmse": 5.446479218151768}, {"step_name": "p5838_step8", "split": "train", "velocity_rollout_rmse": 5.545084851108455}, {"step_name": "p5838_step9", "split": "train", "velocity_rollout_rmse": 12.112448762943112}]`
- Worst exact-RSF steps by velocity RMSE: `[{"step_name": "p5838_step5", "split": "train", "velocity_rollout_rmse": 47.5962437050127}, {"step_name": "p5838_step7", "split": "holdout", "velocity_rollout_rmse": 353.7154666287426}, {"step_name": "p5838_step2", "split": "holdout", "velocity_rollout_rmse": 431.1718583341789}]`

## Interpretation
- The tau equation is evaluated in a semi-observed mode: `tau(t)` is forecast while observed `V(t)` and `V_drive(t)` are supplied as exogenous inputs.
- The reduced velocity and exact-RSF branches are full dynamic velocity rollouts on each step.
- Step representativeness for tau rollout: p5838_step2 is worse than the upper-quartile threshold for this metric. p5838_step7 is worse than the upper-quartile threshold for this metric.
- Step representativeness for exact-RSF velocity rollout: p5838_step2 is worse than the upper-quartile threshold for this metric. p5838_step7 is worse than the upper-quartile threshold for this metric.
- If the exact-RSF gallery shows low stable fractions or large rollout RMSE on many steps, that should be read as evidence that the exact form remains fragile off the original holdout pair.
