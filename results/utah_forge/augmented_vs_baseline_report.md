# Utah FORGE augmented vs baseline report

## RSFit inspection
- Local RSFit file present: `False`
- RSFit file path checked: `C:\Users\carla\Desktop\EECE 798K\Project\data\utah_forge\p5838_RSFit3000.mat`
- Inspection result: `missing_local_file`
- Dc available: `False`
- a available: `False`
- b or b1/b2 available: `False`
- mu0 available: `False`
- theta0 available: `False`

## Datatable inspection
- Raw file: `C:\Users\carla\Desktop\EECE 798K\Project\data\utah_forge\p5838_datatable.mat`
- time column: `time`
- tau column: `tau`
- V column: `v_int`
- mu column: `mu`
- Samples: `1856909`
- Time span: `10869.000` s to `12729.868` s

## Theta reconstruction
- Credible theta reconstruction was not possible in this pass.
- I did not invent Dc or a fitted initial state because that would make the rate-and-state augmentation scientifically weak.

## Augmented model comparison
- The 3D augmented model was not run, so there is no evidence yet that [tau, V, theta] improves over the current 2D baseline.
- Current best 2D baseline remains the reference model in `results/utah_forge/best_model_summary.json`.
