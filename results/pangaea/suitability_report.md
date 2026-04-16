# PANGAEA suitability report

Dataset label: `pangaea` = parameter-validation dataset

Suitability verdict: Not suitable for the time-series SINDy pipeline with the current local files.

Why it is not suitable
- The local file `data/pangaea/301-U1301_velocity_step_test_RSF.tab` parses successfully, but it is a parameter/results table with 78 rows and 22 columns, not a sampled mechanical time series.
- Parsed columns are fit-result fields such as `V init [um/s]`, `V final [um/s]`, `a`, `b1`, `dc1 [um]`, `b2`, `dc2 [um]`, and associated standard deviations.
- The current local file does not contain a time axis or a continuous shear/friction signal suitable for derivative estimation, event segmentation, or rollout validation.
- The dataset metadata explicitly describes the table as rate-and-state friction parameters obtained from inverse modelling of velocity-step tests.

Loader status
- Loader: `load_pangaea_dataset`
- Status: `ok`
- Raw file used: `C:\Users\carla\Desktop\EECE 798K\Project\data\pangaea\301-U1301_velocity_step_test_RSF.tab`
- Column mapping result: `{'time': None, 'tau': None, 'displacement': None, 'velocity': None}`

What this means for the repo
- Keep `pangaea` as a validation/reference namespace, not an active equation-discovery pipeline, unless raw time-series mechanical files are added later.
- Do not force proxy renaming or synthetic time construction for this dataset.
- No event segmentation or SINDy baseline was run for `pangaea` because the required time-series variables are absent.

What would make it suitable later
- A local raw file with continuous measurements containing at least `time` plus friction/shear stress and either velocity or displacement.
- Ideally, a file that preserves the full velocity-step evolution rather than only fitted RSF parameters.
