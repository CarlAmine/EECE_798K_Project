# Multidataset validation status

## `utah_forge` - primary real physics dataset
- Scientific label: primary real physics dataset using the Penn State biaxial experiment files distributed through Utah FORGE.
- Raw files found: 0_README.txt, README.md, p5838_datatable.mat
- Loader status: ok via `load_utah_forge_dataset`
- Variable mapping used: `time -> time`, `tau -> tau`, `V -> v_int`, `displacement -> d_int`, `mu -> mu`
- Variables are physical or proxy: physical (`tau`, `V` from measured `v_int`)
- Segmentation method used: stress-drop / peak-based segmentation on `tau`
- Valid cycle extracted: yes, `results/utah_forge/selected_cycle_001.csv`
- SINDy baseline ran: yes
- Recovered equations: currently weak
- Main outputs:
  - `results/utah_forge/dataset_summary.json`
  - `results/utah_forge/raw_file_inventory.json`
  - `results/utah_forge/selected_cycle_001.csv`
  - `results/utah_forge/discovered_equations.txt`
  - `results/utah_forge/baseline_summary.json`
  - `results/utah_forge/baseline_rollout.png`
- What remains to improve: tighten event selection across more than one experiment file, compare `v_int` against displacement-derived velocity for robustness, and tune the SINDy library/regularization because derivative fit error is still high.

## `lanl` - proxy-state baseline
- Scientific label: proxy-state baseline
- Raw files found: README.md, lanl_train.csvtdunczn5.part, lanl_train_cleaned.csv, lanl_train_processed.csv, train.csv
- Loader status: ok via `load_lanl_train`
- Variable mapping used: `time`, `acoustic_data`, `time_to_failure`, with derived `tau_proxy` and `V_proxy`
- Variables are physical or proxy: proxy (`tau_proxy`, `V_proxy`)
- Segmentation method used: time-to-failure reset segmentation
- Valid cycle extracted: yes, `results/lanl/lanl_selected_cycle.csv`
- SINDy baseline ran: yes
- Recovered equations: currently weak for physical interpretation, but usable as a proxy baseline record
- Main outputs:
  - `results/lanl/dataset_summary.json`
  - `results/lanl/raw_file_inventory.json`
  - `results/lanl/lanl_selected_cycle.csv`
  - `results/lanl/discovered_equations.txt`
  - `results/lanl/baseline_summary.json`
  - `results/lanl/baseline_rollout.png`
- What remains to improve: upgrade the proxy feature construction and cycle selection so the baseline better reflects failure-cycle dynamics instead of just serving as a reference run.

## `pangaea` - parameter-validation dataset
- Scientific label: parameter-validation dataset
- Raw files found: 301-U1301_velocity_step_test_RSF.tab, README.md
- Loader status: ok via `load_pangaea_dataset`
- Variable mapping used: no valid `time`, `tau`, `velocity`, or `displacement` mapping from the current local file
- Variables are physical or proxy: neither for the current pipeline; this is a parameter table
- Segmentation method used: not run
- Valid cycle extracted: no
- SINDy baseline ran: no
- Recovered equations: not applicable
- Main outputs:
  - `results/pangaea/dataset_summary.json`
  - `results/pangaea/raw_file_inventory.json`
  - `results/pangaea/suitability_report.md`
- What remains to improve: obtain local raw time-series mechanical data if `pangaea` is meant to become a secondary real dataset.

## `fdem_zenodo` - simulated granular fault comparison dataset
- Scientific label: simulated granular fault - LightGBM comparison dataset
- Raw files found: README.md, p28_data.bin, submission_function_define.py, submission_lgbm_model.py, submisstion_setting.py
- Loader status: ok via `load_fdem_zenodo_dataset`
- Variable mapping used: `time -> time`, `tau_proxy -> tau`, `V -> spatial mean vy`, `ux_mean -> mean x displacement`, `nss -> cycle index`
- Variables are physical or proxy: mixed; `V` is a spatially averaged simulated velocity, while `tau_proxy` and `mu_proxy` are derived proxies rather than directly measured macroscopic stress/friction
- Segmentation method used: `nss` reset segmentation
- Valid cycle extracted: yes, stacked fit on `results/fdem_zenodo/selected_cycle_001.csv`, `selected_cycle_002.csv`, and `selected_cycle_003.csv`
- SINDy baseline ran: yes
- Recovered equations: normalized-domain coefficients are now well-conditioned, but the proxy-state rollout remains weak
- Main outputs:
  - `results/fdem_zenodo/dataset_summary.json`
  - `results/fdem_zenodo/raw_file_inventory.json`
  - `results/fdem_zenodo/variable_diagnostics.json`
  - `results/fdem_zenodo/cycle_diagnostics.csv`
  - `results/fdem_zenodo/selected_cycle_001.csv`
  - `results/fdem_zenodo/discovered_equations.txt`
  - `results/fdem_zenodo/baseline_summary.json`
  - `results/fdem_zenodo/sindy_vs_lgbm_comparison.md`
- What remains to improve: the raw-state mismatch is fixed, but the discovered model still diverges after about 1 s on the selected long cycles, so this should still be treated as a proxy-state simulated benchmark rather than a strong physics-recovery result.
