# FDEM Zenodo

Source: <https://zenodo.org/records/7370626>

Expected local files:

- `p28_data.bin`
- `submission_function_define.py`
- `submission_lgbm_model.py`
- `submisstion_setting.py`

Optional local sidecar for mechanical channel mapping:

- `p28_mechanical.csv`
- `p28_mechanical.parquet`
- `mechanical_channels.csv`

Notes:

- The published helper code documents `p28_data.bin` as a `float64` matrix reshaped to `25000 x 8814`.
- The final two columns are used as `time` and `nss` in the published LightGBM workflow.
- A documented slip-velocity channel is still required before the repo can run a true `tau/V` SINDy baseline without inventing proxy variables.
