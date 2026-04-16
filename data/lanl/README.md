Place LANL raw and derived files for the LANL pipeline here.

Expected raw files:
- `train.csv`
- optional legacy local snapshot: `lanl_train.csvtdunczn5.part`

Current pipeline notes:
- LANL is treated as a proxy-state dataset.
- `tau_proxy` is built from smoothed `acoustic_data`.
- `V_proxy` is derived from the smoothed proxy.

