from __future__ import annotations

from ..config import LANL_CONFIG
from ..io import describe_dataframe
from ..io.lanl import locate_lanl_raw_file, load_lanl_train, scan_lanl_reset_indices


def build_lanl_summary(nrows: int = 200_000) -> dict:
    df = load_lanl_train(nrows=nrows)
    reset_indices = scan_lanl_reset_indices(count=2)
    return {
        "dataset": LANL_CONFIG.to_summary(),
        "raw_file": str(locate_lanl_raw_file()),
        "inspection_subset_rows": int(len(df)),
        "schema": describe_dataframe(df),
        "reset_indices_preview": reset_indices,
        "analysis_ready": True,
        "state_definition": {
            "state_1": "tau_proxy (smoothed acoustic_data)",
            "state_2": "V_proxy (derivative of tau_proxy)",
        },
    }

