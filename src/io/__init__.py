from __future__ import annotations

import pandas as pd


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def describe_dataframe(df: pd.DataFrame) -> dict:
    return {
        "n_samples": int(len(df)),
        "n_vars": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
    }


from .lanl import load_lanl_train, locate_lanl_raw_file, scan_lanl_reset_indices, sample_lanl_cycle
from .load_fdem_zenodo import load_fdem_zenodo_dataset, locate_fdem_binary, locate_fdem_zenodo_files
from .pangaea import load_pangaea_dataset, locate_pangaea_raw_file, infer_pangaea_column_mapping
from .utah_forge import (
    locate_utah_forge_files,
    load_utah_forge_dataset,
    load_utah_forge_readme,
    infer_utah_forge_column_mapping,
)
