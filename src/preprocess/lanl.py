from __future__ import annotations

import numpy as np
import pandas as pd

from ..derivatives import derivative_savgol
from .common import build_canonical_modeling_frame, remove_invalid_rows, smooth_series


def clean_lanl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df[["time", "acoustic_data", "time_to_failure"]].copy()
    cleaned = remove_invalid_rows(cleaned)
    cleaned["time"] = np.arange(len(cleaned), dtype=float)
    return cleaned


def add_lanl_proxies(
    df: pd.DataFrame,
    acoustic_col: str = "acoustic_data",
    time_col: str = "time",
    tau_col: str = "tau_proxy",
    velocity_col: str = "V_proxy",
    smooth_window: int = 101,
    smooth_polyorder: int = 3,
) -> pd.DataFrame:
    result = df.copy()
    smoothed = smooth_series(result[acoustic_col].to_numpy(dtype=float), window=smooth_window, polyorder=smooth_polyorder)
    result[tau_col] = smoothed
    result[f"{acoustic_col}_smoothed"] = smoothed
    result[velocity_col] = derivative_savgol(smoothed, t=result[time_col].to_numpy(dtype=float), window=smooth_window, polyorder=smooth_polyorder)
    return result


def build_lanl_modeling_state(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return build_canonical_modeling_frame(df, time_col="time", state_cols=["tau_proxy", "V_proxy"], state_labels=["tau_proxy", "V_proxy"])

