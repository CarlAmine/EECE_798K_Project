from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..derivatives import derivative_savgol, derivative_spline


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned.reset_index(drop=True)


def smooth_series(values: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return values
    if window % 2 == 0:
        window += 1
    if window > len(values):
        window = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window <= polyorder:
        window = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    return savgol_filter(values, window, polyorder)


def standardize_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    return scale_columns(df, columns, method="zscore")


def scale_columns(df: pd.DataFrame, columns: list[str], method: str = "zscore") -> tuple[pd.DataFrame, dict]:
    result = df.copy()
    method = method.lower()
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}

    if method == "none":
        for column in columns:
            centers[column] = 0.0
            scales[column] = 1.0
        return result, {"method": method, "centers": centers, "scales": scales}

    for column in columns:
        values = result[column]
        if method == "zscore":
            center = float(values.mean())
            scale = float(values.std())
        elif method == "robust":
            center = float(values.median())
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            scale = q3 - q1
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        if scale == 0 or not np.isfinite(scale):
            scale = 1.0
        centers[column] = center
        scales[column] = scale
        result[column] = (values - center) / scale

    metadata = {"method": method, "centers": centers, "scales": scales}
    if method == "zscore":
        metadata["means"] = centers.copy()
        metadata["stds"] = scales.copy()
    elif method == "robust":
        metadata["medians"] = centers.copy()
        metadata["iqr"] = scales.copy()
    return result, metadata


def derive_velocity_from_displacement(
    df: pd.DataFrame,
    displacement_col: str,
    time_col: str,
    output_col: str = "V",
    method: str = "savgol",
    window: int = 31,
    polyorder: int = 3,
) -> pd.DataFrame:
    result = df.copy()
    values = result[displacement_col].to_numpy(dtype=float)
    time = result[time_col].to_numpy(dtype=float)
    if method == "savgol":
        result[output_col] = derivative_savgol(values, t=time, window=window, polyorder=polyorder)
    elif method == "spline":
        result[output_col] = derivative_spline(values, t=time)
    else:
        raise ValueError(f"Unsupported velocity derivation method: {method}")
    return result


def build_canonical_modeling_frame(
    df: pd.DataFrame,
    time_col: str,
    state_cols: list[str],
    state_labels: list[str],
) -> tuple[pd.DataFrame, dict]:
    modeling_df = pd.DataFrame(
        {
            "time": df[time_col].to_numpy(dtype=float),
            "state_1": df[state_cols[0]].to_numpy(dtype=float),
            "state_2": df[state_cols[1]].to_numpy(dtype=float),
        }
    )
    metadata = {
        "time_column": time_col,
        "state_columns": state_cols,
        "state_labels": {
            "state_1": state_labels[0],
            "state_2": state_labels[1],
        },
    }
    return modeling_df, metadata
