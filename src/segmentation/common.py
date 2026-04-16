from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..preprocess.common import smooth_series


def extract_segment(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    return df.iloc[start_idx:end_idx].reset_index(drop=True).copy()


def extract_segments_df(df: pd.DataFrame, segments: list[tuple[int, int]]) -> list[pd.DataFrame]:
    return [extract_segment(df, start, end) for start, end in segments]


def segment_between_peaks(signal: np.ndarray, prominence: float | None = None, min_cycle_length: int = 25) -> list[tuple[int, int]]:
    if prominence is None:
        prominence = 0.25 * float(np.std(signal))
    peaks, _ = find_peaks(signal, prominence=prominence)
    segments = []
    for index in range(len(peaks) - 1):
        start, end = int(peaks[index]), int(peaks[index + 1])
        if end - start >= min_cycle_length:
            segments.append((start, end))
    return segments


def segment_by_stress_drops(
    df: pd.DataFrame,
    tau_col: str = "tau",
    smooth_window: int = 31,
    smooth_polyorder: int = 3,
    prominence: float | None = None,
    min_cycle_length: int = 25,
) -> list[tuple[int, int]]:
    tau = df[tau_col].to_numpy(dtype=float)
    smoothed = smooth_series(tau, window=smooth_window, polyorder=smooth_polyorder)
    return segment_between_peaks(smoothed, prominence=prominence, min_cycle_length=min_cycle_length)


def summarize_segments(segments: list[tuple[int, int]]) -> dict:
    lengths = [end - start for start, end in segments]
    return {
        "count": int(len(segments)),
        "lengths": lengths,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
    }

