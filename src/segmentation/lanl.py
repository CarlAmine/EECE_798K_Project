from __future__ import annotations

import numpy as np
import pandas as pd

from .common import extract_segments_df


def segment_lanl_cycles(df: pd.DataFrame, failure_col: str = "time_to_failure", min_cycle_length: int = 50_000) -> list[tuple[int, int]]:
    resets = np.where(np.diff(df[failure_col].to_numpy(dtype=float)) > 0)[0] + 1
    if len(resets) == 0:
        return [(0, len(df))] if len(df) >= min_cycle_length else []

    starts = np.concatenate(([0], resets))
    ends = np.concatenate((resets, [len(df)]))
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= min_cycle_length]


def extract_lanl_cycle_segments(df: pd.DataFrame, failure_col: str = "time_to_failure", min_cycle_length: int = 50_000) -> list[pd.DataFrame]:
    return extract_segments_df(df, segment_lanl_cycles(df, failure_col=failure_col, min_cycle_length=min_cycle_length))

