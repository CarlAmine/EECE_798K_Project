from __future__ import annotations

import pandas as pd

from .common import segment_by_stress_drops


def segment_utah_forge_events(df: pd.DataFrame, tau_col: str = "tau", min_cycle_length: int = 25) -> list[tuple[int, int]]:
    if tau_col not in df.columns:
        raise ValueError("Utah FORGE segmentation requires a tau column.")
    return segment_by_stress_drops(df, tau_col=tau_col, min_cycle_length=min_cycle_length)

