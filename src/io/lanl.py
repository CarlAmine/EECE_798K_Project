from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LANL_CONFIG
from ..utils.paths import find_first_existing
from . import validate_columns


def locate_lanl_raw_file(preferred_path: str | Path | None = None) -> Path:
    if preferred_path is not None:
        candidate = Path(preferred_path)
        if candidate.exists():
            return candidate

    candidates = [LANL_CONFIG.raw_dir / name for name in LANL_CONFIG.preferred_raw_names]
    fallback_candidates = [
        LANL_CONFIG.raw_dir.parent / "train.csv",
        LANL_CONFIG.raw_dir.parent / "lanl_train.csvtdunczn5.part",
    ]
    found = find_first_existing([*candidates, *fallback_candidates])
    if found is None:
        raise FileNotFoundError(
            "LANL raw data not found. Place train.csv under data/lanl/train.csv."
        )
    return found


def load_lanl_train(csv_path: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    csv_path = locate_lanl_raw_file(csv_path)
    df = pd.read_csv(csv_path, nrows=nrows)
    validate_columns(df, ["acoustic_data", "time_to_failure"])
    df = df.copy()
    df["time"] = np.arange(len(df), dtype=float)
    return df[["time", "acoustic_data", "time_to_failure"]]


def scan_lanl_reset_indices(csv_path: str | Path | None = None, count: int = 2, chunk_size: int = 250_000) -> list[int]:
    csv_path = locate_lanl_raw_file(csv_path)
    reset_indices: list[int] = []
    prev_ttf = None
    row_offset = 0

    for chunk in pd.read_csv(csv_path, usecols=["time_to_failure"], chunksize=chunk_size):
        ttf = chunk["time_to_failure"].to_numpy()
        if prev_ttf is not None and ttf[0] > prev_ttf:
            reset_indices.append(row_offset)
            if len(reset_indices) >= count:
                break

        local_resets = np.where(np.diff(ttf) > 0)[0] + 1
        for index in local_resets:
            reset_indices.append(row_offset + int(index))
            if len(reset_indices) >= count:
                break

        if len(reset_indices) >= count:
            break

        prev_ttf = float(ttf[-1])
        row_offset += len(ttf)

    return reset_indices


def sample_lanl_cycle(
    start_row: int,
    end_row: int,
    max_rows: int,
    csv_path: str | Path | None = None,
    chunk_size: int = 250_000,
) -> tuple[pd.DataFrame, int]:
    csv_path = locate_lanl_raw_file(csv_path)
    cycle_length = end_row - start_row
    step = max(1, math.ceil(cycle_length / max_rows))
    pieces: list[pd.DataFrame] = []
    row_offset = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_start = row_offset
        chunk_end = row_offset + len(chunk)
        overlap_start = max(start_row, chunk_start)
        overlap_end = min(end_row, chunk_end)
        if overlap_start < overlap_end:
            overlap = chunk.iloc[overlap_start - chunk_start : overlap_end - chunk_start].copy()
            if step > 1:
                global_indices = np.arange(overlap_start, overlap_end)
                overlap = overlap.loc[((global_indices - start_row) % step) == 0].copy()
            pieces.append(overlap)

        if chunk_end >= end_row:
            break
        row_offset = chunk_end

    sampled = pd.concat(pieces, ignore_index=True)
    validate_columns(sampled, ["acoustic_data", "time_to_failure"])
    sampled["time"] = np.arange(len(sampled), dtype=float) * step
    return sampled[["time", "acoustic_data", "time_to_failure"]], step

