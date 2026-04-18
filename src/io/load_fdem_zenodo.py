from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FDEM_ZENODO_CONFIG
from ..utils.paths import find_first_existing
from . import describe_dataframe


FDEM_HELPER_FILENAMES = ("submission_function_define.py", "submission_lgbm_model.py", "submisstion_setting.py")
FDEM_BINARY_SHAPE = (25_000, 8_814)
FDEM_SENSOR_BLOCK_WIDTH = 4
FDEM_SENSOR_VALUE_COLUMNS = 8_812
FDEM_TIME_COLUMN = 8_812
FDEM_NSS_COLUMN = 8_813
FDEM_MISSING_NOTE = (
    "The FDEM Zenodo binary p28_data.bin is not present under data/fdem_zenodo/. "
    "Place the published binary beside the helper scripts to enable loader validation."
)


def locate_fdem_zenodo_files(preferred_path: str | Path | None = None) -> dict[str, list[Path]]:
    binary_files: list[Path] = []
    helper_files: list[Path] = []
    other_files: list[Path] = []

    if preferred_path is not None:
        preferred = Path(preferred_path)
        if preferred.exists():
            binary_files.append(preferred)

    for pattern in FDEM_ZENODO_CONFIG.raw_globs:
        for path in sorted(FDEM_ZENODO_CONFIG.raw_dir.glob(pattern)):
            lower = path.name.lower()
            if lower.endswith(".bin"):
                binary_files.append(path)
            elif lower in {name.lower() for name in FDEM_HELPER_FILENAMES}:
                helper_files.append(path)
            else:
                other_files.append(path)

    return {
        "binary_files": sorted(set(binary_files)),
        "helper_files": sorted(set(helper_files)),
        "other_files": sorted(set(other_files)),
    }


def locate_fdem_binary(preferred_path: str | Path | None = None) -> Path:
    if preferred_path is not None:
        candidate = Path(preferred_path)
        if candidate.exists():
            return candidate

    candidates = [FDEM_ZENODO_CONFIG.raw_dir / name for name in FDEM_ZENODO_CONFIG.preferred_raw_names if name.endswith(".bin")]
    inventory = locate_fdem_zenodo_files()
    found = find_first_existing([*candidates, *inventory["binary_files"]])
    if found is None:
        raise FileNotFoundError(FDEM_MISSING_NOTE)
    return found


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def inspect_fdem_helper_scripts() -> dict:
    inventory = locate_fdem_zenodo_files()
    helper_payloads = {path.name: _read_text(path) for path in inventory["helper_files"]}
    reshape_match = None
    for text in helper_payloads.values():
        reshape_match = re.search(r"reshape\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if reshape_match:
            break

    shape = None
    if reshape_match:
        shape = {"n_rows": int(reshape_match.group(1)), "n_cols": int(reshape_match.group(2))}

    uses_time_and_nss = any("t = data_gather[:, -2]" in text and "nss = data_gather[:, -1]" in text for text in helper_payloads.values())
    uses_nss_as_target = any("Ground_truth = nss" in text or "Friction_coef = nss" in text for text in helper_payloads.values())
    return {
        "helper_files": list(helper_payloads),
        "binary_shape_hint": shape,
        "uses_time_and_nss_tail_columns": uses_time_and_nss,
        "uses_nss_as_ground_truth": uses_nss_as_target,
        "source_notes": (
            "The published helper code reshapes p28_data.bin to a 25000 x 8814 float64 matrix, "
            "uses the penultimate column as time, and uses the final column (`nss`) as the LightGBM ground-truth friction signal."
            if shape and uses_time_and_nss and uses_nss_as_target
            else "Could not fully confirm the published binary role assignments from the local helper scripts."
        ),
        "sensor_order_note": (
            "The helper scripts do not explicitly label the within-sensor block order. "
            "This loader therefore uses the repository's published local block assumption [ux, uy, vx, vy] "
            "when computing Ek from the velocity channels."
        ),
    }


def infer_fdem_zenodo_column_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    normalized = {"".join(character.lower() for character in column if character.isalnum()): column for column in df.columns}
    mapping: dict[str, str | None] = {}
    for semantic_name, aliases in FDEM_ZENODO_CONFIG.column_aliases.items():
        resolved = None
        for alias in aliases:
            key = "".join(character.lower() for character in alias if character.isalnum())
            resolved = normalized.get(key)
            if resolved is not None:
                break
        mapping[semantic_name] = resolved
    return mapping


def load_fdem_zenodo_dataset(file_path: str | Path | None = None) -> tuple[pd.DataFrame | None, dict]:
    helper_summary = inspect_fdem_helper_scripts()
    try:
        binary_path = locate_fdem_binary(file_path)
    except FileNotFoundError as exc:
        missing_path = FDEM_ZENODO_CONFIG.raw_dir / "DATA_MISSING.txt"
        missing_path.write_text(str(exc) + "\n", encoding="utf-8")
        summary = {
            "source_url": FDEM_ZENODO_CONFIG.source_url,
            "raw_file": None,
            "file_format": ".bin",
            "schema": None,
            "column_mapping": {},
            "binary_shape": list(FDEM_BINARY_SHAPE),
            "dense_sensor_feature_count": int(FDEM_SENSOR_VALUE_COLUMNS),
            "helper_summary": helper_summary,
            "velocity_mode": FDEM_ZENODO_CONFIG.velocity_mode,
            "analysis_ready": False,
            "notes": FDEM_ZENODO_CONFIG.notes,
            "suitability_note": str(exc),
        }
        return None, summary

    n_rows, n_cols = FDEM_BINARY_SHAPE
    file_size = binary_path.stat().st_size
    expected_values = file_size // np.dtype(np.float64).itemsize
    if expected_values != n_rows * n_cols:
        raise ValueError(
            f"Unexpected FDEM binary size for {binary_path.name}: file contains {expected_values} float64 values, "
            f"which does not match the published shape {n_rows} x {n_cols}."
        )

    data = np.fromfile(binary_path, dtype=np.float64).reshape(n_rows, n_cols)
    time_values = data[:, FDEM_TIME_COLUMN]
    nss_values = data[:, FDEM_NSS_COLUMN]

    # The helper scripts treat the final column `nss` as the published LightGBM target / friction coefficient.
    mu_values = nss_values.copy()

    # The helper code does not expose a direct Ek column, so we compute kinetic energy from the velocity channels.
    # The within-sensor block order is not explicitly labeled in the helper scripts; we retain the repo's published
    # local assumption [ux, uy, vx, vy], so the velocity channels are indices 2 and 3 in each sensor block.
    vx = data[:, 2:FDEM_SENSOR_VALUE_COLUMNS:FDEM_SENSOR_BLOCK_WIDTH]
    vy = data[:, 3:FDEM_SENSOR_VALUE_COLUMNS:FDEM_SENSOR_BLOCK_WIDTH]
    ek_values = 0.5 * np.sum(vx**2 + vy**2, axis=1)

    base = pd.DataFrame(
        {
            "time": time_values,
            "mu": mu_values,
            "Ek": ek_values,
            "nss": nss_values,
        }
    )
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "mu", "Ek", "nss"]).reset_index(drop=True)
    if not base["time"].is_monotonic_increasing:
        base = base.sort_values("time").reset_index(drop=True)
    base = base.loc[base["time"].diff().fillna(1.0) > 0].reset_index(drop=True)

    mapping = infer_fdem_zenodo_column_mapping(base)
    summary = {
        "source_url": FDEM_ZENODO_CONFIG.source_url,
        "raw_file": str(binary_path),
        "file_format": ".bin",
        "schema": describe_dataframe(base),
        "column_mapping": mapping,
        "binary_shape": [int(n_rows), int(n_cols)],
        "dense_sensor_feature_count": int(FDEM_SENSOR_VALUE_COLUMNS),
        "helper_summary": helper_summary,
        "velocity_mode": FDEM_ZENODO_CONFIG.velocity_mode,
        "analysis_ready": True,
        "notes": FDEM_ZENODO_CONFIG.notes,
        "proxy_notes": {
            "mu": "Taken from the final binary column because the helper scripts use `nss` as the published LightGBM ground-truth / friction target.",
            "Ek": "Computed as 0.5 * sum(vx^2 + vy^2) over all 2203 sensors using the repository's established [ux, uy, vx, vy] block order.",
            "nss": "Retained as the original published alias of the final binary column and used as the stick-slip segmentation signal.",
            "h": "Not extracted because the helper scripts do not expose plate/gouge sensor groups needed for a defensible thickness estimate.",
        },
    }
    return base, summary
