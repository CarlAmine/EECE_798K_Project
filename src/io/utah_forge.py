from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from ..config import UTAH_FORGE_CONFIG
from ..utils.paths import ensure_directory, repo_root
from . import describe_dataframe


MATLAB_EXE = Path(r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe")
UTAH_FORGE_EXPORT_COLUMNS = ("time", "tau", "v_int", "d_int", "mu")


def locate_utah_forge_files() -> dict[str, list[Path]]:
    mat_files: list[Path] = []
    text_files: list[Path] = []
    for pattern in UTAH_FORGE_CONFIG.raw_globs:
        for path in sorted(UTAH_FORGE_CONFIG.raw_dir.glob(pattern)):
            if path.suffix.lower() == ".mat":
                mat_files.append(path)
            else:
                text_files.append(path)
    return {"mat_files": mat_files, "text_files": text_files}


def load_utah_forge_readme(file_path: str | Path | None = None) -> str:
    resources = locate_utah_forge_files()
    if file_path is not None:
        candidate = Path(file_path)
    else:
        candidate = next((path for path in resources["text_files"] if path.name.lower() == "0_readme.txt"), None)

    if candidate is None or not candidate.exists():
        raise FileNotFoundError(
            "Utah FORGE README not found. Place 0_README.txt and the *_datatable.mat files under data/utah_forge/."
        )
    return candidate.read_text(encoding="utf-8", errors="ignore")


def _normalize_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def infer_utah_forge_column_mapping(df: pd.DataFrame) -> dict:
    normalized = {_normalize_name(column): column for column in df.columns}
    mapping = {}
    for semantic_name, aliases in UTAH_FORGE_CONFIG.column_aliases.items():
        resolved = None
        for alias in aliases:
            resolved = normalized.get(_normalize_name(alias))
            if resolved is not None:
                break
        mapping[semantic_name] = resolved
    return mapping


def _mat_to_frame(file_path: Path) -> pd.DataFrame:
    data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.dtype.names:
            frame = pd.DataFrame({name: np.ravel(value[name]) for name in value.dtype.names})
            if not frame.empty:
                return frame
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.size > 0 and np.issubdtype(value.dtype, np.number):
            candidates.append((key, value))

    if not candidates:
        raise ValueError(f"Could not identify a numeric data table in {file_path}")

    name, matrix = max(candidates, key=lambda item: item[1].shape[0] * item[1].shape[1])
    columns = [f"{name}_{index}" for index in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=columns)


def _posix_path(path: Path) -> str:
    return path.resolve().as_posix()


def _matlab_export_paths(file_path: Path) -> tuple[Path, Path]:
    cache_dir = ensure_directory(UTAH_FORGE_CONFIG.results_dir / "_cache")
    return (
        cache_dir / f"{file_path.stem}_selected_columns.csv",
        cache_dir / f"{file_path.stem}_matlab_summary.json",
    )


def _run_matlab_table_export(file_path: Path) -> tuple[pd.DataFrame, dict]:
    if not MATLAB_EXE.exists():
        raise RuntimeError(
            "Utah FORGE MAT files are stored as MATLAB tables. MATLAB R2024a was not found at the expected path, "
            "so the loader cannot extract the real columns from the local raw file."
        )

    csv_path, summary_path = _matlab_export_paths(file_path)
    helper_dir = repo_root() / "scripts"
    helper_path = helper_dir / "export_utah_forge_table.m"
    ensure_directory(csv_path.parent)
    ensure_directory(summary_path.parent)
    ensure_directory(UTAH_FORGE_CONFIG.results_dir / "_matlab_prefs")

    needs_refresh = (
        not csv_path.exists()
        or not summary_path.exists()
        or csv_path.stat().st_mtime < file_path.stat().st_mtime
        or summary_path.stat().st_mtime < file_path.stat().st_mtime
    )
    if needs_refresh:
        columns_literal = ",".join(UTAH_FORGE_EXPORT_COLUMNS)
        command = (
            f"addpath('{_posix_path(helper_dir)}'); "
            f"export_utah_forge_table('{_posix_path(file_path)}', '{_posix_path(csv_path)}', "
            f"'{_posix_path(summary_path)}', '{columns_literal}');"
        )
        env = os.environ.copy()
        env.setdefault("MATLAB_PREFDIR", str((UTAH_FORGE_CONFIG.results_dir / "_matlab_prefs").resolve()))
        completed = subprocess.run(
            [str(MATLAB_EXE), "-batch", command],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            message = stderr or stdout or "MATLAB export failed without additional output."
            raise RuntimeError(f"MATLAB export failed for {file_path.name}: {message}")

    df = pd.read_csv(csv_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return df, summary


def load_utah_forge_dataset(file_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    resources = locate_utah_forge_files()
    if file_path is not None:
        selected = Path(file_path)
    else:
        selected = resources["mat_files"][0] if resources["mat_files"] else None

    if selected is None or not selected.exists():
        raise FileNotFoundError(
            "Utah FORGE raw MAT data not found. Place one or more *_datatable.mat files under data/utah_forge/."
        )

    matlab_summary: dict | None = None
    try:
        df = _mat_to_frame(selected)
        if len(df.columns) <= 2 or not {"time", "tau"}.intersection(df.columns):
            raise ValueError("SciPy did not recover the MATLAB table columns.")
    except Exception:
        df, matlab_summary = _run_matlab_table_export(selected)
    try:
        readme_preview = load_utah_forge_readme()[:1000]
    except FileNotFoundError:
        readme_preview = ""

    mapping = infer_utah_forge_column_mapping(df)
    summary = {
        "source_url": UTAH_FORGE_CONFIG.source_url,
        "raw_file": str(selected),
        "file_format": ".mat",
        "schema": describe_dataframe(df),
        "column_mapping": mapping,
        "available_variables": matlab_summary.get("variable_names") if matlab_summary else list(df.columns),
        "raw_table_shape": matlab_summary.get("table_shape") if matlab_summary else [int(len(df)), int(len(df.columns))],
        "matlab_export": matlab_summary,
        "velocity_mode": UTAH_FORGE_CONFIG.velocity_mode,
        "readme_preview": readme_preview,
        "analysis_ready": bool(mapping.get("time") and mapping.get("tau") and (mapping.get("velocity") or mapping.get("displacement"))),
        "notes": UTAH_FORGE_CONFIG.notes,
    }
    return df, summary
