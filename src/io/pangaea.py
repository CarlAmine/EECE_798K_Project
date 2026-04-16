from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from ..config import PANGAEA_CONFIG
from ..utils.paths import find_first_existing
from . import describe_dataframe


def locate_pangaea_raw_file(preferred_path: str | Path | None = None) -> Path:
    if preferred_path is not None:
        candidate = Path(preferred_path)
        if candidate.exists():
            return candidate

    candidates = [PANGAEA_CONFIG.raw_dir / name for name in PANGAEA_CONFIG.preferred_raw_names]
    glob_candidates: list[Path] = []
    for pattern in PANGAEA_CONFIG.raw_globs:
        glob_candidates.extend(sorted(PANGAEA_CONFIG.raw_dir.glob(pattern)))

    found = find_first_existing([*candidates, *glob_candidates])
    if found is None:
        raise FileNotFoundError(
            "PANGAEA raw data not found. Place the downloaded tab-delimited text file under data/pangaea/."
        )
    return found


def _normalize_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def _read_pangaea_text(file_path: Path) -> tuple[pd.DataFrame, list[str]]:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_index = next((index + 1 for index, line in enumerate(lines) if line.strip() == "*/"), None)
    if header_index is None:
        header_index = None
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if "\t" in line and not stripped.startswith(("#", "/*", "*", "*/")):
                header_index = index
                break

    if header_index is None:
        raise ValueError(f"Could not find a tab-delimited header in {file_path}")

    metadata_lines = lines[:header_index]
    payload = "\n".join(lines[header_index:])
    df = pd.read_csv(StringIO(payload), sep="\t")
    return df, metadata_lines


def infer_pangaea_column_mapping(df: pd.DataFrame) -> dict:
    normalized = {_normalize_name(column): column for column in df.columns}
    mapping = {}
    for semantic_name, aliases in PANGAEA_CONFIG.column_aliases.items():
        resolved = None
        for alias in aliases:
            resolved = normalized.get(_normalize_name(alias))
            if resolved is not None:
                break
        mapping[semantic_name] = resolved
    return mapping


def load_pangaea_dataset(file_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    file_path = locate_pangaea_raw_file(file_path)
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".tab", ".tsv"}:
        df, metadata_lines = _read_pangaea_text(file_path)
    else:
        df = pd.read_csv(file_path)
        metadata_lines = []

    mapping = infer_pangaea_column_mapping(df)
    summary = {
        "source_url": PANGAEA_CONFIG.source_url,
        "raw_file": str(file_path),
        "file_format": suffix or "text",
        "schema": describe_dataframe(df),
        "column_mapping": mapping,
        "velocity_mode": PANGAEA_CONFIG.velocity_mode,
        "metadata_preview": metadata_lines[:15],
        "analysis_ready": bool(mapping.get("time") and mapping.get("tau") and (mapping.get("velocity") or mapping.get("displacement"))),
        "notes": PANGAEA_CONFIG.notes,
    }
    return df, summary
