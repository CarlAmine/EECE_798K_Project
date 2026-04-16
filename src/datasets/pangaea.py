from __future__ import annotations

from ..config import PANGAEA_CONFIG
from ..io.pangaea import load_pangaea_dataset


def build_pangaea_summary() -> dict:
    try:
        _, summary = load_pangaea_dataset()
        return {
            "dataset": PANGAEA_CONFIG.to_summary(),
            **summary,
            "state_definition": {
                "state_1": "tau (measured shear stress if available)",
                "state_2": "V (measured velocity or derived from displacement)",
            },
        }
    except Exception as exc:
        return {
            "dataset": PANGAEA_CONFIG.to_summary(),
            "analysis_ready": False,
            "raw_file": None,
            "error": str(exc),
            "state_definition": {
                "state_1": "tau",
                "state_2": "V",
            },
        }
