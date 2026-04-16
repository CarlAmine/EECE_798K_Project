from __future__ import annotations

from ..config import UTAH_FORGE_CONFIG
from ..io.utah_forge import load_utah_forge_dataset, locate_utah_forge_files


def build_utah_forge_summary() -> dict:
    try:
        _, summary = load_utah_forge_dataset()
        return {
            "dataset": UTAH_FORGE_CONFIG.to_summary(),
            **summary,
            "state_definition": {
                "state_1": "tau (measured shear stress if available)",
                "state_2": "V (measured velocity or derived from displacement)",
            },
        }
    except Exception as exc:
        return {
            "dataset": UTAH_FORGE_CONFIG.to_summary(),
            "analysis_ready": False,
            "resources": {key: [str(path) for path in value] for key, value in locate_utah_forge_files().items()},
            "error": str(exc),
            "state_definition": {
                "state_1": "tau",
                "state_2": "V",
            },
        }
