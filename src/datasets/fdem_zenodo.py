from __future__ import annotations

from ..config import FDEM_ZENODO_CONFIG
from ..io.load_fdem_zenodo import load_fdem_zenodo_dataset, locate_fdem_zenodo_files


def build_fdem_zenodo_summary() -> dict:
    try:
        df, summary = load_fdem_zenodo_dataset()
        return {
            "dataset": FDEM_ZENODO_CONFIG.to_summary(),
            **summary,
            "dataset_role": "simulated granular fault - LightGBM comparison dataset",
            "scientific_label": "simulated granular fault - LightGBM comparison dataset",
            "segmentation_method": "nss-reset segmentation",
            "state_definition": {
                "state_1": "mu (macroscopic friction-like target used in the published LightGBM workflow)",
                "state_2": "Ek (computed kinetic energy from sensor velocity fields)",
            },
            "variable_mapping_table": {
                "time": "time",
                "mu": "mu",
                "Ek": "Ek",
                "cycle_index": "nss",
            },
            "physical_interpretation_note": (
                "This is simulated FDEM data rather than a physical laboratory measurement. "
                "mu is taken from the published helper-script target column and Ek is computed from the velocity fields, "
                "so SINDy coefficient interpretation is approximate rather than directly comparable to laboratory stress measurements."
            ),
            "n_rows_loaded": int(len(df)) if df is not None else 0,
        }
    except Exception as exc:
        inventory = locate_fdem_zenodo_files()
        return {
            "dataset": FDEM_ZENODO_CONFIG.to_summary(),
            "analysis_ready": False,
            "resources": {key: [str(path) for path in value] for key, value in inventory.items()},
            "error": str(exc),
            "dataset_role": "simulation_stick_slip_candidate",
            "state_definition": {
                "state_1": "tau",
                "state_2": "V",
            },
        }
