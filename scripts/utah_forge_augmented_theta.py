from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import UTAH_FORGE_CONFIG
from src.io.utah_forge import load_utah_forge_dataset
from src.preprocess.utah_forge import build_utah_forge_state
from src.utils.paths import ensure_directory


def inspect_rsfit_file(file_path: Path) -> dict:
    return {
        "file_present": file_path.exists(),
        "file_path": str(file_path),
        "inspection_status": "missing_local_file" if not file_path.exists() else "present_but_not_parsed",
        "Dc": None,
        "a": None,
        "b": None,
        "b1": None,
        "b2": None,
        "mu0": None,
        "theta0": None,
        "notes": (
            "The local workspace does not contain p5838_RSFit3000.mat, so Dc/a/b/mu0/theta0 could not be inspected here."
            if not file_path.exists()
            else "Local RSFit file exists but parser support was not needed in this blocked pass."
        ),
    }


def build_datatable_summary() -> tuple[pd.DataFrame, dict]:
    data_path = UTAH_FORGE_CONFIG.raw_dir / "p5838_datatable.mat"
    raw_df, load_summary = load_utah_forge_dataset(data_path)
    state_df, state_meta = build_utah_forge_state(raw_df, load_summary["column_mapping"])
    summary = {
        "raw_file": str(data_path),
        "time_column": load_summary["column_mapping"].get("time"),
        "tau_column": load_summary["column_mapping"].get("tau"),
        "velocity_column": load_summary["column_mapping"].get("velocity"),
        "friction_column": load_summary["column_mapping"].get("friction_coefficient"),
        "n_samples": int(len(state_df)),
        "time_start": float(state_df["time"].iloc[0]),
        "time_end": float(state_df["time"].iloc[-1]),
        "tau_mean": float(state_df["tau"].mean()),
        "tau_std": float(state_df["tau"].std()),
        "V_mean": float(state_df["V"].mean()),
        "V_std": float(state_df["V"].std()),
        "preserved_columns": state_meta.get("preserved_columns", []),
    }
    return state_df, summary


def main() -> None:
    results_dir = ensure_directory(UTAH_FORGE_CONFIG.results_dir)
    theta_plot_dir = ensure_directory(results_dir / "theta_reconstruction_plots")

    rsfit_path = UTAH_FORGE_CONFIG.raw_dir / "p5838_RSFit3000.mat"
    rsfit_summary = inspect_rsfit_file(rsfit_path)
    state_df, datatable_summary = build_datatable_summary()

    baseline_summary_path = results_dir / "best_model_summary.json"
    baseline_summary = (
        json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        if baseline_summary_path.exists()
        else None
    )

    reconstructed_theta_path = results_dir / "reconstructed_theta.csv"
    theta_placeholder = state_df[["time", "tau", "V"]].copy()
    if "mu" in state_df.columns:
        theta_placeholder["mu"] = state_df["mu"].to_numpy(dtype=float)
    theta_placeholder["theta"] = np.nan
    theta_placeholder["theta_source"] = "not_reconstructed"
    theta_placeholder["theta_reason"] = (
        "p5838_RSFit3000.mat is missing locally, so Dc/theta0 are unavailable."
    )
    theta_placeholder.to_csv(reconstructed_theta_path, index=False)
    (theta_plot_dir / "README.md").write_text(
        "# Theta Reconstruction Plots\n\n"
        "No theta reconstruction plots were generated because p5838_RSFit3000.mat is not present locally and Dc could not be recovered credibly.\n",
        encoding="utf-8",
    )

    augmented_summary = {
        "experiment_id": "p5838",
        "rsfit_summary": rsfit_summary,
        "datatable_summary": datatable_summary,
        "theta_reconstruction_attempted": False,
        "theta_reconstruction_credible": False,
        "theta_reconstruction_reason": (
            "Dc is not available locally because p5838_RSFit3000.mat is missing from data/utah_forge/."
        ),
        "reconstructed_theta_csv": str(reconstructed_theta_path),
        "theta_plots_dir": str(theta_plot_dir),
        "augmented_model_ran": False,
        "augmented_model_improved_over_baseline": False,
        "augmented_model_reason": (
            "The 3D augmented model was not fit because a credible theta(t) reconstruction requires Dc from the RSFit file."
        ),
        "best_augmented_equations": [],
        "baseline_reference": baseline_summary,
    }
    (results_dir / "augmented_model_summary.json").write_text(json.dumps(augmented_summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Utah FORGE augmented vs baseline report",
        "",
        "## RSFit inspection",
        f"- Local RSFit file present: `{rsfit_summary['file_present']}`",
        f"- RSFit file path checked: `{rsfit_summary['file_path']}`",
        f"- Inspection result: `{rsfit_summary['inspection_status']}`",
        f"- Dc available: `{rsfit_summary['Dc'] is not None}`",
        f"- a available: `{rsfit_summary['a'] is not None}`",
        f"- b or b1/b2 available: `{(rsfit_summary['b'] is not None) or (rsfit_summary['b1'] is not None) or (rsfit_summary['b2'] is not None)}`",
        f"- mu0 available: `{rsfit_summary['mu0'] is not None}`",
        f"- theta0 available: `{rsfit_summary['theta0'] is not None}`",
        "",
        "## Datatable inspection",
        f"- Raw file: `{datatable_summary['raw_file']}`",
        f"- time column: `{datatable_summary['time_column']}`",
        f"- tau column: `{datatable_summary['tau_column']}`",
        f"- V column: `{datatable_summary['velocity_column']}`",
        f"- mu column: `{datatable_summary['friction_column']}`",
        f"- Samples: `{datatable_summary['n_samples']}`",
        f"- Time span: `{datatable_summary['time_start']:.3f}` s to `{datatable_summary['time_end']:.3f}` s",
        "",
        "## Theta reconstruction",
        "- Credible theta reconstruction was not possible in this pass.",
        "- I did not invent Dc or a fitted initial state because that would make the rate-and-state augmentation scientifically weak.",
        "",
        "## Augmented model comparison",
        "- The 3D augmented model was not run, so there is no evidence yet that [tau, V, theta] improves over the current 2D baseline.",
        "- Current best 2D baseline remains the reference model in `results/utah_forge/best_model_summary.json`.",
    ]
    (results_dir / "augmented_vs_baseline_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    (results_dir / "discovered_equations_augmented.txt").write_text(
        "No augmented equations were fit because p5838_RSFit3000.mat is not present locally and Dc could not be recovered credibly.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
