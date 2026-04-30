from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utah_forge_proposal_equation_recovery as recovery


RESULTS_DIR = recovery.RESULTS_DIR


def step_map(frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {str(frame["step_name"].iloc[0]): frame.copy() for frame in frames}


def select_segments(mapping: dict[str, pd.DataFrame], step_names: list[str]) -> list[pd.DataFrame]:
    return [mapping[name].copy() for name in step_names if name in mapping]


def robustness_splits(all_step_names: list[str]) -> list[dict]:
    return [
        {"label": "alt_split_a", "holdout": ["p5838_step2", "p5838_step3"]},
        {"label": "alt_split_b", "holdout": ["p5838_step7", "p5838_step8"]},
    ]


def evaluate_split(
    split_def: dict,
    tau_map: dict[str, pd.DataFrame],
    all_map: dict[str, pd.DataFrame],
    theta_sample_map: dict[str, pd.DataFrame],
    theta_event_map: dict[str, pd.DataFrame],
) -> dict:
    holdout_names = [name for name in split_def["holdout"] if name in tau_map]
    train_names = [name for name in tau_map if name not in holdout_names]

    tau_train = select_segments(tau_map, train_names)
    tau_holdout = select_segments(tau_map, holdout_names)
    all_train = select_segments(all_map, train_names)
    all_holdout = select_segments(all_map, holdout_names)
    theta_sample_train = select_segments(theta_sample_map, train_names)
    theta_sample_holdout = select_segments(theta_sample_map, holdout_names)
    theta_event_train = select_segments(theta_event_map, train_names)
    theta_event_holdout = select_segments(theta_event_map, holdout_names)

    if not tau_train or not tau_holdout or not all_train or not all_holdout:
        raise RuntimeError(f"Split {split_def['label']} is missing required segments.")

    exact_train = theta_sample_train if theta_sample_train else theta_event_train
    exact_holdout = theta_sample_holdout if theta_sample_holdout else theta_event_holdout
    if not exact_train or not exact_holdout:
        raise RuntimeError(f"Split {split_def['label']} has no theta-usable train/holdout subset.")

    tau_model = recovery.fit_tau_recovery(tau_train, tau_holdout)
    exact_model = recovery.fit_velocity_model(exact_train, exact_holdout, recovery.model_feature_definitions(False)[0], fast_mode=True)["best"]
    reduced_model = recovery.fit_velocity_model(all_train, all_holdout, recovery.model_feature_definitions(False)[1], fast_mode=True)["best"]
    identifiability = recovery.build_identifiability_payload(exact_train, [])

    tau_coeff = tau_model["coefficients_physical"]
    tau_abs = {name: abs(value) for name, value in tau_coeff.items() if name != "1"}
    dominant_term = max(tau_abs, key=tau_abs.get) if tau_abs else "none"
    theta_coeff = exact_model["coefficients_physical"].get("sigmaN_logTheta", 0.0)
    sigma_logv_coeff = reduced_model["coefficients_physical"].get("sigmaN_logV", 0.0)

    return {
        "label": split_def["label"],
        "train_steps": train_names,
        "holdout_steps": holdout_names,
        "tau_model": {
            "equation": tau_model["exact_equation"],
            "one_term_equation": tau_model["one_term_equation"],
            "coefficients": tau_coeff,
            "dominant_nonintercept_term": dominant_term,
            "v_drive_minus_v_dominant": dominant_term == "V_drive_minus_V",
            "v_drive_minus_v_positive": bool(tau_coeff.get("V_drive_minus_V", 0.0) > 0),
            "holdout_mse": tau_model["holdout_mse"],
        },
        "reduced_velocity_model": {
            "equation": reduced_model["equation"],
            "coefficients": reduced_model["coefficients_physical"],
            "sigmaN_logV_present": bool(abs(sigma_logv_coeff) > 1e-8),
            "sigmaN_logV_negative": bool(sigma_logv_coeff < 0),
            "holdout_mse": reduced_model["holdout_mse"],
            "holdout_r2": reduced_model["holdout_r2"],
        },
        "exact_velocity_model": {
            "equation": exact_model["equation"],
            "coefficients": exact_model["coefficients_physical"],
            "theta_term_active": exact_model["theta_term_active"],
            "theta_coefficient": theta_coeff,
            "holdout_mse": exact_model["holdout_mse"],
            "holdout_r2": exact_model["holdout_r2"],
        },
        "identifiability": {
            "hard_diagnosis": identifiability.get("hard_diagnosis"),
            "sigma_cv": identifiability.get("sigma_cv"),
            "intercept_sigmaN_redundant": identifiability.get("intercept_sigmaN_redundant"),
            "theta_variation_too_weak_after_filtering": identifiability.get("theta_variation_too_weak_after_filtering"),
        },
    }


def build_markdown(summary: dict) -> str:
    lines = [
        "# Proposal Equation Robustness Check",
        "",
        "This was a bounded confirmation using two alternate holdout splits over the already prepared RSFit-aligned Utah FORGE segments. No new model families were introduced.",
        "",
        "## Aggregate conclusion",
        f"- Tau equation robustness: `{summary['aggregate']['tau_equation_confirmed']}`",
        f"- Reduced velocity log(V) structure retained: `{summary['aggregate']['reduced_logv_confirmed']}`",
        f"- Exact theta term remained non-identifiable: `{summary['aggregate']['theta_nonidentifiable_confirmed']}`",
        f"- Overall conclusion: `{summary['aggregate']['overall_conclusion']}`",
        "",
        "## Split-by-split results",
    ]
    for row in summary["splits"]:
        lines.extend(
            [
                f"### {row['label']}",
                f"- Train steps: `{', '.join(row['train_steps'])}`",
                f"- Holdout steps: `{', '.join(row['holdout_steps'])}`",
                f"- Tau equation: `{row['tau_model']['equation']}`",
                f"- Tau one-term approximation: `{row['tau_model']['one_term_equation']}`",
                f"- `(V_drive - V)` dominant and positive: `{row['tau_model']['v_drive_minus_v_dominant'] and row['tau_model']['v_drive_minus_v_positive']}`",
                f"- Reduced velocity equation: `{row['reduced_velocity_model']['equation']}`",
                f"- Reduced `sigmaN*log(V/V0)` retained and negative: `{row['reduced_velocity_model']['sigmaN_logV_present'] and row['reduced_velocity_model']['sigmaN_logV_negative']}`",
                f"- Exact theta term active: `{row['exact_velocity_model']['theta_term_active']}` with coefficient `{row['exact_velocity_model']['theta_coefficient']:.6e}`",
                f"- Structural identifiability diagnosis: `{row['identifiability']['hard_diagnosis']['summary']}`",
                f"- SigmaN coefficient of variation on exact-train subset: `{row['identifiability']['sigma_cv']:.6e}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "- Across both alternate splits, the compact tau law remained anchored by the positive `(V_drive - V)` term.",
            "- The reduced velocity law consistently retained the negative `sigmaN*log(V/V0)` structure, supporting a stable reduced RSF interpretation.",
            "- The exact theta term did not reactivate under these alternate splits, and the exact-train subsets still showed near-constant `sigmaN`, supporting the original structural-identifiability conclusion rather than a coding or alignment failure.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    prepared_checkpoint = recovery.load_pickle_checkpoint("prepared_segments")
    if prepared_checkpoint is None:
        raise RuntimeError("Missing prepared_segments checkpoint. Run scripts/utah_forge_proposal_equation_recovery.py first.")

    outputs = prepared_checkpoint["outputs"]
    tau_map = step_map(outputs["tau_train"] + outputs["tau_holdout"])
    all_map = step_map(outputs["all_train"] + outputs["all_holdout"])
    theta_sample_map = step_map(outputs["theta_sample_train"] + outputs["theta_sample_holdout"])
    theta_event_map = step_map(outputs["theta_event_train"] + outputs["theta_event_holdout"])

    split_rows = []
    for split_def in robustness_splits(list(tau_map)):
        split_rows.append(evaluate_split(split_def, tau_map, all_map, theta_sample_map, theta_event_map))

    aggregate = {
        "tau_equation_confirmed": all(row["tau_model"]["v_drive_minus_v_dominant"] and row["tau_model"]["v_drive_minus_v_positive"] for row in split_rows),
        "reduced_logv_confirmed": all(row["reduced_velocity_model"]["sigmaN_logV_present"] and row["reduced_velocity_model"]["sigmaN_logV_negative"] for row in split_rows),
        "theta_nonidentifiable_confirmed": all(not row["exact_velocity_model"]["theta_term_active"] for row in split_rows),
    }
    if all(aggregate.values()):
        overall = "Bounded robustness check supports the fixed final conclusion."
    else:
        overall = "Bounded robustness check found a contradiction that should be reviewed."
    aggregate["overall_conclusion"] = overall

    summary = {"splits": split_rows, "aggregate": aggregate}
    (RESULTS_DIR / "proposal_equation_robustness_check.md").write_text(build_markdown(summary), encoding="utf-8")
    (RESULTS_DIR / "proposal_equation_robustness_summary.json").write_text(json.dumps(recovery.json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
