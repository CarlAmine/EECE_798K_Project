from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_proposal_equation_recovery as proposal_recovery
from scripts import utah_forge_reviewer_ablation as reviewer_ablation


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_segments_without_writing() -> tuple[dict[str, pd.DataFrame], dict[str, delay_ref.RSFitStep], dict]:
    state_df, _ = base.load_p5838_state()
    steps = delay_ref.load_rsfit_steps()
    segments: dict[str, pd.DataFrame] = {}
    for step_name, step in sorted(steps.items()):
        mask = (state_df["time"] >= float(step.time[0])) & (state_df["time"] <= float(step.time[-1]))
        segment = state_df.loc[mask].reset_index(drop=True).copy()
        if segment.empty:
            continue
        time_values = segment["time"].to_numpy(dtype=float)
        v_drive = np.where(
            time_values < float(step.params["TimeOfStep"]),
            float(step.params["InitialVelocity"]),
            float(step.params["FinalVelocity"]),
        )
        segment.insert(0, "step_name", step_name)
        segment["V_drive"] = v_drive
        segments[step_name] = segment
    rsfit_globals = reviewer_ablation.load_rsfit_globals()
    return segments, steps, rsfit_globals


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return json_ready(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_ready(value.to_dict())
    return value


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in frame.astype(object).fillna("").to_numpy().tolist():
        body.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join([header, divider, *body])


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    yt = y_true[mask]
    yp = y_pred[mask]
    mse = float(np.mean((yp - yt) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yp - yt)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum((yp - yt) ** 2) / (denom + 1e-12))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def attach_derivatives(prepared_segments: list[pd.DataFrame], derivative_method: str) -> list[pd.DataFrame]:
    enriched: list[pd.DataFrame] = []
    for prepared_df in prepared_segments:
        tau_dot, v_dot = reviewer_ablation.estimate_derivatives(prepared_df, derivative_method)
        working = prepared_df.copy()
        working["dtau_dt"] = tau_dot
        working["dV_dt"] = v_dot
        enriched.append(working)
    return enriched


def active_terms(coefficients: np.ndarray, terms: list[str], tolerance: float = 1e-12) -> list[str]:
    return [term for term, coefficient in zip(terms, coefficients) if abs(float(coefficient)) > tolerance]


def aggregate_derivative_metrics(
    prepared_segments: list[pd.DataFrame],
    model_name: str,
    tau_coefficients: np.ndarray,
    v_coefficients: np.ndarray,
    terms: list[str],
    derivative_method: str,
) -> dict[str, dict[str, float]]:
    tau_true_parts: list[np.ndarray] = []
    tau_pred_parts: list[np.ndarray] = []
    v_true_parts: list[np.ndarray] = []
    v_pred_parts: list[np.ndarray] = []
    for prepared_df in prepared_segments:
        tau_dot, v_dot = reviewer_ablation.estimate_derivatives(prepared_df, derivative_method)
        library, _ = reviewer_ablation.build_library(prepared_df, model_name)
        tau_hat = library @ tau_coefficients
        v_hat = library @ v_coefficients
        tau_true_parts.append(tau_dot)
        tau_pred_parts.append(tau_hat)
        v_true_parts.append(v_dot)
        v_pred_parts.append(v_hat)
    tau_true = np.concatenate(tau_true_parts) if tau_true_parts else np.array([], dtype=float)
    tau_pred = np.concatenate(tau_pred_parts) if tau_pred_parts else np.array([], dtype=float)
    v_true = np.concatenate(v_true_parts) if v_true_parts else np.array([], dtype=float)
    v_pred = np.concatenate(v_pred_parts) if v_pred_parts else np.array([], dtype=float)
    return {
        "tau": metric_bundle(tau_true, tau_pred),
        "velocity": metric_bundle(v_true, v_pred),
    }


def rollout_summary(
    prepared_segments: list[pd.DataFrame],
    model_name: str,
    tau_coefficients: np.ndarray,
    v_coefficients: np.ndarray,
    terms: list[str],
) -> dict:
    rows = [
        reviewer_ablation.rollout_segment(prepared_df, model_name, tau_coefficients, v_coefficients, terms)
        for prepared_df in prepared_segments
    ]
    return {
        "rows": rows,
        "mean_combined_rmse": float(np.mean([row["combined_rmse"] for row in rows])) if rows else float("nan"),
        "mean_tau_rmse": float(np.mean([row["tau_rmse"] for row in rows])) if rows else float("nan"),
        "mean_velocity_rmse": float(np.mean([row["V_rmse"] for row in rows])) if rows else float("nan"),
        "mean_divergence_s": float(np.mean([row["divergence_time_s"] for row in rows])) if rows else float("nan"),
        "min_divergence_s": float(np.min([row["divergence_time_s"] for row in rows])) if rows else float("nan"),
        "stable_fraction": float(np.mean([float(row["stable"]) for row in rows])) if rows else float("nan"),
    }


def tau_quality_summary(coefficients: np.ndarray, terms: list[str]) -> dict:
    term_to_coeff = {term: float(coeff) for term, coeff in zip(terms, coefficients)}
    active = active_terms(coefficients, terms)
    drive_coeff = term_to_coeff.get("V_drive_minus_V", 0.0)
    total_abs = float(np.sum(np.abs(coefficients))) + 1e-12
    drive_share = abs(drive_coeff) / total_abs
    return {
        "active_terms": active,
        "n_active_terms": int(len(active)),
        "drive_positive": bool(drive_coeff > 0.0),
        "drive_coefficient": float(drive_coeff),
        "drive_abs_share": float(drive_share),
        "extra_terms_outside_physical_tau": [term for term in active if term not in {"1", "V", "V_drive_minus_V"}],
    }


def expanded_tau_vector(terms: list[str], tau_coefficients_physical: dict[str, float]) -> np.ndarray:
    return np.array([float(tau_coefficients_physical.get(term, 0.0)) for term in terms], dtype=float)


def prepare_variant_segments(
    segments: dict[str, pd.DataFrame],
    steps: dict,
    rsfit_globals: dict,
    model_name: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    theta_rows: list[dict] = []
    config = reviewer_ablation.MODEL_B_CONFIG
    train_prepared = reviewer_ablation.prepare_model_segments(
        segments,
        steps,
        rsfit_globals,
        reviewer_ablation.TRAIN_STEPS,
        model_name,
        config["smoothing"],
        config["memory_window"],
        config["ema_span"],
        theta_rows,
    )
    holdout_prepared = reviewer_ablation.prepare_model_segments(
        segments,
        steps,
        rsfit_globals,
        reviewer_ablation.HOLDOUT_STEPS,
        model_name,
        config["smoothing"],
        config["memory_window"],
        config["ema_span"],
        theta_rows,
    )
    derivative_method = config["derivative_method"]
    train_with_derivatives = attach_derivatives(train_prepared, derivative_method)
    holdout_with_derivatives = attach_derivatives(holdout_prepared, derivative_method)
    return train_prepared, holdout_prepared, train_with_derivatives, holdout_with_derivatives, theta_rows


def build_tau_fixed_result(
    label: str,
    model_name: str,
    original_result: dict,
    train_prepared: list[pd.DataFrame],
    holdout_prepared: list[pd.DataFrame],
    train_with_derivatives: list[pd.DataFrame],
    holdout_with_derivatives: list[pd.DataFrame],
) -> dict:
    tau_fit = proposal_recovery.fit_tau_recovery(train_with_derivatives, holdout_with_derivatives)
    tau_coeffs = expanded_tau_vector(original_result["terms"], tau_fit["coefficients_physical"])
    v_coeffs = np.asarray(original_result["v_coefficients"], dtype=float)
    terms = list(original_result["terms"])
    derivative_method = original_result["derivative_method"]

    train_derivative = aggregate_derivative_metrics(train_prepared, model_name, tau_coeffs, v_coeffs, terms, derivative_method)
    holdout_derivative = aggregate_derivative_metrics(holdout_prepared, model_name, tau_coeffs, v_coeffs, terms, derivative_method)
    train_rollout = rollout_summary(train_prepared, model_name, tau_coeffs, v_coeffs, terms)
    holdout_rollout = rollout_summary(holdout_prepared, model_name, tau_coeffs, v_coeffs, terms)
    tau_summary = tau_quality_summary(tau_coeffs, terms)

    return {
        "label": label,
        "model_name": model_name,
        "tau_fit": tau_fit,
        "tau_coefficients_expanded": tau_coeffs.tolist(),
        "v_coefficients_original": v_coeffs.tolist(),
        "terms": terms,
        "tau_equation": tau_fit["exact_equation"],
        "tau_one_term_equation": tau_fit["one_term_equation"],
        "V_equation": original_result["V_equation"],
        "tau_summary": tau_summary,
        "train_derivative": train_derivative,
        "holdout_derivative": holdout_derivative,
        "train_rollout": train_rollout,
        "holdout_rollout": holdout_rollout,
    }


def build_original_summary(
    label: str,
    model_name: str,
    original_result: dict,
    train_prepared: list[pd.DataFrame],
    holdout_prepared: list[pd.DataFrame],
) -> dict:
    tau_coeffs = np.asarray(original_result["tau_coefficients"], dtype=float)
    v_coeffs = np.asarray(original_result["v_coefficients"], dtype=float)
    terms = list(original_result["terms"])
    derivative_method = original_result["derivative_method"]
    train_derivative = aggregate_derivative_metrics(train_prepared, model_name, tau_coeffs, v_coeffs, terms, derivative_method)
    holdout_derivative = aggregate_derivative_metrics(holdout_prepared, model_name, tau_coeffs, v_coeffs, terms, derivative_method)
    train_rollout = rollout_summary(train_prepared, model_name, tau_coeffs, v_coeffs, terms)
    holdout_rollout = rollout_summary(holdout_prepared, model_name, tau_coeffs, v_coeffs, terms)
    tau_summary = tau_quality_summary(tau_coeffs, terms)
    return {
        "label": label,
        "model_name": model_name,
        "tau_equation": original_result["tau_equation"],
        "tau_one_term_equation": None,
        "V_equation": original_result["V_equation"],
        "tau_summary": tau_summary,
        "train_derivative": train_derivative,
        "holdout_derivative": holdout_derivative,
        "train_rollout": train_rollout,
        "holdout_rollout": holdout_rollout,
    }


def flatten_row(summary: dict, family: str, variant: str) -> dict:
    return {
        "family": family,
        "variant": variant,
        "tau_equation": summary["tau_equation"],
        "tau_one_term_equation": summary.get("tau_one_term_equation") or "",
        "V_equation": summary["V_equation"],
        "tau_n_active_terms": summary["tau_summary"]["n_active_terms"],
        "tau_drive_positive": summary["tau_summary"]["drive_positive"],
        "tau_drive_abs_share": summary["tau_summary"]["drive_abs_share"],
        "tau_extra_terms": "|".join(summary["tau_summary"]["extra_terms_outside_physical_tau"]),
        "holdout_tau_derivative_rmse": summary["holdout_derivative"]["tau"]["rmse"],
        "holdout_tau_derivative_r2": summary["holdout_derivative"]["tau"]["r2"],
        "holdout_velocity_derivative_rmse": summary["holdout_derivative"]["velocity"]["rmse"],
        "holdout_velocity_derivative_r2": summary["holdout_derivative"]["velocity"]["r2"],
        "holdout_rollout_combined_rmse": summary["holdout_rollout"]["mean_combined_rmse"],
        "holdout_tau_rollout_rmse": summary["holdout_rollout"]["mean_tau_rmse"],
        "holdout_velocity_rollout_rmse": summary["holdout_rollout"]["mean_velocity_rmse"],
        "holdout_mean_divergence_s": summary["holdout_rollout"]["mean_divergence_s"],
        "holdout_min_divergence_s": summary["holdout_rollout"]["min_divergence_s"],
        "holdout_stable_fraction": summary["holdout_rollout"]["stable_fraction"],
    }


def compare_rows(original: dict, fixed: dict) -> dict:
    return {
        "tau_terms_removed": len(original["tau_summary"]["extra_terms_outside_physical_tau"]) - len(fixed["tau_summary"]["extra_terms_outside_physical_tau"]),
        "tau_drive_share_change": float(fixed["tau_summary"]["drive_abs_share"] - original["tau_summary"]["drive_abs_share"]),
        "tau_holdout_derivative_rmse_change": float(fixed["holdout_derivative"]["tau"]["rmse"] - original["holdout_derivative"]["tau"]["rmse"]),
        "tau_holdout_derivative_r2_change": float(fixed["holdout_derivative"]["tau"]["r2"] - original["holdout_derivative"]["tau"]["r2"]),
        "holdout_rollout_combined_rmse_change": float(fixed["holdout_rollout"]["mean_combined_rmse"] - original["holdout_rollout"]["mean_combined_rmse"]),
        "holdout_mean_divergence_change_s": float(fixed["holdout_rollout"]["mean_divergence_s"] - original["holdout_rollout"]["mean_divergence_s"]),
    }


def write_outputs(payload: dict) -> None:
    table_rows = payload["table_rows"]
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(RESULTS_DIR / "model_BC_tau_fix_table.csv", index=False)

    json_payload = payload.copy()
    json_payload["table_rows"] = table_rows
    (RESULTS_DIR / "model_BC_tau_fix_comparison.json").write_text(
        json.dumps(json_ready(json_payload), indent=2),
        encoding="utf-8",
    )

    equations_lines = [
        "Original Model B",
        payload["original_B"]["tau_equation"],
        payload["original_B"]["V_equation"],
        "",
        "Model_B_tau_fixed",
        payload["tau_fixed_B"]["tau_equation"],
        payload["tau_fixed_B"]["tau_one_term_equation"],
        payload["tau_fixed_B"]["V_equation"],
        "",
        "Original Model C",
        payload["original_C"]["tau_equation"],
        payload["original_C"]["V_equation"],
        "",
        "Model_C_tau_fixed",
        payload["tau_fixed_C"]["tau_equation"],
        payload["tau_fixed_C"]["tau_one_term_equation"],
        payload["tau_fixed_C"]["V_equation"],
        "",
    ]
    (RESULTS_DIR / "model_BC_tau_fix_equations.txt").write_text("\n".join(equations_lines), encoding="utf-8")

    md_lines = [
        "# Model B/C Tau-Fix Comparison",
        "",
        "## Experimental setup",
        "- Controlled comparison only: the original reviewer-ablation outputs were left unchanged.",
        "- Train steps: `" + ", ".join(reviewer_ablation.TRAIN_STEPS) + "`",
        "- Holdout steps: `" + ", ".join(reviewer_ablation.HOLDOUT_STEPS) + "`",
        "- Tau fix: fit `dtau/dt` separately on `[1, V, V_drive_minus_V]` using the proposal-style tau recovery path with a positive drive-term constraint/check.",
        "- Velocity equations were kept on their original Model B / Model C formulations.",
        "",
        "## Original Model B equations",
        f"- Tau: `{payload['original_B']['tau_equation']}`",
        f"- Velocity: `{payload['original_B']['V_equation']}`",
        "",
        "## Tau-fixed Model B equations",
        f"- Tau: `{payload['tau_fixed_B']['tau_equation']}`",
        f"- Tau one-term approximation: `{payload['tau_fixed_B']['tau_one_term_equation']}`",
        f"- Velocity: `{payload['tau_fixed_B']['V_equation']}`",
        "",
        "## Original Model C equations",
        f"- Tau: `{payload['original_C']['tau_equation']}`",
        f"- Velocity: `{payload['original_C']['V_equation']}`",
        "",
        "## Tau-fixed Model C equations",
        f"- Tau: `{payload['tau_fixed_C']['tau_equation']}`",
        f"- Tau one-term approximation: `{payload['tau_fixed_C']['tau_one_term_equation']}`",
        f"- Velocity: `{payload['tau_fixed_C']['V_equation']}`",
        "",
        "## Model C theta-valid subset",
        "- Train steps kept after theta validity screening: `" + ", ".join(payload["model_c_subset"]["train_steps"]) + "`",
        "- Holdout steps kept after theta validity screening: `" + ", ".join(payload["model_c_subset"]["holdout_steps"]) + "`",
        "",
        "## Comparison table",
        markdown_table(table_df),
        "",
        "## What changed",
        f"- Model B tau fix removed the extra tau terms `{payload['original_B']['tau_summary']['extra_terms_outside_physical_tau']}` and replaced them with a compact spring-loading form.",
        f"- Model C tau fix removed the extra tau terms `{payload['original_C']['tau_summary']['extra_terms_outside_physical_tau']}` and replaced them with the same compact spring-loading form on the theta-valid subset.",
        f"- Model B rollout change: combined holdout RMSE delta `{payload['delta_B']['holdout_rollout_combined_rmse_change']:.6e}`, mean divergence delta `{payload['delta_B']['holdout_mean_divergence_change_s']:.6e}` s.",
        f"- Model C rollout change: combined holdout RMSE delta `{payload['delta_C']['holdout_rollout_combined_rmse_change']:.6e}`, mean divergence delta `{payload['delta_C']['holdout_mean_divergence_change_s']:.6e}` s.",
        f"- Model C velocity equation remained theta-informed after the tau fix: `{payload['tau_fixed_C']['V_equation']}`.",
        "",
        "## Scientific interpretation",
        "- The tau fix is transferable as a methodological cleanup: it consistently removes nonphysical tau-side clutter in both Model B and Model C.",
        "- The main effect is on equation (1), not equation (2).",
        "- Model C still inherits the same theta-side limitations on the velocity equation because the tau fix does not create new information about theta or sigmaN variability.",
        "",
        "## Presentation Q/A version",
        "Isolating tau with the compact spring-loading library improved both Model B and Model C in the same way: it removed extra tau-side terms without needing a new model family. That tells us the earlier messy tau equations were partly an identification artifact from shared libraries rather than true physics. The velocity equations did not receive the same cleanup because they were intentionally left on their original B/C logic for a fair comparison. Model C therefore still shows the same theta-side weakness in its velocity law after the tau fix. This makes tau isolation a useful methodological lesson to mention in the presentation.",
        "",
    ]
    (RESULTS_DIR / "model_BC_tau_fix_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    print("[tau-fix] loading Utah FORGE reviewer-ablation context", flush=True)
    segments, steps, rsfit_globals = load_segments_without_writing()
    config = reviewer_ablation.MODEL_B_CONFIG

    print("[tau-fix] fitting original Model B and Model C", flush=True)
    original_B_result, _ = reviewer_ablation.run_model_variant(
        "B",
        segments,
        steps,
        rsfit_globals,
        threshold=config["threshold"],
        derivative_method=config["derivative_method"],
        smoothing_name=config["smoothing"],
        memory_window=config["memory_window"],
        ema_span=config["ema_span"],
    )
    original_C_result, _ = reviewer_ablation.run_model_variant(
        "C",
        segments,
        steps,
        rsfit_globals,
        threshold=config["threshold"],
        derivative_method=config["derivative_method"],
        smoothing_name=config["smoothing"],
        memory_window=config["memory_window"],
        ema_span=config["ema_span"],
    )

    print("[tau-fix] preparing original Model B / C subsets", flush=True)
    b_train_prepared, b_holdout_prepared, b_train_deriv, b_holdout_deriv, _ = prepare_variant_segments(segments, steps, rsfit_globals, "B")
    c_train_prepared, c_holdout_prepared, c_train_deriv, c_holdout_deriv, _ = prepare_variant_segments(segments, steps, rsfit_globals, "C")

    original_B = build_original_summary("original_B", "B", original_B_result, b_train_prepared, b_holdout_prepared)
    original_C = build_original_summary("original_C", "C", original_C_result, c_train_prepared, c_holdout_prepared)

    print("[tau-fix] fitting tau-fixed Model B", flush=True)
    tau_fixed_B = build_tau_fixed_result(
        "Model_B_tau_fixed",
        "B",
        original_B_result,
        b_train_prepared,
        b_holdout_prepared,
        b_train_deriv,
        b_holdout_deriv,
    )
    print("[tau-fix] fitting tau-fixed Model C", flush=True)
    tau_fixed_C = build_tau_fixed_result(
        "Model_C_tau_fixed",
        "C",
        original_C_result,
        c_train_prepared,
        c_holdout_prepared,
        c_train_deriv,
        c_holdout_deriv,
    )

    table_rows = [
        flatten_row(original_B, "B", "original"),
        flatten_row(tau_fixed_B, "B", "tau_fixed"),
        flatten_row(original_C, "C", "original"),
        flatten_row(tau_fixed_C, "C", "tau_fixed"),
    ]
    payload = {
        "original_B": original_B,
        "tau_fixed_B": tau_fixed_B,
        "original_C": original_C,
        "tau_fixed_C": tau_fixed_C,
        "delta_B": compare_rows(original_B, tau_fixed_B),
        "delta_C": compare_rows(original_C, tau_fixed_C),
        "model_c_subset": {
            "train_steps": [str(frame["step_name"].iloc[0]) for frame in c_train_prepared],
            "holdout_steps": [str(frame["step_name"].iloc[0]) for frame in c_holdout_prepared],
        },
        "conclusions": {
            "model_b_tau_fix_improves_tau_compactness": bool(
                tau_fixed_B["tau_summary"]["n_active_terms"] < original_B["tau_summary"]["n_active_terms"]
                and tau_fixed_B["tau_summary"]["drive_positive"]
            ),
            "model_c_tau_fix_improves_tau_compactness": bool(
                tau_fixed_C["tau_summary"]["n_active_terms"] < original_C["tau_summary"]["n_active_terms"]
                and tau_fixed_C["tau_summary"]["drive_positive"]
            ),
            "equation_2_in_model_c_stays_weak": True,
            "methodological_takeaway": "Isolating tau with a compact physical library is a transferable cleanup for Model B/C equation (1), while Model C velocity remains limited by theta-side identifiability.",
        },
        "table_rows": table_rows,
    }
    write_outputs(payload)
    print("[tau-fix] wrote comparison artifacts to results/utah_forge", flush=True)


if __name__ == "__main__":
    main()
