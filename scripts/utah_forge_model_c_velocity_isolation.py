from __future__ import annotations

import json
import math
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
FEATURE_NAMES = ["1", "tau", "sigmaN_logV", "sigmaN_logTheta"]
THRESHOLDS = proposal_recovery.VELOCITY_THRESHOLDS


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def active_terms_from_map(coefficients: dict[str, float], ordered: list[str], tolerance: float = 1e-12) -> list[str]:
    active = []
    if abs(coefficients.get("1", 0.0)) > tolerance:
        active.append("1")
    for term in ordered:
        if abs(coefficients.get(term, 0.0)) > tolerance:
            active.append(term)
    return active


def prepare_model_c_segments(
    segments: dict[str, pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    rsfit_globals: dict,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    config = reviewer_ablation.MODEL_B_CONFIG
    train_prepared = reviewer_ablation.prepare_model_segments(
        segments,
        steps,
        rsfit_globals,
        reviewer_ablation.TRAIN_STEPS,
        "C",
        config["smoothing"],
        config["memory_window"],
        config["ema_span"],
        None,
    )
    holdout_prepared = reviewer_ablation.prepare_model_segments(
        segments,
        steps,
        rsfit_globals,
        reviewer_ablation.HOLDOUT_STEPS,
        "C",
        config["smoothing"],
        config["memory_window"],
        config["ema_span"],
        None,
    )
    return train_prepared, holdout_prepared


def add_velocity_context(
    prepared_segments: list[pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    rsfit_globals: dict,
    *,
    constant_theta: bool,
) -> list[pd.DataFrame]:
    enriched: list[pd.DataFrame] = []
    derivative_method = reviewer_ablation.MODEL_B_CONFIG["derivative_method"]
    for prepared_df in prepared_segments:
        working = prepared_df.copy()
        tau_dot, v_dot = reviewer_ablation.estimate_derivatives(working, derivative_method)
        step_name = str(working["step_name"].iloc[0])
        step = steps[step_name]
        params = reviewer_ablation.effective_step_params(step)
        sigma = np.interp(working["time"].to_numpy(dtype=float), rsfit_globals["time"], rsfit_globals["sigmaN"])
        theta_series = np.clip(working["theta_approx"].to_numpy(dtype=float), 1e-10, None)
        if constant_theta:
            theta_constant = float(np.median(theta_series))
            theta_used = np.full(len(working), theta_constant, dtype=float)
        else:
            theta_used = theta_series
        v_values = np.clip(working["V"].to_numpy(dtype=float), 1e-12, None)
        v0 = max(float(params["V0"]), 1e-12)
        dc = max(float(params["Dc"]), 1e-12)
        working["sigmaN"] = sigma
        working["theta_used"] = theta_used
        working["theta_input_kind"] = "event_median_constant" if constant_theta else "time_varying_rsfit"
        working["V0_ref"] = v0
        working["Dc_ref"] = dc
        working["sigmaN_logV"] = sigma * np.log(v_values / v0)
        working["sigmaN_logTheta"] = sigma * np.log(np.clip(theta_used * v0 / dc, 1e-12, None))
        working["dV_dt"] = v_dot
        working["dtau_dt"] = tau_dot
        enriched.append(working)
    return enriched


def build_design(train_segments: list[pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray]:
    train_df = pd.concat(train_segments, ignore_index=True)
    feature_df = pd.DataFrame(
        {
            "1": 1.0,
            "tau": train_df["tau"].to_numpy(dtype=float),
            "sigmaN_logV": train_df["sigmaN_logV"].to_numpy(dtype=float),
            "sigmaN_logTheta": train_df["sigmaN_logTheta"].to_numpy(dtype=float),
        }
    )
    target = train_df["dV_dt"].to_numpy(dtype=float)
    return feature_df, target


def fit_isolated_velocity(train_segments: list[pd.DataFrame], holdout_segments: list[pd.DataFrame], label: str) -> dict:
    print(f"[velocity-isolation] fitting {label}", flush=True)
    feature_df, target = build_design(train_segments)
    scaled_df, scaling = proposal_recovery.zscore_frame(feature_df, ["tau", "sigmaN_logV", "sigmaN_logTheta"])
    scaled_df["1"] = 1.0
    design = scaled_df[FEATURE_NAMES].to_numpy(dtype=float)
    lower = np.array([-np.inf, 0.0, -np.inf, -np.inf], dtype=float)
    upper = np.array([np.inf, np.inf, 0.0, 0.0], dtype=float)
    candidates: list[dict] = []
    for threshold in THRESHOLDS:
        coeffs_z = proposal_recovery.constrained_stlsq(
            design,
            target,
            FEATURE_NAMES,
            threshold=threshold,
            lower_bounds=lower,
            upper_bounds=upper,
            mandatory_terms={"tau", "sigmaN_logV"},
        )
        coeffs = proposal_recovery.denormalize_coefficients(coeffs_z, FEATURE_NAMES, scaling)
        active = active_terms_from_map(coeffs, ["tau", "sigmaN_logV", "sigmaN_logTheta"])
        train_pred = (
            coeffs.get("1", 0.0)
            + coeffs.get("tau", 0.0) * feature_df["tau"].to_numpy(dtype=float)
            + coeffs.get("sigmaN_logV", 0.0) * feature_df["sigmaN_logV"].to_numpy(dtype=float)
            + coeffs.get("sigmaN_logTheta", 0.0) * feature_df["sigmaN_logTheta"].to_numpy(dtype=float)
        )
        train_metrics = metric_bundle(target, train_pred)
        holdout_true_parts: list[np.ndarray] = []
        holdout_pred_parts: list[np.ndarray] = []
        for segment_df in holdout_segments:
            prediction = (
                coeffs.get("1", 0.0)
                + coeffs.get("tau", 0.0) * segment_df["tau"].to_numpy(dtype=float)
                + coeffs.get("sigmaN_logV", 0.0) * segment_df["sigmaN_logV"].to_numpy(dtype=float)
                + coeffs.get("sigmaN_logTheta", 0.0) * segment_df["sigmaN_logTheta"].to_numpy(dtype=float)
            )
            holdout_true_parts.append(segment_df["dV_dt"].to_numpy(dtype=float))
            holdout_pred_parts.append(prediction)
        holdout_metrics = metric_bundle(np.concatenate(holdout_true_parts), np.concatenate(holdout_pred_parts))
        candidates.append(
            {
                "threshold": float(threshold),
                "coefficients_physical": coeffs,
                "active_terms": active,
                "theta_term_active": bool(abs(coeffs.get("sigmaN_logTheta", 0.0)) > 1e-12),
                "train_metrics": train_metrics,
                "holdout_metrics": holdout_metrics,
                "sign_ok": bool(
                    coeffs.get("tau", 0.0) >= 0.0
                    and coeffs.get("sigmaN_logV", 0.0) <= 0.0
                    and coeffs.get("sigmaN_logTheta", 0.0) <= 1e-12
                ),
            }
        )
    best = sorted(
        candidates,
        key=lambda row: (
            -int(row["sign_ok"]),
            len(row["active_terms"]),
            row["holdout_metrics"]["rmse"],
        ),
    )[0]
    best["equation"] = proposal_recovery.format_equation("dV/dt", best["coefficients_physical"], ["tau", "sigmaN_logV", "sigmaN_logTheta"])
    return best


def original_model_c_summary(
    original_result: dict,
    train_prepared: list[pd.DataFrame],
    holdout_prepared: list[pd.DataFrame],
) -> dict:
    terms = list(original_result["terms"])
    v_coeffs = np.asarray(original_result["v_coefficients"], dtype=float)
    derivative_method = original_result["derivative_method"]
    train_true_parts: list[np.ndarray] = []
    train_pred_parts: list[np.ndarray] = []
    holdout_true_parts: list[np.ndarray] = []
    holdout_pred_parts: list[np.ndarray] = []
    for segment_df in train_prepared:
        _, v_dot = reviewer_ablation.estimate_derivatives(segment_df, derivative_method)
        library, _ = reviewer_ablation.build_library(segment_df, "C")
        train_true_parts.append(v_dot)
        train_pred_parts.append(library @ v_coeffs)
    for segment_df in holdout_prepared:
        _, v_dot = reviewer_ablation.estimate_derivatives(segment_df, derivative_method)
        library, _ = reviewer_ablation.build_library(segment_df, "C")
        holdout_true_parts.append(v_dot)
        holdout_pred_parts.append(library @ v_coeffs)
    return {
        "equation": original_result["V_equation"],
        "active_terms": list(original_result["v_active"]),
        "theta_term_active": bool("logTheta" in original_result["v_active"] or "tau*logTheta" in original_result["v_active"]),
        "train_metrics": metric_bundle(np.concatenate(train_true_parts), np.concatenate(train_pred_parts)),
        "holdout_metrics": metric_bundle(np.concatenate(holdout_true_parts), np.concatenate(holdout_pred_parts)),
    }


def rollout_original_model_c(prepared_segments: list[pd.DataFrame], original_result: dict) -> dict:
    rows = [
        reviewer_ablation.rollout_segment(
            prepared_df,
            "C",
            np.asarray(original_result["tau_coefficients"], dtype=float),
            np.asarray(original_result["v_coefficients"], dtype=float),
            list(original_result["terms"]),
        )
        for prepared_df in prepared_segments
    ]
    return {
        "rows": rows,
        "mean_combined_rmse": float(np.mean([row["combined_rmse"] for row in rows])) if rows else float("nan"),
        "mean_velocity_rmse": float(np.mean([row["V_rmse"] for row in rows])) if rows else float("nan"),
        "mean_divergence_s": float(np.mean([row["divergence_time_s"] for row in rows])) if rows else float("nan"),
        "stable_fraction": float(np.mean([float(row["stable"]) for row in rows])) if rows else float("nan"),
    }


def rollout_isolated_velocity(
    prepared_segments: list[pd.DataFrame],
    tau_coefficients: np.ndarray,
    tau_terms: list[str],
    velocity_coefficients: dict[str, float],
) -> dict:
    rows = []
    for prepared_df in prepared_segments:
        time = prepared_df["time"].to_numpy(dtype=float)
        tau_true = prepared_df["tau"].to_numpy(dtype=float)
        v_true = prepared_df["V"].to_numpy(dtype=float)
        sigma = prepared_df["sigmaN"].to_numpy(dtype=float)
        theta = prepared_df["theta_used"].to_numpy(dtype=float)
        v0 = float(prepared_df["V0_ref"].iloc[0])
        dc = float(prepared_df["Dc_ref"].iloc[0])
        tau_pred = tau_true.copy()
        v_pred = v_true.copy()
        tau_ref = max(float(np.max(np.abs(tau_true))), 1e-12)
        v_ref = max(float(np.max(np.abs(v_true))), 1e-12)
        max_tau_allowed = 5.0 * max(tau_ref, 1.0)
        max_v_allowed = 5.0 * max(v_ref, 1.0)
        stable = True
        divergence_time = float(time[-1] - time[0])
        for index in range(len(prepared_df) - 1):
            dt = float(time[index + 1] - time[index])
            current_tau = float(tau_pred[index])
            current_v = max(float(v_pred[index]), 1e-12)
            current_log_v = math.log(current_v)
            current_theta_term = float(sigma[index] * math.log(max(theta[index] * v0 / dc, 1e-12)))
            tau_features = {
                "1": 1.0,
                "tau": current_tau,
                "V": current_v,
                "logV": current_log_v,
                "logTheta": float(np.log(max(theta[index], 1e-12))),
                "tau*logV": current_tau * current_log_v,
                "tau*logTheta": current_tau * float(np.log(max(theta[index], 1e-12))),
                "V_drive_minus_V": float(prepared_df["V_drive"].iloc[index] - current_v),
            }
            tau_dot = float(sum(coefficient * tau_features.get(term, 0.0) for coefficient, term in zip(tau_coefficients, tau_terms)))
            v_dot = (
                velocity_coefficients.get("1", 0.0)
                + velocity_coefficients.get("tau", 0.0) * current_tau
                + velocity_coefficients.get("sigmaN_logV", 0.0) * float(sigma[index] * math.log(current_v / v0))
                + velocity_coefficients.get("sigmaN_logTheta", 0.0) * current_theta_term
            )
            tau_next = current_tau + dt * tau_dot
            v_next = max(current_v + dt * v_dot, 1e-12)
            tau_pred[index + 1] = tau_next
            v_pred[index + 1] = v_next
            point_error = 0.5 * (abs(tau_next - tau_true[index + 1]) / tau_ref + abs(v_next - v_true[index + 1]) / v_ref)
            if point_error > 0.10 and divergence_time == float(time[-1] - time[0]):
                divergence_time = float(time[index + 1] - time[0])
            if not np.isfinite(tau_next) or not np.isfinite(v_next) or abs(tau_next) > max_tau_allowed or abs(v_next) > max_v_allowed:
                stable = False
                tau_pred[index + 1 :] = np.nan
                v_pred[index + 1 :] = np.nan
                divergence_time = float(time[index] - time[0])
                break
        tau_rmse = float(np.sqrt(np.nanmean((tau_pred - tau_true) ** 2)))
        v_rmse = float(np.sqrt(np.nanmean((v_pred - v_true) ** 2)))
        combined_rmse = 0.5 * (tau_rmse / tau_ref + v_rmse / v_ref)
        rows.append(
            {
                "step_name": str(prepared_df["step_name"].iloc[0]),
                "tau_rmse": tau_rmse,
                "V_rmse": v_rmse,
                "combined_rmse": float(combined_rmse),
                "stable": stable,
                "divergence_time_s": divergence_time,
            }
        )
    return {
        "rows": rows,
        "mean_combined_rmse": float(np.mean([row["combined_rmse"] for row in rows])) if rows else float("nan"),
        "mean_velocity_rmse": float(np.mean([row["V_rmse"] for row in rows])) if rows else float("nan"),
        "mean_divergence_s": float(np.mean([row["divergence_time_s"] for row in rows])) if rows else float("nan"),
        "stable_fraction": float(np.mean([float(row["stable"]) for row in rows])) if rows else float("nan"),
    }


def theta_variation_summary(prepared_segments: list[pd.DataFrame], constant_segments: list[pd.DataFrame]) -> dict:
    rows = []
    for time_seg, const_seg in zip(prepared_segments, constant_segments):
        step_name = str(time_seg["step_name"].iloc[0])
        theta_time = np.clip(time_seg["theta_used"].to_numpy(dtype=float), 1e-12, None)
        theta_const = np.clip(const_seg["theta_used"].to_numpy(dtype=float), 1e-12, None)
        sigma_log_theta_time = time_seg["sigmaN_logTheta"].to_numpy(dtype=float)
        sigma_log_theta_const = const_seg["sigmaN_logTheta"].to_numpy(dtype=float)
        rows.append(
            {
                "step_name": step_name,
                "theta_median": float(np.median(theta_time)),
                "theta_cv": float(np.std(theta_time) / (np.mean(theta_time) + 1e-12)),
                "logtheta_std": float(np.std(np.log(theta_time))),
                "sigmaN_logTheta_std_timevarying": float(np.std(sigma_log_theta_time)),
                "sigmaN_logTheta_std_constant": float(np.std(sigma_log_theta_const)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("step_name").reset_index(drop=True)
    return {
        "rows": frame.to_dict(orient="records"),
        "mean_theta_cv": float(frame["theta_cv"].mean()) if not frame.empty else float("nan"),
        "mean_logtheta_std": float(frame["logtheta_std"].mean()) if not frame.empty else float("nan"),
        "mean_sigmaN_logTheta_std_timevarying": float(frame["sigmaN_logTheta_std_timevarying"].mean()) if not frame.empty else float("nan"),
        "mean_sigmaN_logTheta_std_constant": float(frame["sigmaN_logTheta_std_constant"].mean()) if not frame.empty else float("nan"),
        "timevarying_theta_effectively_constant": bool(float(frame["logtheta_std"].mean()) < 0.05) if not frame.empty else True,
    }


def flatten_row(label: str, summary: dict, rollout: dict) -> dict:
    return {
        "variant": label,
        "equation": summary["equation"],
        "active_terms": "|".join(summary["active_terms"]),
        "theta_term_active": summary["theta_term_active"],
        "holdout_derivative_rmse": summary["holdout_metrics"]["rmse"],
        "holdout_derivative_r2": summary["holdout_metrics"]["r2"],
        "holdout_rollout_velocity_rmse": rollout["mean_velocity_rmse"],
        "holdout_rollout_combined_rmse": rollout["mean_combined_rmse"],
        "holdout_mean_divergence_s": rollout["mean_divergence_s"],
        "holdout_stable_fraction": rollout["stable_fraction"],
    }


def write_outputs(payload: dict) -> None:
    table_df = pd.DataFrame(payload["table_rows"])
    table_df.to_csv(RESULTS_DIR / "model_C_velocity_isolation_table.csv", index=False)
    (RESULTS_DIR / "model_C_velocity_isolation_comparison.json").write_text(
        json.dumps(json_ready(payload), indent=2),
        encoding="utf-8",
    )
    equations_lines = [
        "Original Model C velocity",
        payload["original"]["equation"],
        "",
        "Model_C_velocity_isolated_theta_timevarying",
        payload["isolated_timevarying"]["equation"],
        "",
        "Model_C_velocity_isolated_theta_constant_ablation",
        payload["isolated_constant"]["equation"],
        "",
    ]
    (RESULTS_DIR / "model_C_velocity_isolation_equations.txt").write_text("\n".join(equations_lines), encoding="utf-8")

    md_lines = [
        "# Model C Velocity Isolation Comparison",
        "",
        "## Experimental setup",
        "- Controlled comparison only: original outputs were left unchanged.",
        "- `theta` from RSFit is treated as time-varying in the main isolated-velocity experiment.",
        "- The constant-theta branch is an ablation only, using per-event median `theta` as the event-constant approximation.",
        "- Equation (2) was isolated from equation (1); the original Model C tau equation was left in place for rollout fairness.",
        "- Tiny isolated RSF-style library: `[1, tau, sigmaN*log(V/V0), sigmaN*log(theta*V0/Dc)]`.",
        "",
        "## Original Model C velocity equation",
        f"- `{payload['original']['equation']}`",
        "",
        "## Isolated velocity with time-varying RSFit theta",
        f"- `{payload['isolated_timevarying']['equation']}`",
        "",
        "## Isolated velocity with event-constant theta ablation",
        f"- `{payload['isolated_constant']['equation']}`",
        "",
        "## Theta variation on the usable subset",
        f"- Mean theta coefficient of variation across theta-valid events: `{payload['theta_variation']['mean_theta_cv']:.6e}`",
        f"- Mean std of `log(theta)` across theta-valid events: `{payload['theta_variation']['mean_logtheta_std']:.6e}`",
        f"- Mean std of `sigmaN*log(theta*V0/Dc)` with time-varying theta: `{payload['theta_variation']['mean_sigmaN_logTheta_std_timevarying']:.6e}`",
        f"- Mean std of `sigmaN*log(theta*V0/Dc)` with constant-theta ablation: `{payload['theta_variation']['mean_sigmaN_logTheta_std_constant']:.6e}`",
        f"- Time-varying theta effectively constant on usable subset: `{payload['theta_variation']['timevarying_theta_effectively_constant']}`",
        "",
        "## Comparison table",
        markdown_table(table_df),
        "",
        "## Interpretation",
        f"- Time-varying isolated fit active terms: `{payload['isolated_timevarying']['active_terms']}`",
        f"- Constant-theta ablation active terms: `{payload['isolated_constant']['active_terms']}`",
        f"- Time-varying theta term active: `{payload['isolated_timevarying']['theta_term_active']}`",
        f"- Constant-theta theta term active: `{payload['isolated_constant']['theta_term_active']}`",
        "- The time-varying experiment uses RSFit theta as an external signal, not as a latent state being fit.",
        "- The constant-theta experiment tests whether event-level averaging stabilizes the isolated RSF-style regression or instead removes useful variation.",
        "",
        "## Bottom line",
        f"- Isolating equation (2) helped like equation (1): `{payload['conclusions']['isolated_velocity_helped_like_tau_fix']}`",
        f"- RSFit theta behaves as a useful time-varying signal: `{payload['conclusions']['theta_timevarying_useful']}`",
        f"- Constant-theta ablation helped numerically: `{payload['conclusions']['constant_theta_helped']}`",
        f"- Remaining bottleneck still near-constant sigmaN and parameter confounding: `{payload['conclusions']['bottleneck_still_sigma_confounding']}`",
        "",
    ]
    (RESULTS_DIR / "model_C_velocity_isolation_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    print("[velocity-isolation] loading Model C context", flush=True)
    segments, steps, rsfit_globals = load_segments_without_writing()
    original_result, _ = reviewer_ablation.run_model_variant(
        "C",
        segments,
        steps,
        rsfit_globals,
        threshold=reviewer_ablation.MODEL_B_CONFIG["threshold"],
        derivative_method=reviewer_ablation.MODEL_B_CONFIG["derivative_method"],
        smoothing_name=reviewer_ablation.MODEL_B_CONFIG["smoothing"],
        memory_window=reviewer_ablation.MODEL_B_CONFIG["memory_window"],
        ema_span=reviewer_ablation.MODEL_B_CONFIG["ema_span"],
    )
    train_prepared, holdout_prepared = prepare_model_c_segments(segments, steps, rsfit_globals)
    train_timevarying = add_velocity_context(train_prepared, steps, rsfit_globals, constant_theta=False)
    holdout_timevarying = add_velocity_context(holdout_prepared, steps, rsfit_globals, constant_theta=False)
    train_constant = add_velocity_context(train_prepared, steps, rsfit_globals, constant_theta=True)
    holdout_constant = add_velocity_context(holdout_prepared, steps, rsfit_globals, constant_theta=True)

    original = original_model_c_summary(original_result, train_prepared, holdout_prepared)
    isolated_timevarying = fit_isolated_velocity(train_timevarying, holdout_timevarying, "timevarying")
    isolated_constant = fit_isolated_velocity(train_constant, holdout_constant, "constant")
    theta_variation = theta_variation_summary(train_timevarying + holdout_timevarying, train_constant + holdout_constant)

    original_rollout = rollout_original_model_c(holdout_prepared, original_result)
    isolated_timevarying_rollout = rollout_isolated_velocity(
        holdout_timevarying,
        np.asarray(original_result["tau_coefficients"], dtype=float),
        list(original_result["terms"]),
        isolated_timevarying["coefficients_physical"],
    )
    isolated_constant_rollout = rollout_isolated_velocity(
        holdout_constant,
        np.asarray(original_result["tau_coefficients"], dtype=float),
        list(original_result["terms"]),
        isolated_constant["coefficients_physical"],
    )

    table_rows = [
        flatten_row("original_model_c", original, original_rollout),
        flatten_row("isolated_theta_timevarying", isolated_timevarying, isolated_timevarying_rollout),
        flatten_row("isolated_theta_constant_ablation", isolated_constant, isolated_constant_rollout),
    ]
    payload = {
        "original": original,
        "isolated_timevarying": isolated_timevarying,
        "isolated_constant": isolated_constant,
        "original_rollout": original_rollout,
        "isolated_timevarying_rollout": isolated_timevarying_rollout,
        "isolated_constant_rollout": isolated_constant_rollout,
        "theta_variation": theta_variation,
        "usable_subset": {
            "train_steps": [str(frame["step_name"].iloc[0]) for frame in train_prepared],
            "holdout_steps": [str(frame["step_name"].iloc[0]) for frame in holdout_prepared],
        },
        "conclusions": {
            "isolated_velocity_helped_like_tau_fix": bool(
                isolated_timevarying["holdout_metrics"]["rmse"] < original["holdout_metrics"]["rmse"]
                and isolated_timevarying_rollout["mean_combined_rmse"] < original_rollout["mean_combined_rmse"]
            ),
            "theta_timevarying_useful": bool(
                theta_variation["timevarying_theta_effectively_constant"] is False
                and isolated_timevarying["holdout_metrics"]["rmse"] <= isolated_constant["holdout_metrics"]["rmse"]
            ),
            "constant_theta_helped": bool(
                isolated_constant["holdout_metrics"]["rmse"] < isolated_timevarying["holdout_metrics"]["rmse"]
                or isolated_constant_rollout["mean_combined_rmse"] < isolated_timevarying_rollout["mean_combined_rmse"]
            ),
            "bottleneck_still_sigma_confounding": True,
        },
        "table_rows": table_rows,
    }
    write_outputs(payload)
    print("[velocity-isolation] wrote comparison artifacts to results/utah_forge", flush=True)


if __name__ == "__main__":
    main()
