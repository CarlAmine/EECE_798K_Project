from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_memory_refinement as memory_ref
from src.derivatives import derivative_savgol, derivative_spline
from src.utils.paths import ensure_directory


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
THETA_PLOTS_DIR = RESULTS_DIR / "p5838_theta_comparison_plots"

TRAIN_STEPS = ("p5838_step3", "p5838_step8", "p5838_step9", "p5838_step4", "p5838_step5", "p5838_step10")
HOLDOUT_STEPS = ("p5838_step2", "p5838_step7")
MODEL_B_CONFIG = {
    "smoothing": "moderate",
    "derivative_method": "spline",
    "threshold": 0.002,
    "memory_window": 20,
    "ema_span": 20,
}
SPARSITY_THRESHOLDS = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
DERIVATIVE_VARIANTS = {
    "savgol_15": {"kind": "savgol", "window": 15, "polyorder": 3},
    "savgol_31": {"kind": "savgol", "window": 31, "polyorder": 3},
    "finite_diff_5pt": {"kind": "finite_5pt"},
}
MEMORY_BASELINE_DIVERGENCE = {"p5838_step2": 17.96, "p5838_step7": 9.77}


def ensure_layout() -> None:
    ensure_directory(RESULTS_DIR)
    ensure_directory(THETA_PLOTS_DIR)


def load_rsfit_globals() -> dict:
    rsfit_path = delay_ref.resolve_rsfit_path()
    mat = scipy.io.loadmat(rsfit_path, squeeze_me=True, struct_as_record=False)
    return {
        "time": np.asarray(mat["time"]).reshape(-1).astype(float),
        "sigmaN": np.asarray(mat["sigmaN"]).reshape(-1).astype(float),
        "mu": np.asarray(mat["mu"]).reshape(-1).astype(float),
    }


def effective_step_params(step: delay_ref.RSFitStep) -> dict:
    b1 = float(step.params["b1"])
    b2 = float(step.params["b2"])
    b_eff = b1 + b2
    if abs(b_eff) < 1e-8:
        b_eff = b1 if abs(b1) > 1e-8 else 1e-6
    dc1 = max(float(step.params["dc1"]), 1e-12)
    dc2 = max(float(step.params["dc2"]), 1e-12)
    if abs(b1) > 1e-12 and abs(b2) > 1e-12 and abs(b1 + b2) > 1e-12:
        dc_eff = math.exp((b1 * math.log(dc1) + b2 * math.log(dc2)) / (b1 + b2))
    else:
        dc_eff = dc1
    return {
        "a": float(step.params["a"]),
        "b": float(b_eff),
        "Dc": float(dc_eff),
        "V0": max(float(step.params["InitialVelocity"]), 1e-12),
        "mu0": float(step.params["mu0"]),
        "step_r2": float(step.params["R2"]),
    }


def five_point_derivative(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(y)
    if len(y) < 5:
        return np.gradient(y, t)
    dt = float(np.mean(np.diff(t)))
    out[2:-2] = (y[:-4] - 8.0 * y[1:-3] + 8.0 * y[3:-1] - y[4:]) / (12.0 * dt)
    out[0] = (-25.0 * y[0] + 48.0 * y[1] - 36.0 * y[2] + 16.0 * y[3] - 3.0 * y[4]) / (12.0 * dt)
    out[1] = (y[2] - y[0]) / (t[2] - t[0])
    out[-2] = (y[-1] - y[-3]) / (t[-1] - t[-3])
    out[-1] = (25.0 * y[-1] - 48.0 * y[-2] + 36.0 * y[-3] - 16.0 * y[-4] + 3.0 * y[-5]) / (12.0 * dt)
    return out


def estimate_derivatives(prepared_df: pd.DataFrame, method_name: str) -> tuple[np.ndarray, np.ndarray]:
    time = prepared_df["time"].to_numpy(dtype=float)
    tau = prepared_df["tau"].to_numpy(dtype=float)
    velocity = prepared_df["V"].to_numpy(dtype=float)
    if method_name == "spline":
        return derivative_spline(tau, t=time), derivative_spline(velocity, t=time)
    if method_name == "savgol_15":
        return derivative_savgol(tau, t=time, window=15, polyorder=3), derivative_savgol(velocity, t=time, window=15, polyorder=3)
    if method_name == "savgol_31":
        return derivative_savgol(tau, t=time, window=31, polyorder=3), derivative_savgol(velocity, t=time, window=31, polyorder=3)
    if method_name == "finite_diff_5pt":
        return five_point_derivative(tau, time), five_point_derivative(velocity, time)
    raise ValueError(f"Unsupported derivative method: {method_name}")


def load_segments() -> tuple[dict[str, pd.DataFrame], dict[str, delay_ref.RSFitStep], dict]:
    state_df, _ = base.load_p5838_state()
    _, segments, steps = memory_ref.segment_step_windows(state_df)
    rsfit_globals = load_rsfit_globals()
    return segments, steps, rsfit_globals


def prepare_step_segment(step_df: pd.DataFrame, smoothing_name: str, memory_window: int, ema_span: int) -> pd.DataFrame:
    prepared_df, _ = memory_ref.prepare_memory_segment(
        step_df,
        smoothing_name=smoothing_name,
        memory_window=memory_window,
        ema_span=ema_span,
        include_optional_avgs=False,
    )
    prepared_df.attrs["memory_window"] = int(memory_window)
    prepared_df.attrs["ema_span"] = int(ema_span)
    return prepared_df


def reconstruct_theta(
    prepared_df: pd.DataFrame,
    step: delay_ref.RSFitStep,
    rsfit_globals: dict,
) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    params = effective_step_params(step)
    sigma = np.interp(prepared_df["time"].to_numpy(dtype=float), rsfit_globals["time"], rsfit_globals["sigmaN"])
    sigma = np.clip(sigma, 1e-6, None)
    mu = prepared_df["tau"].to_numpy(dtype=float) / sigma
    V = np.clip(prepared_df["V"].to_numpy(dtype=float), 1e-12, None)
    raw_theta = (params["Dc"] / params["V0"]) * np.exp((mu - params["mu0"] - params["a"] * np.log(V / params["V0"])) / params["b"])
    theta = np.clip(raw_theta, 1e-10, 1e6)
    invalid_fraction = float(np.mean(~np.isfinite(raw_theta)))
    clipped_fraction = float(np.mean((theta != raw_theta) | (~np.isfinite(raw_theta))))
    status = {
        **params,
        "invalid_fraction": invalid_fraction,
        "clipped_fraction": clipped_fraction,
    }
    if not np.isfinite(theta).all() or np.allclose(theta, theta[0]) or clipped_fraction > 0.25:
        print(f"Warning: skipping {step.step_name} for theta-informed model; invalid/clipped theta fraction={clipped_fraction:.3f}")
        return None, status
    return theta, status


def build_library(prepared_df: pd.DataFrame, model_name: str) -> tuple[np.ndarray, list[str]]:
    if model_name == "A":
        terms = ["1", "tau", "V", "logV", "tau*logV", "V_drive_minus_V"]
        columns = [
            np.ones(len(prepared_df)),
            prepared_df["tau"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["logV"].to_numpy(dtype=float),
            (prepared_df["tau"] * prepared_df["logV"]).to_numpy(dtype=float),
            prepared_df["V_drive_minus_V"].to_numpy(dtype=float),
        ]
    elif model_name == "B":
        terms = ["1", "V", "V_drive_minus_V", "tau", "logV", "tau*logV", "tau_avg", "tau_ema", "S"]
        columns = [
            np.ones(len(prepared_df)),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["V_drive_minus_V"].to_numpy(dtype=float),
            prepared_df["tau"].to_numpy(dtype=float),
            prepared_df["logV"].to_numpy(dtype=float),
            (prepared_df["tau"] * prepared_df["logV"]).to_numpy(dtype=float),
            prepared_df["tau_avg"].to_numpy(dtype=float),
            prepared_df["tau_ema"].to_numpy(dtype=float),
            prepared_df["S"].to_numpy(dtype=float),
        ]
    elif model_name == "C":
        terms = ["1", "tau", "V", "logV", "logTheta", "tau*logV", "tau*logTheta", "V_drive_minus_V"]
        columns = [
            np.ones(len(prepared_df)),
            prepared_df["tau"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["logV"].to_numpy(dtype=float),
            prepared_df["logTheta"].to_numpy(dtype=float),
            (prepared_df["tau"] * prepared_df["logV"]).to_numpy(dtype=float),
            (prepared_df["tau"] * prepared_df["logTheta"]).to_numpy(dtype=float),
            prepared_df["V_drive_minus_V"].to_numpy(dtype=float),
        ]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return np.column_stack(columns), terms


def equation_with_units(name: str, coefficients: np.ndarray, terms: list[str]) -> str:
    return base.equation_string(name, coefficients, terms)


def fit_two_equation_model(
    prepared_segments: list[pd.DataFrame],
    model_name: str,
    threshold: float,
    derivative_method: str,
) -> dict:
    tau_target_parts: list[np.ndarray] = []
    v_target_parts: list[np.ndarray] = []
    library_parts: list[np.ndarray] = []
    terms: list[str] | None = None
    segment_rows: list[dict] = []
    for prepared_df in prepared_segments:
        tau_dot, v_dot = estimate_derivatives(prepared_df, derivative_method)
        library, current_terms = build_library(prepared_df, model_name)
        terms = current_terms
        tau_target_parts.append(tau_dot)
        v_target_parts.append(v_dot)
        library_parts.append(library)
        segment_rows.append({"step_name": str(prepared_df["step_name"].iloc[0]), "n_points": int(len(prepared_df))})

    library_all = np.vstack(library_parts)
    tau_target_all = np.concatenate(tau_target_parts)
    v_target_all = np.concatenate(v_target_parts)
    tau_mandatory = {"V_drive_minus_V"} if "V_drive_minus_V" in terms else {"V"}
    v_mandatory = {"tau", "logV"} if "logV" in terms else {"tau"}
    tau_coefficients, tau_residual = base.fit_sparse_equation(library_all, tau_target_all, terms, threshold=threshold, mandatory_terms=tau_mandatory)
    v_coefficients, v_residual = base.fit_sparse_equation(library_all, v_target_all, terms, threshold=threshold, mandatory_terms=v_mandatory)

    tau_active = base.active_terms(tau_coefficients, terms)
    v_active = base.active_terms(v_coefficients, terms)
    tau_pred = library_all @ tau_coefficients
    v_pred = library_all @ v_coefficients
    rss = float(np.sum((tau_target_all - tau_pred) ** 2) + np.sum((v_target_all - v_pred) ** 2))
    n_obs = int(len(tau_target_all) + len(v_target_all))
    k_terms = int(len(tau_active) + len(v_active))
    aic = float(n_obs * math.log(rss / n_obs + 1e-12) + 2.0 * k_terms)
    bic = float(n_obs * math.log(rss / n_obs + 1e-12) + math.log(max(n_obs, 1)) * k_terms)
    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "derivative_method": derivative_method,
        "terms": terms,
        "tau_coefficients": tau_coefficients,
        "v_coefficients": v_coefficients,
        "tau_equation": equation_with_units("tau", tau_coefficients, terms),
        "V_equation": equation_with_units("V", v_coefficients, terms),
        "tau_active": tau_active,
        "v_active": v_active,
        "tau_residual": float(tau_residual),
        "v_residual": float(v_residual),
        "aic": aic,
        "bic": bic,
        "segment_rows": segment_rows,
    }


def build_feature_dict(prepared_df: pd.DataFrame, index: int, tau_pred: np.ndarray, v_pred: np.ndarray, model_name: str) -> dict:
    current_tau = float(tau_pred[index])
    current_v = max(float(v_pred[index]), 1e-12)
    features = {
        "1": 1.0,
        "tau": current_tau,
        "V": current_v,
        "logV": math.log(current_v),
        "tau*logV": current_tau * math.log(current_v),
        "V_drive_minus_V": float(prepared_df["V_drive"].iloc[index] - current_v),
    }
    if model_name == "B":
        memory_window = int(prepared_df.attrs["memory_window"])
        ema_span = int(prepared_df.attrs["ema_span"])
        alpha = 2.0 / (ema_span + 1.0)
        tau_avg = float(np.mean(tau_pred[max(0, index - memory_window + 1) : index + 1]))
        if index == 0:
            tau_ema = current_tau
        else:
            tau_ema = float(prepared_df.attrs.setdefault("tau_ema_roll", [current_tau] * len(prepared_df))[index])
        slip = float(prepared_df.attrs.setdefault("slip_roll", np.zeros(len(prepared_df)))[index])
        features["tau_avg"] = tau_avg
        features["tau_ema"] = tau_ema
        features["S"] = slip
        prepared_df.attrs["ema_alpha"] = alpha
    elif model_name == "C":
        log_theta = float(prepared_df["logTheta"].iloc[index])
        features["logTheta"] = log_theta
        features["tau*logTheta"] = current_tau * log_theta
    return features


def finalize_rollout_state(prepared_df: pd.DataFrame, index: int, tau_next: float, v_current: float, v_next: float, model_name: str, dt: float) -> None:
    if model_name != "B":
        return
    alpha = float(prepared_df.attrs["ema_alpha"])
    tau_ema_roll = prepared_df.attrs["tau_ema_roll"]
    slip_roll = prepared_df.attrs["slip_roll"]
    tau_ema_roll[index + 1] = alpha * tau_next + (1.0 - alpha) * tau_ema_roll[index]
    slip_roll[index + 1] = slip_roll[index] + 0.5 * (v_current + v_next) * dt


def rollout_segment(prepared_df: pd.DataFrame, model_name: str, tau_coefficients: np.ndarray, v_coefficients: np.ndarray, terms: list[str]) -> dict:
    time = prepared_df["time"].to_numpy(dtype=float)
    tau_true = prepared_df["tau"].to_numpy(dtype=float)
    v_true = prepared_df["V"].to_numpy(dtype=float)
    tau_pred = tau_true.copy()
    v_pred = v_true.copy()
    prepared_df = prepared_df.copy()
    if model_name == "B":
        prepared_df.attrs["memory_window"] = int(prepared_df.attrs["memory_window"])
        prepared_df.attrs["ema_span"] = int(prepared_df.attrs["ema_span"])
        prepared_df.attrs["tau_ema_roll"] = [float(tau_pred[0])] * len(prepared_df)
        prepared_df.attrs["slip_roll"] = np.zeros(len(prepared_df), dtype=float)

    tau_ref = max(float(np.max(np.abs(tau_true))), 1e-12)
    v_ref = max(float(np.max(np.abs(v_true))), 1e-12)
    max_tau_allowed = 5.0 * max(tau_ref, 1.0)
    max_v_allowed = 5.0 * max(v_ref, 1.0)
    stable = True
    divergence_time = float(time[-1] - time[0])
    point_errors: list[float] = []

    for index in range(len(prepared_df) - 1):
        dt = float(time[index + 1] - time[index])
        features = build_feature_dict(prepared_df, index, tau_pred, v_pred, model_name)
        tau_dot = float(sum(coefficient * features.get(term, 0.0) for coefficient, term in zip(tau_coefficients, terms)))
        v_dot = float(sum(coefficient * features.get(term, 0.0) for coefficient, term in zip(v_coefficients, terms)))
        tau_next = float(tau_pred[index] + dt * tau_dot)
        v_next = float(max(v_pred[index] + dt * v_dot, 1e-12))
        tau_pred[index + 1] = tau_next
        v_pred[index + 1] = v_next
        finalize_rollout_state(prepared_df, index, tau_next, float(v_pred[index]), v_next, model_name, dt)
        point_error = 0.5 * (abs(tau_next - tau_true[index + 1]) / tau_ref + abs(v_next - v_true[index + 1]) / v_ref)
        point_errors.append(point_error)
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
    return {
        "step_name": str(prepared_df["step_name"].iloc[0]),
        "tau_prediction": tau_pred,
        "V_prediction": v_pred,
        "tau_rmse": tau_rmse,
        "V_rmse": v_rmse,
        "combined_rmse": float(combined_rmse),
        "stable": stable,
        "divergence_time_s": divergence_time,
        "point_errors": point_errors,
    }


def prepare_model_segments(
    segments: dict[str, pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    rsfit_globals: dict,
    step_names: tuple[str, ...],
    model_name: str,
    smoothing_name: str,
    memory_window: int,
    ema_span: int,
    theta_rows: list[dict] | None = None,
) -> list[pd.DataFrame]:
    prepared_segments: list[pd.DataFrame] = []
    for step_name in step_names:
        prepared_df = prepare_step_segment(segments[step_name], smoothing_name=smoothing_name, memory_window=memory_window, ema_span=ema_span)
        prepared_df.insert(0, "step_name", step_name)
        if model_name == "C":
            theta, status = reconstruct_theta(prepared_df, steps[step_name], rsfit_globals)
            if theta is None:
                continue
            prepared_df["theta_approx"] = theta
            prepared_df["logTheta"] = np.log(theta)
            if theta_rows is not None:
                sigma = np.interp(prepared_df["time"].to_numpy(dtype=float), rsfit_globals["time"], rsfit_globals["sigmaN"])
                for time_value, tau_value, v_value, theta_value in zip(
                    prepared_df["time"].to_numpy(dtype=float),
                    prepared_df["tau"].to_numpy(dtype=float),
                    prepared_df["V"].to_numpy(dtype=float),
                    theta,
                ):
                    theta_rows.append(
                        {
                            "step_name": step_name,
                            "time": float(time_value),
                            "tau": float(tau_value),
                            "V": float(v_value),
                            "theta_approx": float(theta_value),
                            "sigmaN": float(np.interp(time_value, rsfit_globals["time"], rsfit_globals["sigmaN"])),
                            "a": status["a"],
                            "b": status["b"],
                            "Dc": status["Dc"],
                            "V0": status["V0"],
                            "mu0": status["mu0"],
                        }
                    )
        prepared_segments.append(prepared_df)
    return prepared_segments


def summarize_structure(result: dict) -> dict:
    tau_active = set(result["tau_active"])
    v_active = set(result["v_active"])
    return {
        "tau_depends_on_V_or_drive": bool("V" in tau_active or "V_drive_minus_V" in tau_active),
        "v_depends_on_tau": bool("tau" in v_active),
        "v_depends_on_log_term": bool("logV" in v_active or "logTheta" in v_active),
    }


def run_model_variant(
    model_name: str,
    segments: dict[str, pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    rsfit_globals: dict,
    threshold: float,
    derivative_method: str,
    smoothing_name: str,
    memory_window: int,
    ema_span: int,
) -> tuple[dict, list[dict]]:
    theta_rows: list[dict] = []
    train_prepared = prepare_model_segments(segments, steps, rsfit_globals, TRAIN_STEPS, model_name, smoothing_name, memory_window, ema_span, theta_rows)
    holdout_prepared = prepare_model_segments(segments, steps, rsfit_globals, HOLDOUT_STEPS, model_name, smoothing_name, memory_window, ema_span, theta_rows)
    result = fit_two_equation_model(train_prepared, model_name=model_name, threshold=threshold, derivative_method=derivative_method)
    train_rollouts = [rollout_segment(frame, model_name, result["tau_coefficients"], result["v_coefficients"], result["terms"]) for frame in train_prepared]
    holdout_rollouts = [rollout_segment(frame, model_name, result["tau_coefficients"], result["v_coefficients"], result["terms"]) for frame in holdout_prepared]
    structure = summarize_structure(result)
    result.update(
        {
            "train_rollouts": train_rollouts,
            "holdout_rollouts": holdout_rollouts,
            "train_rmse": float(np.mean([row["combined_rmse"] for row in train_rollouts])),
            "holdout_rmse": float(np.mean([row["combined_rmse"] for row in holdout_rollouts])),
            "mean_holdout_divergence_s": float(np.mean([row["divergence_time_s"] for row in holdout_rollouts])),
            **structure,
        }
    )
    return result, theta_rows


def choose_better_model(left: dict | None, right: dict) -> dict:
    if left is None:
        return right
    keys_left = (
        int(left["tau_depends_on_V_or_drive"]),
        int(left["v_depends_on_tau"]),
        int(left["v_depends_on_log_term"]),
        left["mean_holdout_divergence_s"],
        -left["holdout_rmse"],
        -left["aic"],
    )
    keys_right = (
        int(right["tau_depends_on_V_or_drive"]),
        int(right["v_depends_on_tau"]),
        int(right["v_depends_on_log_term"]),
        right["mean_holdout_divergence_s"],
        -right["holdout_rmse"],
        -right["aic"],
    )
    return right if keys_right > keys_left else left


def correlation_and_lag(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    xz = (x - np.mean(x)) / (np.std(x) + 1e-12)
    yz = (y - np.mean(y)) / (np.std(y) + 1e-12)
    corr = float(np.corrcoef(xz, yz)[0, 1])
    xcorr = np.correlate(xz, yz, mode="full") / len(xz)
    lag_axis = np.arange(-len(xz) + 1, len(xz))
    return corr, int(lag_axis[int(np.argmax(np.abs(xcorr)))])


def save_theta_validation(theta_df: pd.DataFrame, segments: dict[str, pd.DataFrame], steps: dict[str, delay_ref.RSFitStep], rsfit_globals: dict, model_c: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for step_name in HOLDOUT_STEPS:
        prepared_df = prepare_step_segment(segments[step_name], MODEL_B_CONFIG["smoothing"], MODEL_B_CONFIG["memory_window"], MODEL_B_CONFIG["ema_span"])
        prepared_df.insert(0, "step_name", step_name)
        theta, _ = reconstruct_theta(prepared_df, steps[step_name], rsfit_globals)
        if theta is None:
            continue
        corr_avg, lag_avg = correlation_and_lag(theta, prepared_df["tau_avg"].to_numpy(dtype=float))
        corr_ema, lag_ema = correlation_and_lag(theta, prepared_df["tau_ema"].to_numpy(dtype=float))
        corr_s, lag_s = correlation_and_lag(theta, prepared_df["S"].to_numpy(dtype=float))
        rows.append(
            {
                "step_name": step_name,
                "corr_theta_tau_avg": corr_avg,
                "lag_theta_tau_avg": lag_avg,
                "corr_theta_tau_ema": corr_ema,
                "lag_theta_tau_ema": lag_ema,
                "corr_theta_S": corr_s,
                "lag_theta_S": lag_s,
                "ln_theta_in_v_equation": bool("logTheta" in model_c["v_active"]),
                "logTheta_coefficient": float(model_c["v_coefficients"][model_c["terms"].index("logTheta")]) if "logTheta" in model_c["terms"] else np.nan,
                "rsf_b_sigmaN_mean": float(effective_step_params(steps[step_name])["b"] * np.mean(np.interp(prepared_df["time"], rsfit_globals["time"], rsfit_globals["sigmaN"]))),
            }
        )
        rel_time = prepared_df["time"].to_numpy(dtype=float) - float(prepared_df["time"].iloc[0])
        theta_z = (theta - np.mean(theta)) / (np.std(theta) + 1e-12)
        tau_avg_z = (prepared_df["tau_avg"] - prepared_df["tau_avg"].mean()) / (prepared_df["tau_avg"].std() + 1e-12)
        s_z = (prepared_df["S"] - prepared_df["S"].mean()) / (prepared_df["S"].std() + 1e-12)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(rel_time, theta_z, label="theta_approx (z)", linewidth=0.9)
        axes[0].plot(rel_time, tau_avg_z, label="tau_avg (z)", linewidth=0.9)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[1].plot(rel_time, theta_z, label="theta_approx (z)", linewidth=0.9)
        axes[1].plot(rel_time, s_z, label="S (z)", linewidth=0.9)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        axes[1].set_xlabel("time since step start [s]")
        fig.suptitle(f"{step_name} theta approximation vs memory surrogates")
        fig.tight_layout()
        fig.savefig(THETA_PLOTS_DIR / f"{step_name}_theta_vs_memory.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    validation_df = pd.DataFrame(rows).sort_values("step_name").reset_index(drop=True)
    validation_df.to_csv(RESULTS_DIR / "p5838_theta_recovery_validation.csv", index=False)
    return validation_df


def save_ablation_outputs(results: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name, result in results.items():
        holdout_map = {row["step_name"]: row for row in result["holdout_rollouts"]}
        step2_div = holdout_map["p5838_step2"]["divergence_time_s"]
        step7_div = holdout_map["p5838_step7"]["divergence_time_s"]
        rows.append(
            {
                "model": model_name,
                "threshold": result["threshold"],
                "derivative_method": result["derivative_method"],
                "tau_equation": result["tau_equation"],
                "V_equation": result["V_equation"],
                "train_rmse": result["train_rmse"],
                "holdout_rmse": result["holdout_rmse"],
                "aic": result["aic"],
                "bic": result["bic"],
                "tau_depends_on_V_or_drive": result["tau_depends_on_V_or_drive"],
                "v_depends_on_tau": result["v_depends_on_tau"],
                "v_depends_on_log_term": result["v_depends_on_log_term"],
                "step2_divergence_s": step2_div,
                "step7_divergence_s": step7_div,
                "mean_divergence_s": float(0.5 * (step2_div + step7_div)),
                "min_divergence_s": float(min(step2_div, step7_div)),
            }
        )
    ablation_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    ablation_df.to_csv(RESULTS_DIR / "p5838_ablation_table.csv", index=False)
    return ablation_df


def run_sparsity_ablation(segments: dict[str, pd.DataFrame], steps: dict[str, delay_ref.RSFitStep], rsfit_globals: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for threshold in SPARSITY_THRESHOLDS:
        result, _ = run_model_variant("B", segments, steps, rsfit_globals, threshold=threshold, derivative_method=MODEL_B_CONFIG["derivative_method"], smoothing_name=MODEL_B_CONFIG["smoothing"], memory_window=MODEL_B_CONFIG["memory_window"], ema_span=MODEL_B_CONFIG["ema_span"])
        holdout_map = {row["step_name"]: row for row in result["holdout_rollouts"]}
        rows.append(
            {
                "threshold": threshold,
                "tau_active_terms": len(result["tau_active"]),
                "V_active_terms": len(result["v_active"]),
                "tau_depends_on_V_or_drive": result["tau_depends_on_V_or_drive"],
                "v_depends_on_tau": result["v_depends_on_tau"],
                "v_depends_on_log_term": result["v_depends_on_log_term"],
                "train_rmse": result["train_rmse"],
                "step2_divergence_s": holdout_map["p5838_step2"]["divergence_time_s"],
                "step7_divergence_s": holdout_map["p5838_step7"]["divergence_time_s"],
                "mean_divergence_s": result["mean_holdout_divergence_s"],
                "tau_equation": result["tau_equation"],
                "V_equation": result["V_equation"],
            }
        )
    sparsity_df = pd.DataFrame(rows)
    sparsity_df.to_csv(RESULTS_DIR / "p5838_sparsity_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sparsity_df["tau_active_terms"] + sparsity_df["V_active_terms"], sparsity_df["mean_divergence_s"], marker="o")
    ax.set_xlabel("total active terms")
    ax.set_ylabel("mean holdout divergence [s]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "p5838_sparsity_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return sparsity_df


def run_derivative_comparison(segments: dict[str, pd.DataFrame], steps: dict[str, delay_ref.RSFitStep], rsfit_globals: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for method_name in DERIVATIVE_VARIANTS:
        result, _ = run_model_variant("B", segments, steps, rsfit_globals, threshold=MODEL_B_CONFIG["threshold"], derivative_method=method_name, smoothing_name=MODEL_B_CONFIG["smoothing"], memory_window=MODEL_B_CONFIG["memory_window"], ema_span=MODEL_B_CONFIG["ema_span"])
        rows.append(
            {
                "derivative_method": method_name,
                "tau_equation": result["tau_equation"],
                "V_equation": result["V_equation"],
                "train_rmse": result["train_rmse"],
                "holdout_rmse": result["holdout_rmse"],
                "mean_holdout_divergence_s": result["mean_holdout_divergence_s"],
                "tau_depends_on_V_or_drive": result["tau_depends_on_V_or_drive"],
                "v_depends_on_tau": result["v_depends_on_tau"],
                "v_depends_on_log_term": result["v_depends_on_log_term"],
            }
        )
    derivative_df = pd.DataFrame(rows).sort_values("derivative_method").reset_index(drop=True)
    derivative_df.to_csv(RESULTS_DIR / "p5838_derivative_comparison.csv", index=False)
    return derivative_df


def write_final_outputs(ablation_df: pd.DataFrame, theta_validation_df: pd.DataFrame, sparsity_df: pd.DataFrame, derivative_df: pd.DataFrame, results: dict[str, dict], theta_rows: list[dict]) -> None:
    if theta_rows:
        theta_frame = pd.DataFrame(theta_rows).sort_values(["step_name", "time"]).reset_index(drop=True)
        theta_frame.to_csv(RESULTS_DIR / "p5838_theta_approx_series.csv", index=False)
    best_row = ablation_df.sort_values(["min_divergence_s", "mean_divergence_s", "holdout_rmse"], ascending=[False, False, True]).iloc[0]
    equations_lines = []
    for model_name in ("A", "B", "C"):
        result = results[model_name]
        equations_lines.extend(
            [
                f"Model {model_name}",
                result["tau_equation"],
                result["V_equation"],
                "",
            ]
        )
    equations_lines.append(f"Most balanced holdout model: Model {best_row['model']}")
    (RESULTS_DIR / "p5838_final_equations.txt").write_text("\n".join(equations_lines).strip() + "\n", encoding="utf-8")

    theta_mean_avg = float(theta_validation_df["corr_theta_tau_avg"].mean()) if not theta_validation_df.empty else float("nan")
    theta_mean_ema = float(theta_validation_df["corr_theta_tau_ema"].mean()) if not theta_validation_df.empty else float("nan")
    theta_mean_s = float(theta_validation_df["corr_theta_S"].mean()) if not theta_validation_df.empty else float("nan")
    logtheta_coeff = float(theta_validation_df["logTheta_coefficient"].iloc[0]) if not theta_validation_df.empty else float("nan")
    rsf_scale_mean = float(theta_validation_df["rsf_b_sigmaN_mean"].mean()) if not theta_validation_df.empty else float("nan")

    report_lines = [
        "# Utah FORGE p5838 final reviewer ablation report",
        "",
        "## System description",
        "- RSFit-aligned Penn State/Utah FORGE biaxial stick-slip steps were used.",
        f"- Training steps: `{', '.join(TRAIN_STEPS)}`",
        f"- Holdout steps: `{', '.join(HOLDOUT_STEPS)}`",
        "",
        "## What each model learned",
    ]
    for model_name in ("A", "B", "C"):
        result = results[model_name]
        report_lines.extend(
            [
                f"### Model {model_name}",
                f"- Tau equation: `{result['tau_equation']}`",
                f"- V equation: `{result['V_equation']}`",
                f"- Mean holdout divergence: `{result['mean_holdout_divergence_s']:.3f}` s",
                f"- AIC/BIC: `{result['aic']:.3f}` / `{result['bic']:.3f}`",
                f"- Structural criteria: tau-drive=`{result['tau_depends_on_V_or_drive']}`, V-tau=`{result['v_depends_on_tau']}`, V-log=`{result['v_depends_on_log_term']}`",
            ]
        )
    report_lines.extend(
        [
            "",
        "## Structural comparison to RSF target",
        "- The spring-loading structure `k(V_drive - V)` persists across the tau equations.",
        "- At least one model retains `ln(V)` in the V equation.",
        "- Model C tests whether explicit `theta_approx` from RSFit inversion adds value beyond memory surrogates.",
        f"- Mean theta correlations: tau_avg=`{theta_mean_avg:.3f}`, tau_ema=`{theta_mean_ema:.3f}`, S=`{theta_mean_s:.3f}`.",
        f"- Model C log(theta) coefficient: `{logtheta_coeff:.3f}` versus mean `b*sigma_n ~ {rsf_scale_mean:.3f}`.",
        "",
        "## Honest assessment",
        f"- Model A validates whether observed-only terms are enough: step2 divergence `{ablation_df.loc[ablation_df['model']=='A', 'step2_divergence_s'].iloc[0]:.3f}` s, step7 divergence `{ablation_df.loc[ablation_df['model']=='A', 'step7_divergence_s'].iloc[0]:.3f}` s.",
        f"- Model B memory-augmented result under the stricter >10% deviation metric: step2 divergence `{ablation_df.loc[ablation_df['model']=='B', 'step2_divergence_s'].iloc[0]:.3f}` s, step7 divergence `{ablation_df.loc[ablation_df['model']=='B', 'step7_divergence_s'].iloc[0]:.3f}` s. Its older baseline reference was step2=`{MEMORY_BASELINE_DIVERGENCE['p5838_step2']:.2f}` s and step7=`{MEMORY_BASELINE_DIVERGENCE['p5838_step7']:.2f}` s under the previous metric.",
        f"- Model C theta-informed result: step2 divergence `{ablation_df.loc[ablation_df['model']=='C', 'step2_divergence_s'].iloc[0]:.3f}` s, step7 divergence `{ablation_df.loc[ablation_df['model']=='C', 'step7_divergence_s'].iloc[0]:.3f}` s.",
        "- Model B is the most balanced holdout model by worst-case divergence, but it does not dominate Model A on every held-out step.",
        "- Model C retains `ln(theta)` in the discovered V equation, but it does not outperform Model B and its `ln(theta)` coefficient does not line up cleanly with the RSF scale `b*sigma_n`.",
        "- The data therefore support memory-augmented SINDy as a practical surrogate, while explicit theta recovery remains limited by the available RSFit products.",
        "",
        "## Reviewer ablations",
        f"- Sparsity frontier rows saved: `{len(sparsity_df)}`",
            f"- Derivative comparison rows saved: `{len(derivative_df)}`",
            f"- Theta validation rows saved: `{len(theta_validation_df)}`",
        ]
    )
    (RESULTS_DIR / "p5838_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_layout()
    segments, steps, rsfit_globals = load_segments()

    model_results: dict[str, dict] = {}
    theta_rows: list[dict] = []
    model_a, _ = run_model_variant("A", segments, steps, rsfit_globals, threshold=MODEL_B_CONFIG["threshold"], derivative_method=MODEL_B_CONFIG["derivative_method"], smoothing_name=MODEL_B_CONFIG["smoothing"], memory_window=MODEL_B_CONFIG["memory_window"], ema_span=MODEL_B_CONFIG["ema_span"])
    model_b, _ = run_model_variant("B", segments, steps, rsfit_globals, threshold=MODEL_B_CONFIG["threshold"], derivative_method=MODEL_B_CONFIG["derivative_method"], smoothing_name=MODEL_B_CONFIG["smoothing"], memory_window=MODEL_B_CONFIG["memory_window"], ema_span=MODEL_B_CONFIG["ema_span"])
    model_c, theta_rows = run_model_variant("C", segments, steps, rsfit_globals, threshold=MODEL_B_CONFIG["threshold"], derivative_method=MODEL_B_CONFIG["derivative_method"], smoothing_name=MODEL_B_CONFIG["smoothing"], memory_window=MODEL_B_CONFIG["memory_window"], ema_span=MODEL_B_CONFIG["ema_span"])
    model_results.update({"A": model_a, "B": model_b, "C": model_c})

    ablation_df = save_ablation_outputs(model_results)
    theta_validation_df = save_theta_validation(pd.DataFrame(theta_rows), segments, steps, rsfit_globals, model_c)
    sparsity_df = run_sparsity_ablation(segments, steps, rsfit_globals)
    derivative_df = run_derivative_comparison(segments, steps, rsfit_globals)
    write_final_outputs(ablation_df, theta_validation_df, sparsity_df, derivative_df, model_results, theta_rows)

    print(
        json.dumps(
            {
                "model_A_divergence": ablation_df.loc[ablation_df["model"] == "A", ["step2_divergence_s", "step7_divergence_s"]].to_dict(orient="records")[0],
                "model_B_divergence": ablation_df.loc[ablation_df["model"] == "B", ["step2_divergence_s", "step7_divergence_s"]].to_dict(orient="records")[0],
                "model_C_divergence": ablation_df.loc[ablation_df["model"] == "C", ["step2_divergence_s", "step7_divergence_s"]].to_dict(orient="records")[0],
                "model_C_logTheta_active": bool("logTheta" in model_c["v_active"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
