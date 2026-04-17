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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from src.preprocess.common import smooth_series
from src.utils.paths import ensure_directory


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
ROLLOUT_DIR = RESULTS_DIR / "p5838_memory_rollouts"
THETA_PLOTS_DIR = RESULTS_DIR / "p5838_memory_theta_plots"
MEMORY_PLOTS_DIR = RESULTS_DIR / "p5838_memory_feature_plots"

SMOOTHING_CHOICES = ("moderate", "strong")
DERIVATIVE_CHOICES = ("savgol", "spline")
MEMORY_WINDOWS = (5, 10, 20)
EMA_SPANS = (5, 10, 20)
THRESHOLDS = (0.002, 0.005, 0.01)
OPTIONAL_AVG_FEATURES = (False, True)
MIN_STEP_ROWS = 1_500
MIN_POSITIVE_FRACTION = 0.98
HOLDOUT_LONG_STEPS = 2


def ensure_layout() -> None:
    ensure_directory(RESULTS_DIR)
    ensure_directory(ROLLOUT_DIR)
    ensure_directory(THETA_PLOTS_DIR)
    ensure_directory(MEMORY_PLOTS_DIR)


def cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    if len(values) <= 1:
        return result
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    result[1:] = np.cumsum(increments)
    return result


def segment_step_windows(state_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, delay_ref.RSFitStep]]:
    steps = delay_ref.load_rsfit_steps()
    rows: list[dict] = []
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
        rows.append(
            {
                "step_name": step_name,
                "time_start": float(time_values[0]),
                "time_end": float(time_values[-1]),
                "duration_s": float(time_values[-1] - time_values[0]),
                "n_rows": int(len(segment)),
                "positive_fraction": float((segment["V"] > 0).mean()),
                "V_median": float(segment["V"].median()),
                "V_mean": float(segment["V"].mean()),
                "tau_std": float(segment["tau"].std()),
                "step_r2": float(step.params["R2"]),
                "initial_velocity": float(step.params["InitialVelocity"]),
                "final_velocity": float(step.params["FinalVelocity"]),
            }
        )
    inventory_df = pd.DataFrame(rows).sort_values("time_start").reset_index(drop=True)
    inventory_df.to_csv(RESULTS_DIR / "p5838_rsfit_segment_inventory.csv", index=False)
    return inventory_df, segments, steps


def split_train_holdout_segments(inventory_df: pd.DataFrame, segments: dict[str, pd.DataFrame]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str], list[str]]:
    eligible = inventory_df.loc[
        (inventory_df["n_rows"] >= MIN_STEP_ROWS) & (inventory_df["positive_fraction"] >= MIN_POSITIVE_FRACTION)
    ].copy()
    eligible = eligible.sort_values(["duration_s", "n_rows"], ascending=[False, False]).reset_index(drop=True)
    holdout_names = eligible.head(HOLDOUT_LONG_STEPS)["step_name"].tolist()
    train_names = eligible.iloc[HOLDOUT_LONG_STEPS:]["step_name"].tolist()
    train_segments = [segments[name].copy() for name in train_names]
    holdout_segments = [segments[name].copy() for name in holdout_names]
    return train_segments, holdout_segments, train_names, holdout_names


def prepare_memory_segment(
    segment_df: pd.DataFrame,
    smoothing_name: str,
    memory_window: int,
    ema_span: int,
    include_optional_avgs: bool,
) -> tuple[pd.DataFrame, dict]:
    window = base.SIGNAL_WINDOWS[smoothing_name]
    working = segment_df[["time", "tau", "V", "V_drive"]].copy()
    working = base.downsample_frame(working, max_points=base.MODEL_MAX_POINTS)
    working = base.enforce_monotonic_time(working)

    tau_smooth = smooth_series(working["tau"].to_numpy(dtype=float), window=window, polyorder=3)
    v_smooth = smooth_series(working["V"].to_numpy(dtype=float), window=window, polyorder=3)
    positive_values = v_smooth[v_smooth > 0]
    velocity_floor = max(base._safe_quantile(positive_values, 0.01, 1e-3) * 0.25, 1e-3)
    v_positive = np.clip(v_smooth, velocity_floor, None)
    log_v = np.log(v_positive)
    time = working["time"].to_numpy(dtype=float)
    tau_avg = pd.Series(tau_smooth).rolling(window=memory_window, min_periods=1).mean().to_numpy(dtype=float)
    tau_ema = pd.Series(tau_smooth).ewm(span=ema_span, adjust=False).mean().to_numpy(dtype=float)
    slip = cumulative_trapezoid(v_positive, time)

    prepared = pd.DataFrame(
        {
            "time": time,
            "tau": tau_smooth,
            "V": v_positive,
            "V_drive": working["V_drive"].to_numpy(dtype=float),
            "logV": log_v,
            "tau_avg": tau_avg,
            "tau_ema": tau_ema,
            "S": slip,
        }
    )
    prepared["V_drive_minus_V"] = prepared["V_drive"] - prepared["V"]
    if include_optional_avgs:
        prepared["V_avg"] = pd.Series(prepared["V"]).rolling(window=memory_window, min_periods=1).mean().to_numpy(dtype=float)
        prepared["logV_avg"] = pd.Series(prepared["logV"]).rolling(window=memory_window, min_periods=1).mean().to_numpy(dtype=float)
    prepared = base.remove_invalid_rows(prepared)
    metadata = {
        "smoothing": smoothing_name,
        "signal_window": window,
        "memory_window": memory_window,
        "ema_span": ema_span,
        "velocity_floor": float(velocity_floor),
        "include_optional_avgs": include_optional_avgs,
    }
    return prepared.reset_index(drop=True), metadata


def build_memory_libraries(prepared_df: pd.DataFrame, include_optional_avgs: bool) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    tau_terms = ["1", "V", "V_drive_minus_V"]
    tau_library = np.column_stack(
        [
            np.ones(len(prepared_df)),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["V_drive_minus_V"].to_numpy(dtype=float),
        ]
    )

    v_terms = ["tau", "V", "logV", "tau*logV", "tau_avg", "tau_ema", "S"]
    columns = [
        prepared_df["tau"].to_numpy(dtype=float),
        prepared_df["V"].to_numpy(dtype=float),
        prepared_df["logV"].to_numpy(dtype=float),
        (prepared_df["tau"] * prepared_df["logV"]).to_numpy(dtype=float),
        prepared_df["tau_avg"].to_numpy(dtype=float),
        prepared_df["tau_ema"].to_numpy(dtype=float),
        prepared_df["S"].to_numpy(dtype=float),
    ]
    if include_optional_avgs:
        v_terms.extend(["V_avg", "logV_avg"])
        columns.extend(
            [
                prepared_df["V_avg"].to_numpy(dtype=float),
                prepared_df["logV_avg"].to_numpy(dtype=float),
            ]
        )
    v_library = np.column_stack(columns)
    return tau_library, tau_terms, v_library, v_terms


def cancellation_flag(v_coefficients: np.ndarray, v_terms: list[str], v_library: np.ndarray) -> tuple[bool, float]:
    term_to_index = {term: index for index, term in enumerate(v_terms)}
    contribution_scale = {}
    for term in ("tau", "tau_avg", "tau_ema"):
        if term in term_to_index:
            column = v_library[:, term_to_index[term]]
            contribution_scale[term] = abs(float(v_coefficients[term_to_index[term]])) * (float(np.std(column)) + 1e-12)
    pairs = [("tau", "tau_avg"), ("tau", "tau_ema"), ("tau_avg", "tau_ema")]
    max_ratio = 0.0
    for first, second in pairs:
        if first not in term_to_index or second not in term_to_index:
            continue
        coeff_first = float(v_coefficients[term_to_index[first]])
        coeff_second = float(v_coefficients[term_to_index[second]])
        scale_first = contribution_scale.get(first, 0.0)
        scale_second = contribution_scale.get(second, 0.0)
        if scale_first < 1e-6 or scale_second < 1e-6 or coeff_first == 0.0 or coeff_second == 0.0:
            continue
        if np.sign(coeff_first) == np.sign(coeff_second):
            continue
        ratio = min(scale_first, scale_second) / max(scale_first, scale_second)
        max_ratio = max(max_ratio, ratio)
    return bool(max_ratio >= 0.70), float(max_ratio)


def series_mean(values: np.ndarray, window: int, index: int) -> float:
    start = max(0, index - window + 1)
    return float(np.mean(values[start : index + 1]))


def rollout_memory_model(
    prepared_df: pd.DataFrame,
    memory_window: int,
    ema_span: int,
    include_optional_avgs: bool,
    tau_coefficients: np.ndarray,
    tau_terms: list[str],
    v_coefficients: np.ndarray,
    v_terms: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    time = prepared_df["time"].to_numpy(dtype=float)
    tau_true = prepared_df["tau"].to_numpy(dtype=float)
    v_true = prepared_df["V"].to_numpy(dtype=float)
    v_drive = prepared_df["V_drive"].to_numpy(dtype=float)
    velocity_floor = max(float(np.min(v_true)), 1e-6)

    tau_pred = tau_true.copy()
    v_pred = v_true.copy()
    tau_ema_pred = np.zeros(len(prepared_df), dtype=float)
    tau_ema_pred[0] = tau_pred[0]
    slip_pred = np.zeros(len(prepared_df), dtype=float)
    alpha = 2.0 / (ema_span + 1.0)

    tau_scale = float(np.std(tau_true)) or 1.0
    v_scale = float(np.std(v_true)) or 1.0
    max_tau_allowed = 5.0 * max(float(np.max(np.abs(tau_true))), 1.0)
    max_v_allowed = 5.0 * max(float(np.max(np.abs(v_true))), 1.0)
    stable = True
    divergence_time = float(time[-1] - time[0])
    error_series: list[float] = []

    for index in range(len(prepared_df) - 1):
        dt = float(time[index + 1] - time[index])
        current_tau = float(tau_pred[index])
        current_v = max(float(v_pred[index]), velocity_floor)
        current_log_v = math.log(current_v)
        tau_avg = series_mean(tau_pred, memory_window, index)
        tau_ema = float(tau_ema_pred[index])
        slip = float(slip_pred[index])
        features = {
            "1": 1.0,
            "V": current_v,
            "V_drive_minus_V": float(v_drive[index] - current_v),
            "tau": current_tau,
            "logV": current_log_v,
            "tau*logV": current_tau * current_log_v,
            "tau_avg": tau_avg,
            "tau_ema": tau_ema,
            "S": slip,
        }
        if include_optional_avgs:
            v_avg = series_mean(v_pred, memory_window, index)
            log_v_history = np.log(np.clip(v_pred[max(0, index - memory_window + 1) : index + 1], velocity_floor, None))
            features["V_avg"] = float(v_avg)
            features["logV_avg"] = float(np.mean(log_v_history))

        tau_dot = float(sum(coefficient * features[term] for coefficient, term in zip(tau_coefficients, tau_terms)))
        v_dot = float(sum(coefficient * features[term] for coefficient, term in zip(v_coefficients, v_terms)))
        tau_next = current_tau + dt * tau_dot
        v_next = max(current_v + dt * v_dot, velocity_floor)
        tau_pred[index + 1] = tau_next
        v_pred[index + 1] = v_next
        tau_ema_pred[index + 1] = alpha * tau_next + (1.0 - alpha) * tau_ema
        slip_pred[index + 1] = slip + 0.5 * (current_v + v_next) * dt

        point_error = 0.5 * (
            abs(tau_pred[index + 1] - tau_true[index + 1]) / (tau_scale + 1e-12)
            + abs(v_pred[index + 1] - v_true[index + 1]) / (v_scale + 1e-12)
        )
        error_series.append(float(point_error))
        if point_error > 2.0 and divergence_time == float(time[-1] - time[0]):
            divergence_time = float(time[index + 1] - time[0])
        if (
            not np.isfinite(tau_pred[index + 1])
            or not np.isfinite(v_pred[index + 1])
            or abs(tau_pred[index + 1]) > max_tau_allowed
            or abs(v_pred[index + 1]) > max_v_allowed
        ):
            stable = False
            tau_pred[index + 1 :] = np.nan
            v_pred[index + 1 :] = np.nan
            divergence_time = float(time[index] - time[0])
            break

    if stable and np.isfinite(tau_pred).all() and np.isfinite(v_pred).all():
        tau_rel = float(np.linalg.norm(tau_pred - tau_true) / (np.linalg.norm(tau_true) + 1e-12))
        v_rel = float(np.linalg.norm(v_pred - v_true) / (np.linalg.norm(v_true) + 1e-12))
    else:
        tau_rel = float("inf")
        v_rel = float("inf")
        stable = False
    return tau_pred, v_pred, {
        "stable": stable,
        "tau_relative_error": tau_rel,
        "V_relative_error": v_rel,
        "rollout_error": 0.5 * (tau_rel + v_rel),
        "divergence_time_s": divergence_time,
        "error_series": error_series,
    }


def fit_memory_configuration(
    train_segments: list[pd.DataFrame],
    holdout_segments: list[pd.DataFrame],
    smoothing_name: str,
    derivative_method: str,
    threshold: float,
    memory_window: int,
    ema_span: int,
    include_optional_avgs: bool,
) -> dict:
    prepared_train: list[pd.DataFrame] = []
    train_metadata: list[dict] = []
    for segment_df in train_segments:
        prepared_df, metadata = prepare_memory_segment(
            segment_df,
            smoothing_name=smoothing_name,
            memory_window=memory_window,
            ema_span=ema_span,
            include_optional_avgs=include_optional_avgs,
        )
        prepared_train.append(prepared_df)
        train_metadata.append({"step_name": str(segment_df["step_name"].iloc[0]), "n_points": int(len(prepared_df)), **metadata})

    tau_targets: list[np.ndarray] = []
    v_targets: list[np.ndarray] = []
    tau_library_parts: list[np.ndarray] = []
    v_library_parts: list[np.ndarray] = []
    tau_terms: list[str] | None = None
    v_terms: list[str] | None = None
    for prepared_df in prepared_train:
        tau_dot, v_dot = base.estimate_derivatives(prepared_df, method=derivative_method, smoothing_name=smoothing_name)
        tau_library, current_tau_terms, v_library, current_v_terms = build_memory_libraries(prepared_df, include_optional_avgs=include_optional_avgs)
        tau_terms = current_tau_terms
        v_terms = current_v_terms
        tau_targets.append(tau_dot)
        v_targets.append(v_dot)
        tau_library_parts.append(tau_library)
        v_library_parts.append(v_library)

    tau_library_all = np.vstack(tau_library_parts)
    v_library_all = np.vstack(v_library_parts)
    tau_target_all = np.concatenate(tau_targets)
    v_target_all = np.concatenate(v_targets)

    tau_coefficients, tau_residual = base.fit_sparse_equation(
        tau_library_all,
        tau_target_all,
        tau_terms,
        threshold=threshold,
        mandatory_terms={"V"},
    )
    v_coefficients, v_residual = base.fit_sparse_equation(
        v_library_all,
        v_target_all,
        v_terms,
        threshold=threshold,
        mandatory_terms={"tau", "logV"},
    )

    tau_active = base.active_terms(tau_coefficients, tau_terms)
    v_active = base.active_terms(v_coefficients, v_terms)
    has_tau_v_coupling = "V" in tau_active
    has_v_tau_coupling = "tau" in v_active
    has_logv = "logV" in v_active
    has_memory = any(term in v_active for term in ("tau_avg", "tau_ema", "S", "V_avg", "logV_avg"))
    tau_primary_ratio = abs(float(tau_coefficients[tau_terms.index("V")])) / (
        np.sum(np.abs(tau_coefficients)) + 1e-12
    )
    has_cancellation, cancellation_ratio = cancellation_flag(v_coefficients, v_terms, v_library_all)

    train_rollouts: list[dict] = []
    for prepared_df, metadata in zip(prepared_train, train_metadata):
        tau_pred, v_pred, rollout_metrics = rollout_memory_model(
            prepared_df,
            memory_window=memory_window,
            ema_span=ema_span,
            include_optional_avgs=include_optional_avgs,
            tau_coefficients=tau_coefficients,
            tau_terms=tau_terms,
            v_coefficients=v_coefficients,
            v_terms=v_terms,
        )
        train_rollouts.append(
            {"step_name": metadata["step_name"], "tau_prediction": tau_pred, "V_prediction": v_pred, **rollout_metrics}
        )

    holdout_rollouts: list[dict] = []
    for segment_df in holdout_segments:
        prepared_df, _ = prepare_memory_segment(
            segment_df,
            smoothing_name=smoothing_name,
            memory_window=memory_window,
            ema_span=ema_span,
            include_optional_avgs=include_optional_avgs,
        )
        tau_pred, v_pred, rollout_metrics = rollout_memory_model(
            prepared_df,
            memory_window=memory_window,
            ema_span=ema_span,
            include_optional_avgs=include_optional_avgs,
            tau_coefficients=tau_coefficients,
            tau_terms=tau_terms,
            v_coefficients=v_coefficients,
            v_terms=v_terms,
        )
        holdout_rollouts.append(
            {
                "step_name": str(segment_df["step_name"].iloc[0]),
                "prepared_df": prepared_df,
                "tau_prediction": tau_pred,
                "V_prediction": v_pred,
                **rollout_metrics,
            }
        )

    train_rollout_error = float(np.mean([row["rollout_error"] for row in train_rollouts]))
    holdout_rollout_error = float(np.mean([row["rollout_error"] for row in holdout_rollouts])) if holdout_rollouts else float("nan")
    holdout_divergence = float(np.mean([row["divergence_time_s"] for row in holdout_rollouts])) if holdout_rollouts else float("nan")
    holdout_stable = bool(all(row["stable"] for row in holdout_rollouts)) if holdout_rollouts else True
    long_improved = bool(holdout_divergence > 8.0) if holdout_rollouts else False
    stable_train = bool(all(row["stable"] for row in train_rollouts))
    total_terms = len(tau_active) + len(v_active)
    physical_valid = bool(
        has_tau_v_coupling
        and has_v_tau_coupling
        and has_logv
        and has_memory
        and stable_train
        and long_improved
        and not has_cancellation
        and tau_primary_ratio >= 0.45
    )
    score = (
        4.0 * float(long_improved)
        + 3.0 * float(has_tau_v_coupling)
        + 3.0 * float(has_v_tau_coupling)
        + 2.0 * float(has_logv)
        + 2.0 * float(has_memory)
        + 2.0 * float(stable_train)
        + 1.5 * float(holdout_stable)
        - 2.0 * train_rollout_error
        - 2.5 * (0.0 if np.isnan(holdout_rollout_error) else holdout_rollout_error)
        - 1.5 * cancellation_ratio
        - 0.10 * total_terms
    )

    return {
        "smoothing": smoothing_name,
        "derivative_method": derivative_method,
        "threshold": float(threshold),
        "memory_window": int(memory_window),
        "ema_span": int(ema_span),
        "include_optional_avgs": bool(include_optional_avgs),
        "tau_terms": tau_terms,
        "V_terms": v_terms,
        "tau_coefficients": tau_coefficients.tolist(),
        "V_coefficients": v_coefficients.tolist(),
        "tau_equation": base.equation_string("tau", tau_coefficients, tau_terms),
        "V_equation": base.equation_string("V", v_coefficients, v_terms),
        "tau_residual": float(tau_residual),
        "V_residual": float(v_residual),
        "train_rollout_error": train_rollout_error,
        "holdout_rollout_error": holdout_rollout_error,
        "mean_holdout_divergence_s": holdout_divergence,
        "stable_train": stable_train,
        "stable_holdout": holdout_stable,
        "has_tau_v_coupling": has_tau_v_coupling,
        "has_v_tau_coupling": has_v_tau_coupling,
        "has_logv": has_logv,
        "has_memory_feature": has_memory,
        "tau_primary_ratio": float(tau_primary_ratio),
        "has_cancellation": has_cancellation,
        "cancellation_ratio": cancellation_ratio,
        "long_divergence_improved": long_improved,
        "physical_valid": physical_valid,
        "score": float(score),
        "total_terms": int(total_terms),
        "train_rollouts": train_rollouts,
        "holdout_rollouts": holdout_rollouts,
        "train_metadata": train_metadata,
    }


def choose_best_result(results: list[dict]) -> dict:
    return sorted(
        results,
        key=lambda row: (
            int(row["physical_valid"]),
            int(row["long_divergence_improved"]),
            int(not row["has_cancellation"]),
            row["score"],
            -row["holdout_rollout_error"] if not np.isnan(row["holdout_rollout_error"]) else float("-inf"),
            row["mean_holdout_divergence_s"] if not np.isnan(row["mean_holdout_divergence_s"]) else float("-inf"),
            -row["train_rollout_error"],
            -row["total_terms"],
        ),
        reverse=True,
    )[0]


def flatten_result(result: dict) -> dict:
    return {
        "smoothing": result["smoothing"],
        "derivative_method": result["derivative_method"],
        "threshold": result["threshold"],
        "memory_window": result["memory_window"],
        "ema_span": result["ema_span"],
        "include_optional_avgs": result["include_optional_avgs"],
        "tau_terms_active": "|".join(base.active_terms(np.asarray(result["tau_coefficients"]), result["tau_terms"])),
        "V_terms_active": "|".join(base.active_terms(np.asarray(result["V_coefficients"]), result["V_terms"])),
        "tau_residual": result["tau_residual"],
        "V_residual": result["V_residual"],
        "train_rollout_error": result["train_rollout_error"],
        "holdout_rollout_error": result["holdout_rollout_error"],
        "mean_holdout_divergence_s": result["mean_holdout_divergence_s"],
        "stable_train": result["stable_train"],
        "stable_holdout": result["stable_holdout"],
        "has_logv": result["has_logv"],
        "has_memory_feature": result["has_memory_feature"],
        "tau_primary_ratio": result["tau_primary_ratio"],
        "has_cancellation": result["has_cancellation"],
        "cancellation_ratio": result["cancellation_ratio"],
        "long_divergence_improved": result["long_divergence_improved"],
        "physical_valid": result["physical_valid"],
        "score": result["score"],
        "total_terms": result["total_terms"],
        "tau_equation": result["tau_equation"],
        "V_equation": result["V_equation"],
    }


def sweep_memory_models(train_segments: list[pd.DataFrame], holdout_segments: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    results: list[dict] = []
    rows: list[dict] = []
    for smoothing_name in SMOOTHING_CHOICES:
        for derivative_method in DERIVATIVE_CHOICES:
            for threshold in THRESHOLDS:
                for memory_window in MEMORY_WINDOWS:
                    for ema_span in EMA_SPANS:
                        for include_optional_avgs in OPTIONAL_AVG_FEATURES:
                            result = fit_memory_configuration(
                                train_segments=train_segments,
                                holdout_segments=holdout_segments,
                                smoothing_name=smoothing_name,
                                derivative_method=derivative_method,
                                threshold=threshold,
                                memory_window=memory_window,
                                ema_span=ema_span,
                                include_optional_avgs=include_optional_avgs,
                            )
                            results.append(result)
                            row = flatten_result(result)
                            for term, coefficient in zip(result["tau_terms"], result["tau_coefficients"]):
                                row[f"tau_coef__{term}"] = float(coefficient)
                            for term, coefficient in zip(result["V_terms"], result["V_coefficients"]):
                                row[f"V_coef__{term}"] = float(coefficient)
                            rows.append(row)
    sweep_df = pd.DataFrame(rows).sort_values(
        ["physical_valid", "long_divergence_improved", "holdout_rollout_error", "has_cancellation"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    sweep_df.to_csv(RESULTS_DIR / "p5838_memory_model_sweep.csv", index=False)
    return sweep_df, choose_best_result(results)


def plot_memory_sweep(sweep_df: pd.DataFrame) -> None:
    top = sweep_df.head(20).copy()
    top["config_label"] = (
        "w"
        + top["memory_window"].astype(str)
        + "_e"
        + top["ema_span"].astype(str)
        + "_"
        + top["derivative_method"].astype(str)
        + "_"
        + top["smoothing"].astype(str)
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(top["config_label"], top["holdout_rollout_error"], color="tab:blue", alpha=0.8)
    axes[0].set_ylabel("holdout error")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(top["config_label"], top["mean_holdout_divergence_s"], color="tab:green", alpha=0.8)
    axes[1].axhline(8.0, color="tab:red", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("divergence [s]")
    axes[1].set_xlabel("top memory-model configurations")
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(MEMORY_PLOTS_DIR / "p5838_memory_sweep_top20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_long_rollout_plots(best_result: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for rollout in best_result["holdout_rollouts"]:
        prepared_df = rollout["prepared_df"]
        rel_time = prepared_df["time"].to_numpy(dtype=float) - float(prepared_df["time"].iloc[0])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(rel_time, prepared_df["tau"], label="true", linewidth=0.9)
        axes[0].plot(rel_time, rollout["tau_prediction"], label="pred", linewidth=0.9)
        axes[0].set_ylabel("tau")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[1].plot(rel_time, prepared_df["V"], label="true", linewidth=0.9)
        axes[1].plot(rel_time, rollout["V_prediction"], label="pred", linewidth=0.9)
        axes[1].set_ylabel("V")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        error_series = rollout["error_series"]
        axes[2].plot(rel_time[1 : 1 + len(error_series)], error_series, linewidth=0.9)
        axes[2].axhline(2.0, color="tab:red", linestyle="--", alpha=0.6)
        axes[2].set_ylabel("point error")
        axes[2].set_xlabel("time since step start [s]")
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(f"{rollout['step_name']} memory-model holdout rollout")
        fig.tight_layout()
        fig.savefig(ROLLOUT_DIR / f"{rollout['step_name']}_memory_rollout.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        rows.append(
            {
                "step_name": rollout["step_name"],
                "stable": rollout["stable"],
                "tau_relative_error": rollout["tau_relative_error"],
                "V_relative_error": rollout["V_relative_error"],
                "rollout_error": rollout["rollout_error"],
                "divergence_time_s": rollout["divergence_time_s"],
            }
        )
    long_df = pd.DataFrame(rows).sort_values("step_name").reset_index(drop=True)
    long_df.to_csv(RESULTS_DIR / "p5838_memory_long_rollout_summary.csv", index=False)
    return long_df


def fit_segment_consistency(train_segments: list[pd.DataFrame], best_result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    coefficient_rows: list[dict] = []
    for segment_df in train_segments:
        result = fit_memory_configuration(
            train_segments=[segment_df],
            holdout_segments=[],
            smoothing_name=best_result["smoothing"],
            derivative_method=best_result["derivative_method"],
            threshold=best_result["threshold"],
            memory_window=best_result["memory_window"],
            ema_span=best_result["ema_span"],
            include_optional_avgs=best_result["include_optional_avgs"],
        )
        row = flatten_result(result)
        row["step_name"] = str(segment_df["step_name"].iloc[0])
        rows.append(row)
        for equation_name, terms_key, coefficients_key in (
            ("tau", "tau_terms", "tau_coefficients"),
            ("V", "V_terms", "V_coefficients"),
        ):
            for term, coefficient in zip(result[terms_key], result[coefficients_key]):
                coefficient_rows.append(
                    {
                        "step_name": str(segment_df["step_name"].iloc[0]),
                        "equation": equation_name,
                        "term": term,
                        "coefficient": float(coefficient),
                    }
                )
    segment_df = pd.DataFrame(rows).sort_values("step_name").reset_index(drop=True)
    segment_df.to_csv(RESULTS_DIR / "p5838_memory_segment_consistency.csv", index=False)
    coefficient_df = pd.DataFrame(coefficient_rows)
    summary_df = (
        coefficient_df.groupby(["equation", "term"], as_index=False)
        .agg(coefficient_mean=("coefficient", "mean"), coefficient_std=("coefficient", "std"))
        .reset_index(drop=True)
    )
    summary_df.to_csv(RESULTS_DIR / "p5838_memory_coefficient_stability.csv", index=False)
    return segment_df, summary_df


def validate_memory_against_theta(
    segment_names: list[str],
    segment_lookup: dict[str, pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    best_result: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for step_name in segment_names:
        segment_df = segment_lookup[step_name]
        prepared_df, _ = prepare_memory_segment(
            segment_df,
            smoothing_name=best_result["smoothing"],
            memory_window=best_result["memory_window"],
            ema_span=best_result["ema_span"],
            include_optional_avgs=best_result["include_optional_avgs"],
        )
        step = steps[step_name]
        theta_interp = np.interp(prepared_df["time"].to_numpy(dtype=float), step.time, step.theta_eff)
        theta_z = (theta_interp - np.mean(theta_interp)) / (np.std(theta_interp) + 1e-12)
        corr_tau_avg = float(np.corrcoef(prepared_df["tau_avg"], theta_interp)[0, 1])
        corr_tau_ema = float(np.corrcoef(prepared_df["tau_ema"], theta_interp)[0, 1])
        corr_slip = float(np.corrcoef(prepared_df["S"], theta_interp)[0, 1])
        rows.append(
            {
                "step_name": step_name,
                "corr_tau_avg_theta": corr_tau_avg,
                "corr_tau_ema_theta": corr_tau_ema,
                "corr_slip_theta": corr_slip,
                "step_r2": float(step.params["R2"]),
            }
        )
        rel_time = prepared_df["time"].to_numpy(dtype=float) - float(prepared_df["time"].iloc[0])
        tau_avg_z = (prepared_df["tau_avg"] - prepared_df["tau_avg"].mean()) / (prepared_df["tau_avg"].std() + 1e-12)
        tau_ema_z = (prepared_df["tau_ema"] - prepared_df["tau_ema"].mean()) / (prepared_df["tau_ema"].std() + 1e-12)
        slip_z = (prepared_df["S"] - prepared_df["S"].mean()) / (prepared_df["S"].std() + 1e-12)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(rel_time, theta_z, label="theta_eff (z)", linewidth=0.9)
        axes[0].plot(rel_time, tau_avg_z, label="tau_avg (z)", linewidth=0.9)
        axes[0].plot(rel_time, tau_ema_z, label="tau_ema (z)", linewidth=0.9)
        axes[0].set_ylabel("z-score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[1].plot(rel_time, theta_z, label="theta_eff (z)", linewidth=0.9)
        axes[1].plot(rel_time, slip_z, label="S (z)", linewidth=0.9)
        axes[1].set_ylabel("z-score")
        axes[1].set_xlabel("time since step start [s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        fig.suptitle(f"{step_name} memory-feature vs theta alignment")
        fig.tight_layout()
        fig.savefig(THETA_PLOTS_DIR / f"{step_name}_memory_theta_alignment.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    theta_df = pd.DataFrame(rows).sort_values("step_name").reset_index(drop=True)
    theta_df.to_csv(RESULTS_DIR / "p5838_memory_theta_alignment.csv", index=False)
    return theta_df


def write_outputs(
    inventory_df: pd.DataFrame,
    train_names: list[str],
    holdout_names: list[str],
    sweep_df: pd.DataFrame,
    best_result: dict,
    long_rollout_df: pd.DataFrame,
    segment_consistency_df: pd.DataFrame,
    coefficient_stability_df: pd.DataFrame,
    theta_df: pd.DataFrame,
) -> None:
    equations_path = RESULTS_DIR / "p5838_memory_best_equations.txt"
    equations_path.write_text(best_result["tau_equation"] + "\n" + best_result["V_equation"] + "\n", encoding="utf-8")

    summary = {
        "train_steps": train_names,
        "holdout_steps": holdout_names,
        "best_model": flatten_result(best_result),
        "tau_terms": best_result["tau_terms"],
        "V_terms": best_result["V_terms"],
        "tau_coefficients": best_result["tau_coefficients"],
        "V_coefficients": best_result["V_coefficients"],
        "long_rollout_rows": long_rollout_df.to_dict(orient="records"),
        "theta_alignment_rows": theta_df.to_dict(orient="records"),
        "coefficient_stability_rows": coefficient_stability_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "p5838_memory_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Utah FORGE p5838 memory-model refinement",
        "",
        "## Data split",
        f"- Training RSFit-aligned steps: `{', '.join(train_names)}`",
        f"- Holdout long steps: `{', '.join(holdout_names)}`",
        "",
        "## Best model",
        f"- Smoothing: `{best_result['smoothing']}`",
        f"- Derivative method: `{best_result['derivative_method']}`",
        f"- Sparsity threshold: `{best_result['threshold']}`",
        f"- Rolling window: `{best_result['memory_window']}` samples",
        f"- EMA span: `{best_result['ema_span']}` samples",
        f"- Optional average features: `{best_result['include_optional_avgs']}`",
        f"- Cancellation detected: `{best_result['has_cancellation']}`",
        f"- Mean holdout divergence time: `{best_result['mean_holdout_divergence_s']:.3f}` s",
        f"- Long-rollout improvement beyond 8 s: `{best_result['long_divergence_improved']}`",
        "",
        "## Equations",
        "```text",
        best_result["tau_equation"],
        best_result["V_equation"],
        "```",
        "",
        "## Theta alignment",
        f"- Mean corr(tau_avg, theta): `{theta_df['corr_tau_avg_theta'].mean():.4f}`",
        f"- Mean corr(tau_ema, theta): `{theta_df['corr_tau_ema_theta'].mean():.4f}`",
        f"- Mean corr(S, theta): `{theta_df['corr_slip_theta'].mean():.4f}`",
        "",
        "## Holdout rollout",
        f"- Mean holdout rollout error: `{long_rollout_df['rollout_error'].replace(np.inf, np.nan).mean():.4f}`",
        f"- Mean holdout divergence time: `{long_rollout_df['divergence_time_s'].mean():.4f}` s",
        "",
        "## Notes",
        "- Raw `tau(t-Δ)` lag terms were removed from the feature library.",
        "- Memory is represented through `tau_avg`, `tau_ema`, and cumulative slip `S`.",
        "- `log(V)` was retained as a required term in the `V` equation.",
    ]
    (RESULTS_DIR / "p5838_memory_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    inventory_df.to_csv(RESULTS_DIR / "p5838_rsfit_segment_inventory.csv", index=False)
    sweep_df.head(20).to_csv(RESULTS_DIR / "p5838_memory_model_top20.csv", index=False)


def main() -> None:
    ensure_layout()
    state_df, _ = base.load_p5838_state()
    inventory_df, segments, steps = segment_step_windows(state_df)
    train_segments, holdout_segments, train_names, holdout_names = split_train_holdout_segments(inventory_df, segments)
    if not train_segments or not holdout_segments:
        raise RuntimeError("Could not build both training and holdout RSFit-aligned segments for p5838.")

    sweep_df, best_result = sweep_memory_models(train_segments, holdout_segments)
    plot_memory_sweep(sweep_df)
    long_rollout_df = save_long_rollout_plots(best_result)
    segment_consistency_df, coefficient_stability_df = fit_segment_consistency(train_segments, best_result)
    theta_df = validate_memory_against_theta(train_names + holdout_names, segments, steps, best_result)
    write_outputs(
        inventory_df=inventory_df,
        train_names=train_names,
        holdout_names=holdout_names,
        sweep_df=sweep_df,
        best_result=best_result,
        long_rollout_df=long_rollout_df,
        segment_consistency_df=segment_consistency_df,
        coefficient_stability_df=coefficient_stability_df,
        theta_df=theta_df,
    )
    print(
        json.dumps(
            {
                "train_steps": train_names,
                "holdout_steps": holdout_names,
                "best_tau_equation": best_result["tau_equation"],
                "best_V_equation": best_result["V_equation"],
                "mean_holdout_divergence_s": best_result["mean_holdout_divergence_s"],
                "long_divergence_improved": best_result["long_divergence_improved"],
                "has_cancellation": best_result["has_cancellation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
