from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
from src.derivatives import derivative_savgol
from src.utils.paths import ensure_directory


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
RSFIT_PLOTS_DIR = RESULTS_DIR / "p5838_rsfit_theta_plots"
LONG_ROLLOUT_DIR = RESULTS_DIR / "p5838_long_rollouts"
DELAY_PLOTS_DIR = RESULTS_DIR / "p5838_delay_validation_plots"
SUBSET_SEED = 798
DELAYS = tuple(range(1, 11))
MULTI_DELAY_COMBOS = ((1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 6), (4, 8))
SUBSET_RUNS = 12


@dataclass(frozen=True)
class RSFitStep:
    step_name: str
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    mu_fit: np.ndarray
    mu_obs: np.ndarray
    theta_eff: np.ndarray
    params: dict


def ensure_layout() -> None:
    ensure_directory(RESULTS_DIR)
    ensure_directory(RSFIT_PLOTS_DIR)
    ensure_directory(LONG_ROLLOUT_DIR)
    ensure_directory(DELAY_PLOTS_DIR)


def select_training_cycles(state_df: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame], list[base.EventCandidate]]:
    candidates = base.detect_clean_cycles(state_df)
    short_candidates = [candidate for candidate in candidates if candidate.duration_s <= 5.0]
    chosen_candidates = short_candidates[: base.MAX_SELECTED_EVENTS] if len(short_candidates) >= base.MAX_SELECTED_EVENTS else candidates[: base.MAX_SELECTED_EVENTS]
    rows: list[dict] = []
    selected_cycles: list[pd.DataFrame] = []
    for candidate in chosen_candidates:
        cycle = state_df.iloc[candidate.start_idx : candidate.end_idx].reset_index(drop=True).copy()
        cycle.insert(0, "event_id", candidate.event_id)
        selected_cycles.append(cycle)
        rows.append(
            {
                "event_id": candidate.event_id,
                "start_idx": candidate.start_idx,
                "end_idx": candidate.end_idx,
                "peak_idx": candidate.peak_idx,
                "trough_idx": candidate.trough_idx,
                "n_samples": int(candidate.end_idx - candidate.start_idx),
                "duration_s": candidate.duration_s,
                "tau_drop": candidate.tau_drop,
                "positive_fraction": candidate.positive_fraction,
                "velocity_range": candidate.velocity_range,
                "velocity_noise_ratio": candidate.velocity_noise_ratio,
                "score": candidate.score,
            }
        )
    event_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return event_df, selected_cycles, candidates


def resolve_rsfit_path() -> Path:
    candidates = sorted((REPO_ROOT / "data" / "utah_forge").glob("p5838_RSFit3000*.mat"))
    if not candidates:
        raise FileNotFoundError("No local p5838 RSFit MAT file was found under data/utah_forge/.")
    return candidates[0]


def prepare_cycle_with_lags(cycle_df: pd.DataFrame, smoothing_name: str, lags: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    max_lag = max(lags)
    prepared_df, metadata = base.prepare_cycle(cycle_df, smoothing_name=smoothing_name, delta=max_lag)
    working = prepared_df.copy()
    for lag in lags:
        working[f"tau_lag_{lag}"] = working["tau"].shift(lag)
    working = base.remove_invalid_rows(working)
    metadata["lags"] = list(lags)
    return working.reset_index(drop=True), metadata


def apply_global_scaling(prepared_cycles: list[pd.DataFrame], lags: tuple[int, ...]) -> tuple[list[pd.DataFrame], dict]:
    tau_values = np.concatenate([frame["tau"].to_numpy(dtype=float) for frame in prepared_cycles])
    log_values = np.concatenate([frame["logV"].to_numpy(dtype=float) for frame in prepared_cycles])
    tau_mean = float(np.mean(tau_values))
    tau_std = float(np.std(tau_values)) or 1.0
    log_mean = float(np.mean(log_values))
    log_std = float(np.std(log_values)) or 1.0

    scaled_cycles: list[pd.DataFrame] = []
    for frame in prepared_cycles:
        scaled = frame.copy()
        scaled["tau_z"] = (scaled["tau"] - tau_mean) / tau_std
        scaled["logV_z"] = (scaled["logV"] - log_mean) / log_std
        for lag in lags:
            scaled[f"tau_lag_{lag}_z"] = (scaled[f"tau_lag_{lag}"] - tau_mean) / tau_std
        scaled_cycles.append(base.remove_invalid_rows(scaled).reset_index(drop=True))

    scaling = {
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "logV_mean": log_mean,
        "logV_std": log_std,
    }
    return scaled_cycles, scaling


def build_dynamic_libraries(prepared_df: pd.DataFrame, lags: tuple[int, ...]) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    tau_terms = ["1", "tau_z", "V"]
    tau_library = np.column_stack(
        [
            np.ones(len(prepared_df)),
            prepared_df["tau_z"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
        ]
    )

    v_terms = ["tau_z", "V", "logV_z", *[f"tau_lag_{lag}_z" for lag in lags]]
    v_library = np.column_stack(
        [
            prepared_df["tau_z"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["logV_z"].to_numpy(dtype=float),
            *[prepared_df[f"tau_lag_{lag}_z"].to_numpy(dtype=float) for lag in lags],
        ]
    )
    return tau_library, tau_terms, v_library, v_terms


def rollout_dynamic(
    prepared_df: pd.DataFrame,
    lags: tuple[int, ...],
    tau_coefficients: np.ndarray,
    tau_terms: list[str],
    v_coefficients: np.ndarray,
    v_terms: list[str],
    scaling: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    time = prepared_df["time"].to_numpy(dtype=float)
    tau_true = prepared_df["tau"].to_numpy(dtype=float)
    v_true = prepared_df["V"].to_numpy(dtype=float)
    tau_pred = tau_true.copy()
    v_pred = v_true.copy()
    tau_mean = scaling["tau_mean"]
    tau_std = scaling["tau_std"]
    log_mean = scaling["logV_mean"]
    log_std = scaling["logV_std"]
    max_lag = max(lags)
    velocity_floor = float(np.min(v_true))
    tau_scale = float(np.std(tau_true)) or 1.0
    v_scale = float(np.std(v_true)) or 1.0
    max_tau_allowed = 5.0 * max(float(np.max(np.abs(tau_true))), 1.0)
    max_v_allowed = 5.0 * max(float(np.max(np.abs(v_true))), 1.0)
    stable = True
    divergence_time = float(time[-1] - time[0])
    error_series: list[float] = []

    for index in range(max_lag, len(prepared_df) - 1):
        dt = float(time[index + 1] - time[index])
        current_tau = tau_pred[index]
        current_v = max(float(v_pred[index]), velocity_floor)
        features = {
            "1": 1.0,
            "tau_z": (current_tau - tau_mean) / tau_std,
            "V": current_v,
            "logV_z": (math.log(current_v) - log_mean) / log_std,
        }
        for lag in lags:
            lag_tau = tau_pred[index - lag]
            features[f"tau_lag_{lag}_z"] = (lag_tau - tau_mean) / tau_std

        tau_dot = float(sum(coefficient * features[term] for coefficient, term in zip(tau_coefficients, tau_terms)))
        v_dot = float(sum(coefficient * features[term] for coefficient, term in zip(v_coefficients, v_terms)))
        tau_pred[index + 1] = current_tau + dt * tau_dot
        v_pred[index + 1] = max(current_v + dt * v_dot, velocity_floor)

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


def fit_dynamic_configuration(
    selected_cycles: list[pd.DataFrame],
    smoothing_name: str,
    derivative_method: str,
    threshold: float,
    lags: tuple[int, ...],
) -> dict:
    prepared_cycles: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    for cycle_df in selected_cycles:
        prepared_df, metadata = prepare_cycle_with_lags(cycle_df, smoothing_name=smoothing_name, lags=lags)
        prepared_cycles.append(prepared_df)
        metadata_rows.append({"event_id": str(cycle_df["event_id"].iloc[0]), "n_points": int(len(prepared_df)), **metadata})

    scaled_cycles, scaling = apply_global_scaling(prepared_cycles, lags=lags)
    tau_target_parts: list[np.ndarray] = []
    v_target_parts: list[np.ndarray] = []
    tau_library_parts: list[np.ndarray] = []
    v_library_parts: list[np.ndarray] = []
    tau_terms: list[str] | None = None
    v_terms: list[str] | None = None
    for scaled_df in scaled_cycles:
        tau_dot, v_dot = base.estimate_derivatives(scaled_df, method=derivative_method, smoothing_name=smoothing_name)
        tau_library, cycle_tau_terms, v_library, cycle_v_terms = build_dynamic_libraries(scaled_df, lags=lags)
        tau_terms = cycle_tau_terms
        v_terms = cycle_v_terms
        tau_target_parts.append(tau_dot)
        v_target_parts.append(v_dot)
        tau_library_parts.append(tau_library)
        v_library_parts.append(v_library)

    tau_coefficients, tau_residual = base.fit_sparse_equation(
        np.vstack(tau_library_parts),
        np.concatenate(tau_target_parts),
        tau_terms,
        threshold=threshold,
        mandatory_terms={"V"},
    )
    v_coefficients, v_residual = base.fit_sparse_equation(
        np.vstack(v_library_parts),
        np.concatenate(v_target_parts),
        v_terms,
        threshold=threshold,
        mandatory_terms={"tau_z"},
    )
    tau_active = base.active_terms(tau_coefficients, tau_terms)
    v_active = base.active_terms(v_coefficients, v_terms)
    has_tau_v_coupling = "V" in tau_active
    has_v_tau_coupling = "tau_z" in v_active
    has_hidden_state_proxy = any(term.startswith("tau_lag_") or term == "logV_z" for term in v_active)

    rollouts: list[dict] = []
    stable_all = True
    for scaled_df, cycle_meta in zip(scaled_cycles, metadata_rows):
        tau_pred, v_pred, rollout_metrics = rollout_dynamic(
            scaled_df,
            lags=lags,
            tau_coefficients=tau_coefficients,
            tau_terms=tau_terms,
            v_coefficients=v_coefficients,
            v_terms=v_terms,
            scaling=scaling,
        )
        stable_all = stable_all and rollout_metrics["stable"]
        rollouts.append({"event_id": cycle_meta["event_id"], "tau_prediction": tau_pred, "V_prediction": v_pred, **rollout_metrics})

    rollout_error = float(np.mean([row["rollout_error"] for row in rollouts]))
    tau_error = float(np.mean([row["tau_relative_error"] for row in rollouts]))
    v_error = float(np.mean([row["V_relative_error"] for row in rollouts]))
    divergence_mean = float(np.mean([row["divergence_time_s"] for row in rollouts]))
    total_terms = len(tau_active) + len(v_active)
    physical_valid = bool(has_tau_v_coupling and has_v_tau_coupling and has_hidden_state_proxy and stable_all)
    physical_score = (
        3.0 * float(has_tau_v_coupling)
        + 3.0 * float(has_v_tau_coupling)
        + 2.0 * float(has_hidden_state_proxy)
        + 2.0 * float(stable_all)
        - 2.0 * rollout_error
        - 0.15 * total_terms
    )

    return {
        "smoothing": smoothing_name,
        "derivative_method": derivative_method,
        "threshold": float(threshold),
        "lags": list(lags),
        "tau_terms": tau_terms,
        "V_terms": v_terms,
        "tau_coefficients": tau_coefficients.tolist(),
        "V_coefficients": v_coefficients.tolist(),
        "tau_equation": base.equation_string("tau", tau_coefficients, tau_terms),
        "V_equation": base.equation_string("V", v_coefficients, v_terms),
        "tau_residual": float(tau_residual),
        "V_residual": float(v_residual),
        "tau_rollout_error": tau_error,
        "V_rollout_error": v_error,
        "rollout_error": rollout_error,
        "mean_divergence_time_s": divergence_mean,
        "stable_all_cycles": stable_all,
        "has_tau_v_coupling": has_tau_v_coupling,
        "has_v_tau_coupling": has_v_tau_coupling,
        "has_hidden_state_proxy": has_hidden_state_proxy,
        "physical_valid": physical_valid,
        "physical_score": float(physical_score),
        "total_terms": total_terms,
        "scaling": scaling,
        "prepared_cycles": metadata_rows,
        "rollouts": rollouts,
    }


def choose_best_result(results: list[dict]) -> dict:
    return sorted(
        results,
        key=lambda row: (
            int(row["physical_valid"]),
            int(row["stable_all_cycles"]),
            row["physical_score"],
            -row["rollout_error"],
            -row["mean_divergence_time_s"],
            -row["total_terms"],
        ),
        reverse=True,
    )[0]


def flatten_result(result: dict) -> dict:
    lag_label = ",".join(str(lag) for lag in result["lags"])
    return {
        "lags": lag_label,
        "smoothing": result["smoothing"],
        "derivative_method": result["derivative_method"],
        "threshold": result["threshold"],
        "tau_terms_active": "|".join(base.active_terms(np.asarray(result["tau_coefficients"]), result["tau_terms"])),
        "V_terms_active": "|".join(base.active_terms(np.asarray(result["V_coefficients"]), result["V_terms"])),
        "tau_residual": result["tau_residual"],
        "V_residual": result["V_residual"],
        "tau_rollout_error": result["tau_rollout_error"],
        "V_rollout_error": result["V_rollout_error"],
        "rollout_error": result["rollout_error"],
        "mean_divergence_time_s": result["mean_divergence_time_s"],
        "stable_all_cycles": result["stable_all_cycles"],
        "physical_valid": result["physical_valid"],
        "physical_score": result["physical_score"],
        "total_terms": result["total_terms"],
        "tau_equation": result["tau_equation"],
        "V_equation": result["V_equation"],
    }


def single_delay_sweep(selected_cycles: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[int, dict]]:
    rows: list[dict] = []
    best_by_delta: dict[int, dict] = {}
    for delta in DELAYS:
        delta_results: list[dict] = []
        for smoothing_name in base.SIGNAL_WINDOWS:
            for derivative_method in base.DERIVATIVE_METHODS:
                for threshold in base.THRESHOLDS:
                    result = fit_dynamic_configuration(
                        selected_cycles=selected_cycles,
                        smoothing_name=smoothing_name,
                        derivative_method=derivative_method,
                        threshold=threshold,
                        lags=(delta,),
                    )
                    result["delta"] = delta
                    delta_results.append(result)
        best = choose_best_result(delta_results)
        best_by_delta[delta] = best
        row = flatten_result(best)
        row["delta"] = delta
        for term, coefficient in zip(best["tau_terms"], best["tau_coefficients"]):
            row[f"tau_coef__{term}"] = float(coefficient)
        for term, coefficient in zip(best["V_terms"], best["V_coefficients"]):
            row[f"V_coef__{term}"] = float(coefficient)
        rows.append(row)
    delay_df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    delay_df.to_csv(RESULTS_DIR / "p5838_delay_sweep_summary.csv", index=False)
    return delay_df, best_by_delta


def multi_delay_sweep(selected_cycles: list[pd.DataFrame], anchor_result: dict) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows: list[dict] = []
    best_results: dict[str, dict] = {}
    for combo in MULTI_DELAY_COMBOS:
        combo_results: list[dict] = []
        for threshold in base.THRESHOLDS:
            result = fit_dynamic_configuration(
                selected_cycles=selected_cycles,
                smoothing_name=anchor_result["smoothing"],
                derivative_method=anchor_result["derivative_method"],
                threshold=threshold,
                lags=combo,
            )
            combo_results.append(result)
        best = choose_best_result(combo_results)
        combo_label = ",".join(str(item) for item in combo)
        best_results[combo_label] = best
        row = flatten_result(best)
        row["combo"] = combo_label
        for term, coefficient in zip(best["V_terms"], best["V_coefficients"]):
            row[f"V_coef__{term}"] = float(coefficient)
        rows.append(row)
    multi_df = pd.DataFrame(rows).sort_values(["physical_valid", "rollout_error"], ascending=[False, True]).reset_index(drop=True)
    multi_df.to_csv(RESULTS_DIR / "p5838_multi_delay_summary.csv", index=False)
    return multi_df, best_results


def plot_delay_sweep(delay_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(delay_df["delta"], delay_df["rollout_error"], marker="o", label="rollout error")
    axes[0].plot(delay_df["delta"], delay_df["tau_residual"], marker="s", label="tau residual")
    axes[0].plot(delay_df["delta"], delay_df["V_residual"], marker="^", label="V residual")
    axes[0].set_ylabel("error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    lag_column = "V_coef__tau_lag_1_z"
    lag_coeffs = []
    for _, row in delay_df.iterrows():
        lag_coeffs.append(row.get(f"V_coef__tau_lag_{int(row['delta'])}_z", np.nan))
    axes[1].plot(delay_df["delta"], delay_df["tau_coef__V"], marker="o", label="tau eq: V")
    axes[1].plot(delay_df["delta"], delay_df["V_coef__tau_z"], marker="s", label="V eq: tau_z")
    axes[1].plot(delay_df["delta"], delay_df["V_coef__logV_z"], marker="^", label="V eq: logV_z")
    axes[1].plot(delay_df["delta"], lag_coeffs, marker="d", label="V eq: lag term")
    axes[1].set_xlabel("delay Δ [samples]")
    axes[1].set_ylabel("coefficient")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(DELAY_PLOTS_DIR / "p5838_delay_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_rsfit_steps() -> dict[str, RSFitStep]:
    rsfit_path = resolve_rsfit_path()
    mat = scipy.io.loadmat(rsfit_path, squeeze_me=True, struct_as_record=False)
    steps: dict[str, RSFitStep] = {}
    for step_index in range(1, 11):
        step_name = f"p5838_step{step_index}"
        step = mat[step_name]
        time = np.asarray(step.TimeData).reshape(-1)
        displacement = np.asarray(step.LoadPointDisplacementData).reshape(-1)
        velocity = derivative_savgol(displacement, t=time, window=101, polyorder=3)
        fit = np.asarray(step.AgingLawFit)
        mu_fit = fit[:, 1] if fit.ndim == 2 and fit.shape[1] >= 2 else np.asarray(step.FrictionDataDetrended).reshape(-1)
        mu_obs = fit[:, 2] if fit.ndim == 2 and fit.shape[1] >= 3 else np.asarray(step.FrictionData).reshape(-1)
        params = {
            "InitialVelocity": float(step.VelocityStepParameters.InitialVelocity),
            "FinalVelocity": float(step.VelocityStepParameters.FinalVelocity),
            "TimeOfStep": float(step.VelocityStepParameters.TimeOfStep),
            "a": float(np.asarray(step.AgingLawParameters.a).reshape(-1)[0]),
            "b1": float(np.asarray(step.AgingLawParameters.b1).reshape(-1)[0]),
            "b2": float(np.asarray(step.AgingLawParameters.b2).reshape(-1)[0]),
            "dc1": float(np.asarray(step.AgingLawParameters.d_c1).reshape(-1)[0]),
            "dc2": float(np.asarray(step.AgingLawParameters.d_c2).reshape(-1)[0]),
            "mu0": float(np.asarray(step.AgingLawParameters.mu_0).reshape(-1)[0]),
            "R2": float(np.asarray(step.AgingLawParameters.R_Squared).reshape(-1)[0]),
        }
        b_eff = params["b1"] + params["b2"]
        if abs(b_eff) < 1e-8:
            b_eff = params["b1"] if abs(params["b1"]) > 1e-8 else 1e-6
        if abs(params["b2"]) > 1e-12 and abs(params["b1"]) > 1e-12:
            dc_eff = math.exp((params["b1"] * math.log(max(params["dc1"], 1e-6)) + params["b2"] * math.log(max(params["dc2"], 1e-6))) / (params["b1"] + params["b2"]))
        else:
            dc_eff = max(params["dc1"], 1e-6)
        v_ref = max(params["InitialVelocity"], 1e-6)
        velocity_safe = np.clip(velocity, 1e-6, None)
        theta_eff = (dc_eff / v_ref) * np.exp((mu_fit - params["mu0"] - params["a"] * np.log(velocity_safe / v_ref)) / b_eff)
        steps[step_name] = RSFitStep(
            step_name=step_name,
            time=time,
            displacement=displacement,
            velocity=velocity_safe,
            mu_fit=mu_fit,
            mu_obs=mu_obs,
            theta_eff=theta_eff,
            params=params,
        )
    return steps


def best_step_for_cycle(cycle_df: pd.DataFrame, steps: dict[str, RSFitStep]) -> RSFitStep:
    start = float(cycle_df["time"].iloc[0])
    end = float(cycle_df["time"].iloc[-1])
    best_name = None
    best_overlap = -1.0
    for name, step in steps.items():
        overlap = max(0.0, min(end, float(step.time[-1])) - max(start, float(step.time[0])))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    return steps[best_name]


def validate_rsfit_theta(selected_cycles: list[pd.DataFrame], best_result: dict) -> pd.DataFrame:
    steps = load_rsfit_steps()
    rows: list[dict] = []
    for cycle_df in selected_cycles:
        prepared_df, _ = prepare_cycle_with_lags(cycle_df, smoothing_name=best_result["smoothing"], lags=tuple(best_result["lags"]))
        scaled = prepared_df.copy()
        lag = int(best_result["lags"][0])
        step = best_step_for_cycle(cycle_df, steps)
        theta_interp = np.interp(scaled["time"].to_numpy(dtype=float), step.time, step.theta_eff)
        mu_interp = np.interp(scaled["time"].to_numpy(dtype=float), step.time, step.mu_fit)
        tau_lag = ((scaled[f"tau_lag_{lag}"] - best_result["scaling"]["tau_mean"]) / best_result["scaling"]["tau_std"]).to_numpy(dtype=float)
        theta_z = (theta_interp - np.mean(theta_interp)) / (np.std(theta_interp) + 1e-12)
        tau_lag_z = (tau_lag - np.mean(tau_lag)) / (np.std(tau_lag) + 1e-12)
        correlation = float(np.corrcoef(tau_lag_z, theta_interp)[0, 1])
        xcorr = np.correlate(tau_lag_z, theta_z, mode="full") / len(tau_lag_z)
        lag_axis = np.arange(-len(tau_lag_z) + 1, len(tau_lag_z))
        max_index = int(np.argmax(np.abs(xcorr)))
        rows.append(
            {
                "event_id": str(cycle_df["event_id"].iloc[0]),
                "matched_step": step.step_name,
                "theta_correlation": correlation,
                "max_abs_cross_correlation": float(xcorr[max_index]),
                "lag_at_max_correlation_samples": int(lag_axis[max_index]),
                "step_r2": step.params["R2"],
            }
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        rel_time = scaled["time"].to_numpy(dtype=float) - float(scaled["time"].iloc[0])
        axes[0].plot(rel_time, tau_lag_z, label="tau_lag_z", linewidth=0.9)
        axes[0].plot(rel_time, theta_z, label="theta_eff (z)", linewidth=0.9)
        axes[0].set_ylabel("z-score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[1].plot(rel_time, mu_interp, label="mu_fit", linewidth=0.9)
        axes[1].set_ylabel("mu")
        axes[1].set_xlabel("time since cycle start [s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        fig.suptitle(f"{cycle_df['event_id'].iloc[0]} vs {step.step_name} theta validation")
        fig.tight_layout()
        fig.savefig(RSFIT_PLOTS_DIR / f"{cycle_df['event_id'].iloc[0]}_theta_validation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    rsfit_df = pd.DataFrame(rows)
    rsfit_df.to_csv(RESULTS_DIR / "p5838_rsfit_theta_validation.csv", index=False)
    return rsfit_df


def denormalize_equations(best_result: dict) -> dict:
    tau_mean = best_result["scaling"]["tau_mean"]
    tau_std = best_result["scaling"]["tau_std"]
    log_mean = best_result["scaling"]["logV_mean"]
    log_std = best_result["scaling"]["logV_std"]
    tau_coeffs = dict(zip(best_result["tau_terms"], best_result["tau_coefficients"]))
    v_coeffs = dict(zip(best_result["V_terms"], best_result["V_coefficients"]))
    lag = int(best_result["lags"][0])

    tau_const = tau_coeffs.get("1", 0.0) - tau_coeffs.get("tau_z", 0.0) * tau_mean / tau_std
    tau_tau = tau_coeffs.get("tau_z", 0.0) / tau_std
    tau_v = tau_coeffs.get("V", 0.0)

    v_const = (
        -v_coeffs.get("tau_z", 0.0) * tau_mean / tau_std
        -v_coeffs.get("logV_z", 0.0) * log_mean / log_std
        -v_coeffs.get(f"tau_lag_{lag}_z", 0.0) * tau_mean / tau_std
    )
    v_tau = v_coeffs.get("tau_z", 0.0) / tau_std
    v_v = v_coeffs.get("V", 0.0)
    v_log = v_coeffs.get("logV_z", 0.0) / log_std
    v_tau_lag = v_coeffs.get(f"tau_lag_{lag}_z", 0.0) / tau_std

    payload = {
        "tau_equation_physical": f"dtau/dt = {tau_const:.6e} + {tau_tau:.6e}*tau + {tau_v:.6e}*V",
        "V_equation_physical": f"dV/dt = {v_const:.6e} + {v_tau:.6e}*tau + {v_v:.6e}*V + {v_log:.6e}*ln(V) + {v_tau_lag:.6e}*tau(t-{lag})",
        "coefficients": {
            "tau_const": tau_const,
            "tau_tau": tau_tau,
            "tau_V": tau_v,
            "V_const": v_const,
            "V_tau": v_tau,
            "V_V": v_v,
            "V_logV": v_log,
            "V_tau_lag": v_tau_lag,
        },
    }
    (RESULTS_DIR / "p5838_physical_unit_equations.txt").write_text(
        payload["tau_equation_physical"] + "\n" + payload["V_equation_physical"] + "\n",
        encoding="utf-8",
    )
    return payload


def select_unseen_long_cycles(candidates: list[base.EventCandidate], selected_event_ids: set[str], state_df: pd.DataFrame, count: int = 2) -> list[pd.DataFrame]:
    unseen: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.event_id in selected_event_ids or candidate.duration_s <= 5.0:
            continue
        cycle = state_df.iloc[candidate.start_idx : candidate.end_idx].reset_index(drop=True).copy()
        cycle.insert(0, "event_id", candidate.event_id)
        unseen.append(cycle)
        if len(unseen) >= count:
            break
    return unseen


def evaluate_long_rollouts(unseen_cycles: list[pd.DataFrame], best_result: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for cycle_df in unseen_cycles:
        prepared_df, _ = prepare_cycle_with_lags(cycle_df, smoothing_name=best_result["smoothing"], lags=tuple(best_result["lags"]))
        scaled_cycles, scaling = apply_global_scaling([prepared_df], lags=tuple(best_result["lags"]))
        scaled_df = scaled_cycles[0]
        tau_pred, v_pred, metrics = rollout_dynamic(
            scaled_df,
            lags=tuple(best_result["lags"]),
            tau_coefficients=np.asarray(best_result["tau_coefficients"], dtype=float),
            tau_terms=best_result["tau_terms"],
            v_coefficients=np.asarray(best_result["V_coefficients"], dtype=float),
            v_terms=best_result["V_terms"],
            scaling=best_result["scaling"],
        )
        rows.append({"event_id": str(cycle_df["event_id"].iloc[0]), **metrics})

        rel_time = scaled_df["time"].to_numpy(dtype=float) - float(scaled_df["time"].iloc[0])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(rel_time, scaled_df["tau"], label="true", linewidth=0.9)
        axes[0].plot(rel_time, tau_pred, label="pred", linewidth=0.9)
        axes[0].set_ylabel("tau")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[1].plot(rel_time, scaled_df["V"], label="true", linewidth=0.9)
        axes[1].plot(rel_time, v_pred, label="pred", linewidth=0.9)
        axes[1].set_ylabel("V")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        axes[2].plot(rel_time[1 : 1 + len(metrics["error_series"])], metrics["error_series"], linewidth=0.9)
        axes[2].axhline(2.0, color="tab:red", linestyle="--", alpha=0.6)
        axes[2].set_ylabel("point error")
        axes[2].set_xlabel("time since segment start [s]")
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(f"{cycle_df['event_id'].iloc[0]} long rollout")
        fig.tight_layout()
        fig.savefig(LONG_ROLLOUT_DIR / f"{cycle_df['event_id'].iloc[0]}_long_rollout.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(RESULTS_DIR / "p5838_long_rollout_summary.csv", index=False)
    return long_df


def subset_cycle(cycle_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    fraction = float(rng.uniform(0.7, 0.95))
    keep = max(int(len(cycle_df) * fraction), min(len(cycle_df), 600))
    start = int(rng.integers(0, max(len(cycle_df) - keep + 1, 1)))
    subset = cycle_df.iloc[start : start + keep].reset_index(drop=True).copy()
    subset.insert(0, "subset_id", f"{cycle_df['event_id'].iloc[0]}__{start}_{keep}")
    return subset


def coefficient_subset_stability(selected_cycles: list[pd.DataFrame], best_result: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SUBSET_SEED)
    coefficient_rows: list[dict] = []
    for subset_index in range(SUBSET_RUNS):
        subset_cycles = [subset_cycle(cycle_df, rng) for cycle_df in selected_cycles]
        result = fit_dynamic_configuration(
            selected_cycles=subset_cycles,
            smoothing_name=best_result["smoothing"],
            derivative_method=best_result["derivative_method"],
            threshold=best_result["threshold"],
            lags=tuple(best_result["lags"]),
        )
        for equation_name, terms_key, coefficients_key in (
            ("tau", "tau_terms", "tau_coefficients"),
            ("V", "V_terms", "V_coefficients"),
        ):
            for term, coefficient in zip(result[terms_key], result[coefficients_key]):
                coefficient_rows.append({"subset_run": subset_index, "equation": equation_name, "term": term, "coefficient": float(coefficient)})
    coefficient_df = pd.DataFrame(coefficient_rows)
    summary_df = (
        coefficient_df.groupby(["equation", "term"], as_index=False)
        .agg(coefficient_mean=("coefficient", "mean"), coefficient_std=("coefficient", "std"))
        .reset_index(drop=True)
    )
    summary_df.to_csv(RESULTS_DIR / "p5838_coefficient_subset_stability.csv", index=False)
    return summary_df


def write_refinement_report(
    delay_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    rsfit_df: pd.DataFrame,
    physical_equations: dict,
    long_rollout_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    best_result: dict,
) -> None:
    payload = {
        "best_single_delay": flatten_result(best_result),
        "delay_sweep_rows": delay_df.to_dict(orient="records"),
        "multi_delay_rows": multi_df.to_dict(orient="records"),
        "rsfit_validation_rows": rsfit_df.to_dict(orient="records"),
        "physical_equations": physical_equations,
        "long_rollout_rows": long_rollout_df.to_dict(orient="records"),
        "coefficient_stability_rows": subset_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "p5838_refinement_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Utah FORGE p5838 refinement report",
        "",
        "## Best single-delay model",
        f"- Lags: `{','.join(str(lag) for lag in best_result['lags'])}`",
        f"- Smoothing: `{best_result['smoothing']}`",
        f"- Derivative: `{best_result['derivative_method']}`",
        f"- Threshold: `{best_result['threshold']}`",
        f"- Stable across training cycles: `{best_result['stable_all_cycles']}`",
        f"- Rollout error: `{best_result['rollout_error']:.4f}`",
        "",
        "## Physical-unit equations",
        f"- `{physical_equations['tau_equation_physical']}`",
        f"- `{physical_equations['V_equation_physical']}`",
        "",
        "## RSFit validation",
        f"- Mean theta correlation: `{rsfit_df['theta_correlation'].mean():.4f}`",
        f"- Mean max cross-correlation: `{rsfit_df['max_abs_cross_correlation'].mean():.4f}`",
        "",
        "## Long unseen rollouts",
        f"- Mean unseen rollout error: `{long_rollout_df['rollout_error'].replace(np.inf, np.nan).mean():.4f}`",
        f"- Mean divergence time: `{long_rollout_df['divergence_time_s'].mean():.4f}` s",
    ]
    (RESULTS_DIR / "p5838_refinement_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    state_df, _ = base.load_p5838_state()
    event_df, selected_cycles, candidates = select_training_cycles(state_df)
    delay_df, best_by_delta = single_delay_sweep(selected_cycles)
    plot_delay_sweep(delay_df)
    best_single_delay = choose_best_result(list(best_by_delta.values()))
    multi_df, _ = multi_delay_sweep(selected_cycles, anchor_result=best_single_delay)
    rsfit_df = validate_rsfit_theta(selected_cycles, best_single_delay)
    physical_equations = denormalize_equations(best_single_delay)
    unseen_cycles = select_unseen_long_cycles(candidates, set(event_df["event_id"]), state_df)
    long_rollout_df = evaluate_long_rollouts(unseen_cycles, best_single_delay)
    subset_df = coefficient_subset_stability(selected_cycles, best_single_delay)
    write_refinement_report(delay_df, multi_df, rsfit_df, physical_equations, long_rollout_df, subset_df, best_single_delay)

    print(
        json.dumps(
            {
                "best_single_delay": int(best_single_delay["lags"][0]),
                "best_rollout_error": float(best_single_delay["rollout_error"]),
                "best_tau_equation_physical": physical_equations["tau_equation_physical"],
                "best_V_equation_physical": physical_equations["V_equation_physical"],
                "mean_theta_correlation": float(rsfit_df["theta_correlation"].mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
