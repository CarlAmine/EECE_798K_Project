from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import UTAH_FORGE_CONFIG
from src.derivatives import derivative_savgol, estimate_derivatives_df
from src.io.utah_forge import load_utah_forge_dataset
from src.preprocess.common import scale_columns, smooth_series
from src.preprocess.utah_forge import build_utah_forge_state
from src.sindy import SINDyModel, build_polynomial_library, compute_rollout_metrics, rollout_polynomial
from src.utils.paths import ensure_directory


EXPECTED_EXPERIMENT_IDS = ("p5838", "p5848", "p5897", "p5905", "p5912")
STATE_COLUMNS = ["tau", "V"]
DIAGNOSTIC_MAX_POINTS = 4_000
MODEL_MAX_POINTS = 2_500
PROMINENCE_FACTOR = 0.15
PEAK_DISTANCE = 2_000
SMOOTHING_LEVELS = {"raw": None, "light": 31, "strong": 151}
DERIVATIVE_METHODS = ("central", "savgol", "spline")
SCALING_METHODS = ("none", "zscore", "robust")
LIBRARY_DEGREES = (1, 2, 3)
THRESHOLD_GRID = {
    "none": (1e-6, 1e-4, 1e-2),
    "zscore": (1e-4, 1e-3, 1e-2),
    "robust": (1e-4, 1e-3, 1e-2),
}


@dataclass(frozen=True)
class EventSelection:
    label: str
    experiment_id: str
    event_id: str
    csv_path: Path
    score: float


def event_cycle_path(label: str) -> Path:
    return UTAH_FORGE_CONFIG.results_dir / f"selected_cycle_{label}.csv"


def ensure_results_layout() -> dict[str, Path]:
    results_dir = ensure_directory(UTAH_FORGE_CONFIG.results_dir)
    derivative_dir = ensure_directory(results_dir / "derivative_diagnostics")
    rollout_dir = ensure_directory(results_dir / "top_model_rollouts")
    return {
        "results_dir": results_dir,
        "derivative_dir": derivative_dir,
        "rollout_dir": rollout_dir,
    }


def make_experiment_row(experiment_id: str, status: str, raw_file: str | None = None) -> dict:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "raw_file": raw_file,
        "event_id": None,
        "event_index": np.nan,
        "event_start_idx": np.nan,
        "event_end_idx": np.nan,
        "peak_idx": np.nan,
        "trough_idx": np.nan,
        "time_start": np.nan,
        "time_end": np.nan,
        "peak_time": np.nan,
        "trough_time": np.nan,
        "duration_s": np.nan,
        "n_samples": np.nan,
        "tau_drop": np.nan,
        "tau_drop_over_dataset_std": np.nan,
        "tau_peak": np.nan,
        "tau_trough": np.nan,
        "tau_recovery": np.nan,
        "tau_gradient_peak": np.nan,
        "V_std": np.nan,
        "V_range": np.nan,
        "velocity_consistency_corr": np.nan,
        "velocity_noise_ratio": np.nan,
        "gap_fraction": np.nan,
        "missing_fraction": np.nan,
        "dt_median": np.nan,
        "dataset_n_samples": np.nan,
        "dataset_duration_s": np.nan,
        "dataset_tau_mean": np.nan,
        "dataset_tau_std": np.nan,
        "dataset_V_mean": np.nan,
        "dataset_V_std": np.nan,
        "event_quality_score": np.nan,
    }


def enforce_monotonic_time(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 1:
        return df.copy()
    time = df["time"].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    keep[1:] = np.diff(time) > 0
    return df.loc[keep].reset_index(drop=True).copy()


def downsample_event(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True).copy()
    indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return df.iloc[indices].reset_index(drop=True).copy()


def smooth_event(df: pd.DataFrame, smoothing_name: str) -> pd.DataFrame:
    result = df.copy()
    window = SMOOTHING_LEVELS[smoothing_name]
    if window is None:
        return result
    for column in STATE_COLUMNS:
        result[column] = smooth_series(result[column].to_numpy(dtype=float), window=window, polyorder=3)
    return result


def compute_velocity_from_displacement(df: pd.DataFrame, window: int = 151) -> np.ndarray:
    displacement = df["displacement"].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)
    return derivative_savgol(displacement, t=time, window=window, polyorder=3) * 1_000.0


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rank_0_1(values: pd.Series) -> pd.Series:
    if values.notna().sum() <= 1:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return values.rank(method="average", pct=True).fillna(0.0)


def compute_event_quality_scores(events_df: pd.DataFrame) -> pd.DataFrame:
    result = events_df.copy()
    available_mask = result["status"] == "available_event"
    available = result.loc[available_mask].copy()
    if available.empty:
        result["event_quality_score"] = np.nan
        return result

    available["tau_drop_score"] = rank_0_1(available["tau_drop"])
    available["velocity_score"] = rank_0_1(available["V_std"])
    available["duration_score"] = rank_0_1(np.log10(available["duration_s"].clip(lower=1e-6)))
    available["consistency_score"] = ((available["velocity_consistency_corr"].clip(-1.0, 1.0) + 1.0) / 2.0).fillna(0.0)
    available["sharpness_score"] = rank_0_1(available["tau_gradient_peak"])
    available["noise_penalty"] = rank_0_1(available["velocity_noise_ratio"])
    available["gap_penalty"] = rank_0_1(available["gap_fraction"])
    available["event_quality_score"] = (
        0.35 * available["tau_drop_score"]
        + 0.18 * available["consistency_score"]
        + 0.16 * available["velocity_score"]
        + 0.15 * available["duration_score"]
        + 0.12 * available["sharpness_score"]
        - 0.03 * available["noise_penalty"]
        - 0.01 * available["gap_penalty"]
    )
    result.loc[available.index, "event_quality_score"] = available["event_quality_score"]
    return result


def detect_candidate_events(experiment_id: str, state_df: pd.DataFrame) -> list[dict]:
    tau = state_df["tau"].to_numpy(dtype=float)
    time = state_df["time"].to_numpy(dtype=float)
    dataset_tau_std = float(np.std(tau))
    tau_smoothed = smooth_series(tau, window=301, polyorder=3)
    prominence = PROMINENCE_FACTOR * dataset_tau_std
    peaks, _ = find_peaks(tau_smoothed, prominence=prominence, distance=PEAK_DISTANCE)
    troughs, _ = find_peaks(-tau_smoothed, prominence=prominence, distance=PEAK_DISTANCE)
    velocity_from_displacement = compute_velocity_from_displacement(state_df)

    candidates: list[dict] = []
    for event_index, peak_idx in enumerate(peaks):
        future_troughs = troughs[troughs > peak_idx]
        if len(future_troughs) == 0:
            continue
        trough_idx = int(future_troughs[0])
        tau_drop = float(tau_smoothed[peak_idx] - tau_smoothed[trough_idx])
        if tau_drop <= 0.2 * dataset_tau_std:
            continue

        previous_troughs = troughs[troughs < peak_idx]
        next_peaks = peaks[peaks > trough_idx]
        previous_trough_idx = int(previous_troughs[-1]) if len(previous_troughs) else 0
        next_peak_idx = int(next_peaks[0]) if len(next_peaks) else len(state_df) - 1

        drop_len = max(peak_idx - previous_trough_idx, trough_idx - peak_idx, 1_500)
        pre_len = int(min(max(drop_len, 1_500), peak_idx - previous_trough_idx if previous_trough_idx < peak_idx else drop_len))
        post_len = int(min(max(drop_len, 1_500), next_peak_idx - trough_idx if trough_idx < next_peak_idx else drop_len))
        start_idx = max(0, peak_idx - pre_len)
        end_idx = min(len(state_df), trough_idx + post_len)
        if end_idx - start_idx < 2_500:
            continue

        event_df = state_df.iloc[start_idx:end_idx].reset_index(drop=True)
        dt = np.diff(event_df["time"].to_numpy(dtype=float))
        dt_median = float(np.median(dt)) if len(dt) else float("nan")
        gap_fraction = float(np.mean(dt > 5.0 * dt_median)) if len(dt) and dt_median > 0 else 0.0

        event_V = event_df["V"].to_numpy(dtype=float)
        event_V_smoothed = smooth_series(event_V, window=151, polyorder=3)
        event_V_from_d = velocity_from_displacement[start_idx:end_idx]
        event_V_from_d_smoothed = smooth_series(event_V_from_d, window=151, polyorder=3)

        tau_gradient_peak = float(
            np.nanmax(np.abs(np.gradient(tau_smoothed[start_idx:end_idx], event_df["time"].to_numpy(dtype=float))))
        )
        tau_recovery = float(tau_smoothed[end_idx - 1] - tau_smoothed[trough_idx])
        velocity_noise_ratio = float(np.std(event_V - event_V_smoothed) / (np.std(event_V_smoothed) + 1e-12))

        candidate = make_experiment_row(experiment_id, status="available_event", raw_file=None)
        candidate.update(
            {
                "event_id": f"{experiment_id}_event_{event_index:03d}",
                "event_index": int(event_index),
                "event_start_idx": int(start_idx),
                "event_end_idx": int(end_idx),
                "peak_idx": int(peak_idx),
                "trough_idx": int(trough_idx),
                "time_start": float(time[start_idx]),
                "time_end": float(time[end_idx - 1]),
                "peak_time": float(time[peak_idx]),
                "trough_time": float(time[trough_idx]),
                "duration_s": float(time[end_idx - 1] - time[start_idx]),
                "n_samples": int(end_idx - start_idx),
                "tau_drop": tau_drop,
                "tau_drop_over_dataset_std": float(tau_drop / (dataset_tau_std + 1e-12)),
                "tau_peak": float(tau_smoothed[peak_idx]),
                "tau_trough": float(tau_smoothed[trough_idx]),
                "tau_recovery": tau_recovery,
                "tau_gradient_peak": tau_gradient_peak,
                "V_std": float(np.std(event_V_smoothed)),
                "V_range": float(np.max(event_V_smoothed) - np.min(event_V_smoothed)),
                "velocity_consistency_corr": correlation(event_V_smoothed, event_V_from_d_smoothed),
                "velocity_noise_ratio": velocity_noise_ratio,
                "gap_fraction": gap_fraction,
                "missing_fraction": 0.0,
                "dt_median": dt_median,
            }
        )
        candidates.append(candidate)

    return candidates


def select_best_events(events_df: pd.DataFrame) -> dict[str, pd.Series]:
    available = events_df.loc[events_df["status"] == "available_event"].copy()
    if available.empty:
        return {}
    available = available.sort_values(["event_quality_score", "tau_drop"], ascending=[False, False]).reset_index(drop=True)
    q1 = float(available["duration_s"].quantile(1.0 / 3.0))
    q2 = float(available["duration_s"].quantile(2.0 / 3.0))
    duration_targets = {"short": q1, "medium": float(available["duration_s"].median()), "long": q2}
    selected: dict[str, pd.Series] = {}
    used_event_ids: set[str] = set()
    duration_masks = {
        "short": available["duration_s"] <= q1,
        "medium": (available["duration_s"] > q1) & (available["duration_s"] <= q2),
        "long": available["duration_s"] > q2,
    }
    for label, target in duration_targets.items():
        candidates = available.loc[duration_masks[label] & ~available["event_id"].isin(used_event_ids)].copy()
        if candidates.empty:
            candidates = available.loc[~available["event_id"].isin(used_event_ids)].copy()
        if candidates.empty:
            break
        candidates["duration_distance"] = np.abs(candidates["duration_s"] - target)
        candidates["selection_score"] = candidates["event_quality_score"] - 0.15 * rank_0_1(candidates["duration_distance"])
        winner = candidates.sort_values(["selection_score", "event_quality_score"], ascending=[False, False]).iloc[0]
        selected[label] = winner
        used_event_ids.add(str(winner["event_id"]))
    return selected


def save_selected_cycles(selected_rows: dict[str, pd.Series], state_frames: dict[str, pd.DataFrame]) -> list[EventSelection]:
    saved: list[EventSelection] = []
    for label, row in selected_rows.items():
        experiment_id = str(row["experiment_id"])
        cycle_df = state_frames[experiment_id].iloc[int(row["event_start_idx"]) : int(row["event_end_idx"])].reset_index(drop=True).copy()
        cycle_df.insert(0, "event_id", str(row["event_id"]))
        cycle_df.insert(0, "experiment_id", experiment_id)
        csv_path = event_cycle_path(label)
        cycle_df.to_csv(csv_path, index=False)
        saved.append(EventSelection(label, experiment_id, str(row["event_id"]), csv_path, float(row["event_quality_score"])))
    return saved


def derivative_window_for_level(level: str) -> int:
    if level == "strong":
        return 151
    if level == "light":
        return 31
    return 31


def estimate_state_derivatives(df: pd.DataFrame, method: str, smoothing_name: str) -> tuple[dict[str, np.ndarray], dict]:
    derivative_window = derivative_window_for_level(smoothing_name)
    derivatives, _ = estimate_derivatives_df(
        df[["time", *STATE_COLUMNS]].copy(),
        time_col="time",
        var_cols=STATE_COLUMNS,
        method=method,
        window=derivative_window,
        polyorder=3,
        add_to_df=False,
    )
    summary = {
        "method": method,
        "smoothing": smoothing_name,
        "derivative_window": derivative_window,
        "dtau_std": float(np.std(derivatives["tau"])),
        "dV_std": float(np.std(derivatives["V"])),
        "dV_roughness": float(np.std(np.diff(derivatives["V"])) / (np.std(derivatives["V"]) + 1e-12)),
    }
    return derivatives, summary


def plot_derivative_diagnostics(event_name: str, event_df: pd.DataFrame, out_dir: Path) -> list[dict]:
    diagnostics_rows: list[dict] = []
    plot_df = downsample_event(event_df, DIAGNOSTIC_MAX_POINTS)
    for method in DERIVATIVE_METHODS:
        fig, axes = plt.subplots(len(SMOOTHING_LEVELS), 4, figsize=(18, 11), sharex="col")
        if len(SMOOTHING_LEVELS) == 1:
            axes = np.array([axes])
        for row_index, (smoothing_name, _) in enumerate(SMOOTHING_LEVELS.items()):
            smoothed = smooth_event(plot_df, smoothing_name)
            derivatives, summary = estimate_state_derivatives(smoothed, method=method, smoothing_name=smoothing_name)
            diagnostics_rows.append({"event_name": event_name, "derivative_method": method, "smoothing": smoothing_name, **summary})

            time = smoothed["time"].to_numpy(dtype=float) - float(smoothed["time"].iloc[0])
            axes[row_index, 0].plot(time, smoothed["tau"], linewidth=0.8)
            axes[row_index, 1].plot(time, smoothed["V"], linewidth=0.8)
            axes[row_index, 2].plot(time, derivatives["tau"], linewidth=0.8)
            axes[row_index, 3].plot(time, derivatives["V"], linewidth=0.8)
            axes[row_index, 0].set_ylabel(smoothing_name)
            for col in range(4):
                axes[row_index, col].grid(True, alpha=0.3)
            axes[row_index, 2].set_title(f"dTau/dt\nstd={summary['dtau_std']:.2e}")
            axes[row_index, 3].set_title(f"dV/dt\nrough={summary['dV_roughness']:.2f}")

        axes[0, 0].set_title("tau")
        axes[0, 1].set_title("V")
        for col in range(4):
            axes[-1, col].set_xlabel("time since event start [s]")
        fig.suptitle(f"{event_name} derivative diagnostics - {method}", y=0.995)
        fig.tight_layout()
        fig.savefig(out_dir / f"{event_name}__{method}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    return diagnostics_rows


def library_descriptions_for_degree(degree: int) -> list[str]:
    _, descriptions = build_polynomial_library(np.zeros((1, 2), dtype=float), degree=degree, var_names=STATE_COLUMNS)
    return descriptions


def affine_library_transform(scale_meta: dict, degree: int) -> tuple[np.ndarray, list[str]]:
    centers = np.array([scale_meta["centers"][column] for column in STATE_COLUMNS], dtype=float)
    scales = np.array([scale_meta["scales"][column] for column in STATE_COLUMNS], dtype=float)
    grid = np.array(list(product([-2.0, -1.0, 0.0, 1.0, 2.0], repeat=2)), dtype=float)
    original_samples = centers + grid * scales
    scaled_samples = (original_samples - centers) / scales
    original_library, descriptions = build_polynomial_library(original_samples, degree=degree, var_names=STATE_COLUMNS)
    scaled_library, _ = build_polynomial_library(scaled_samples, degree=degree, var_names=STATE_COLUMNS)
    transform, _, _, _ = np.linalg.lstsq(original_library, scaled_library, rcond=None)
    return transform, descriptions


def coefficients_to_original_units(coefficients: np.ndarray, scale_meta: dict, degree: int) -> tuple[np.ndarray, list[str]]:
    if scale_meta["method"] == "none":
        return coefficients.copy(), library_descriptions_for_degree(degree)
    transform, descriptions = affine_library_transform(scale_meta, degree)
    output_scales = np.array([scale_meta["scales"][column] for column in STATE_COLUMNS], dtype=float)
    original_coefficients = np.zeros_like(coefficients)
    for variable_index in range(coefficients.shape[1]):
        original_coefficients[:, variable_index] = transform @ (output_scales[variable_index] * coefficients[:, variable_index])
    return original_coefficients, descriptions


def equations_from_coefficients(coefficients: np.ndarray, descriptions: list[str]) -> list[str]:
    model = SINDyModel()
    model.coefficients = coefficients
    model.library_descriptions = descriptions
    return model.equations(STATE_COLUMNS)


def predicted_derivative_std(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    return float(np.std(predicted_values) / (float(np.std(true_values)) + 1e-12))


def fit_sindy_candidate(
    event_name: str,
    event_df: pd.DataFrame,
    smoothing_name: str,
    derivative_method: str,
    scaling_name: str,
    degree: int,
    threshold: float,
) -> dict:
    modeling_df = downsample_event(event_df, MODEL_MAX_POINTS)
    modeling_df = enforce_monotonic_time(modeling_df)
    smoothed_df = smooth_event(modeling_df, smoothing_name)

    scaled_df = smoothed_df[["time", *STATE_COLUMNS]].copy()
    scaled_df, scale_meta = scale_columns(scaled_df, STATE_COLUMNS, method=scaling_name)
    derivatives, derivative_summary = estimate_state_derivatives(scaled_df, method=derivative_method, smoothing_name=smoothing_name)

    X = scaled_df[STATE_COLUMNS].to_numpy(dtype=float)
    Xdot = np.column_stack([derivatives[column] for column in STATE_COLUMNS])
    library, scaled_descriptions = build_polynomial_library(X, degree=degree, var_names=STATE_COLUMNS)
    condition_number = float(np.linalg.cond(library))

    model = SINDyModel(threshold=threshold, max_iter=15)
    diagnostics = model.fit(library, Xdot, scaled_descriptions)
    Xdot_pred = model.predict(library)
    sparse_terms = (np.abs(model.coefficients) > 0).sum(axis=0)
    collapse_tau = bool(sparse_terms[0] <= 1 or predicted_derivative_std(Xdot[:, 0], Xdot_pred[:, 0]) < 0.05)
    collapse_V = bool(sparse_terms[1] <= 1 or predicted_derivative_std(Xdot[:, 1], Xdot_pred[:, 1]) < 0.05)

    original_coefficients, original_descriptions = coefficients_to_original_units(model.coefficients, scale_meta, degree)
    equations = equations_from_coefficients(original_coefficients, original_descriptions)

    rollout_time = smoothed_df["time"].to_numpy(dtype=float)
    rollout_time = rollout_time - rollout_time[0]
    rollout_truth = smoothed_df[STATE_COLUMNS].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rollout_prediction = rollout_polynomial(
                original_coefficients,
                original_descriptions,
                rollout_truth[0],
                rollout_time,
                {"state_1": STATE_COLUMNS[0], "state_2": STATE_COLUMNS[1]},
            )
        except Exception:
            rollout_prediction = np.full_like(rollout_truth, np.nan)

    if np.isfinite(rollout_prediction).all():
        rollout_metrics = compute_rollout_metrics(rollout_truth, rollout_prediction)
    else:
        rollout_metrics = {"rmse": float("inf"), "mae": float("inf"), "relative_error": float("inf")}

    total_sparse_terms = int(diagnostics["total_sparse_terms"])
    interpretable = bool(not collapse_tau and not collapse_V and total_sparse_terms <= 8 and rollout_metrics["relative_error"] < 1.5)
    model_score = float(
        np.mean(diagnostics["residuals"])
        + rollout_metrics["relative_error"]
        + 0.05 * total_sparse_terms
        + 0.2 * int(collapse_tau)
        + 0.2 * int(collapse_V)
        + 0.01 * math.log10(max(condition_number, 1.0))
    )

    return {
        "event_name": event_name,
        "smoothing": smoothing_name,
        "derivative_method": derivative_method,
        "scaling": scaling_name,
        "library_degree": int(degree),
        "threshold": float(threshold),
        "library_condition": condition_number,
        "active_terms_tau": int(sparse_terms[0]),
        "active_terms_V": int(sparse_terms[1]),
        "total_active_terms": total_sparse_terms,
        "residual_tau": float(diagnostics["residuals"][0]),
        "residual_V": float(diagnostics["residuals"][1]),
        "rollout_rmse": float(rollout_metrics["rmse"]),
        "rollout_mae": float(rollout_metrics["mae"]),
        "rollout_relative_error": float(rollout_metrics["relative_error"]),
        "collapse_tau": collapse_tau,
        "collapse_V": collapse_V,
        "interpretable": interpretable,
        "model_score": model_score,
        "equation_tau": equations[0],
        "equation_V": equations[1],
        "coefficients_original": original_coefficients.tolist(),
        "library_terms_original": original_descriptions,
        "time_start": float(smoothed_df["time"].iloc[0]),
        "time_end": float(smoothed_df["time"].iloc[-1]),
        "n_model_points": int(len(smoothed_df)),
        **derivative_summary,
        "rollout_prediction": rollout_prediction.tolist() if np.isfinite(rollout_prediction).all() else None,
    }


def select_top_models(sweep_df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    finite = sweep_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_score", "rollout_relative_error"])
    if finite.empty:
        return finite
    preferred = finite.loc[finite["interpretable"]].copy()
    if preferred.empty:
        preferred = finite.copy()
    preferred = preferred.sort_values(["model_score", "rollout_relative_error", "total_active_terms"], ascending=[True, True, True])
    preferred = preferred.drop_duplicates(
        subset=[
            "selected_cycle_label",
            "smoothing",
            "derivative_method",
            "scaling",
            "library_degree",
            "equation_tau",
            "equation_V",
        ]
    )
    return preferred.head(limit)


def make_rollout_plot(event_name: str, event_df: pd.DataFrame, row: pd.Series, out_path: Path) -> None:
    truth_df = downsample_event(event_df, MODEL_MAX_POINTS)
    truth_df = enforce_monotonic_time(truth_df)
    truth_df = smooth_event(truth_df, str(row["smoothing"]))
    prediction = np.asarray(row["rollout_prediction"], dtype=float)
    if prediction.shape != (len(truth_df), len(STATE_COLUMNS)):
        return

    time = truth_df["time"].to_numpy(dtype=float) - float(truth_df["time"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, column, values in zip(axes, STATE_COLUMNS, prediction.T):
        axis.plot(time, truth_df[column].to_numpy(dtype=float), label=f"true {column}", linewidth=1.0)
        axis.plot(time, values, label=f"pred {column}", linewidth=1.0, linestyle="--")
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[0].set_title(
        f"{event_name} rollout | degree={row['library_degree']} | {row['derivative_method']} | {row['scaling']} | thr={row['threshold']:.1e}"
    )
    axes[-1].set_xlabel("time since event start [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_best_artifacts(best_row: pd.Series, out_dir: Path) -> None:
    (out_dir / "discovered_equations_best.txt").write_text(
        f"{best_row['equation_tau']}\n{best_row['equation_V']}\n",
        encoding="utf-8",
    )
    summary = {
        "experiment_id": best_row["experiment_id"],
        "event_name": best_row["event_name"],
        "selected_cycle_label": best_row["selected_cycle_label"],
        "smoothing": best_row["smoothing"],
        "derivative_method": best_row["derivative_method"],
        "scaling": best_row["scaling"],
        "library_degree": int(best_row["library_degree"]),
        "threshold": float(best_row["threshold"]),
        "active_terms_tau": int(best_row["active_terms_tau"]),
        "active_terms_V": int(best_row["active_terms_V"]),
        "residual_tau": float(best_row["residual_tau"]),
        "residual_V": float(best_row["residual_V"]),
        "rollout_rmse": float(best_row["rollout_rmse"]),
        "rollout_relative_error": float(best_row["rollout_relative_error"]),
        "collapse_tau": bool(best_row["collapse_tau"]),
        "collapse_V": bool(best_row["collapse_V"]),
        "interpretable": bool(best_row["interpretable"]),
        "equations": [best_row["equation_tau"], best_row["equation_V"]],
        "library_terms_original": best_row["library_terms_original"],
    }
    (out_dir / "best_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def report_strength(best_row: pd.Series) -> str:
    if bool(best_row["interpretable"]) and float(best_row["rollout_relative_error"]) < 0.75:
        return "now scientifically interpretable at baseline level"
    if bool(best_row["interpretable"]):
        return "improved but still somewhat weak"
    return "still weak"


def next_bottleneck(best_row: pd.Series) -> str:
    if float(best_row["rollout_relative_error"]) > 1.0:
        return "hidden variable theta"
    if bool(best_row["collapse_tau"]) or bool(best_row["collapse_V"]):
        return "event selection"
    if int(best_row["library_degree"]) >= 3 and not bool(best_row["interpretable"]):
        return "noise / apparatus effects"
    return "missing nonlinear terms"


def write_report(
    out_dir: Path,
    available_experiments: list[str],
    missing_experiments: list[str],
    selections: dict[str, pd.Series],
    best_row: pd.Series,
) -> None:
    selection_lines = []
    for label in ("short", "medium", "long"):
        row = selections.get(label)
        if row is None:
            continue
        selection_lines.append(
            f"- `{label}`: `{row['event_id']}` from `{row['experiment_id']}` (duration {row['duration_s']:.2f} s, score {row['event_quality_score']:.3f})"
        )

    report = "\n".join(
        [
            "# Utah FORGE model improvement report",
            "",
            "## Data coverage",
            f"- Available local experiments: {', '.join(available_experiments) if available_experiments else 'none'}",
            f"- Missing expected experiments: {', '.join(missing_experiments) if missing_experiments else 'none'}",
            "",
            "## Selected modeling events",
            *selection_lines,
            "",
            "## Best current model",
            f"- Best experiment: `{best_row['experiment_id']}`",
            f"- Best cycle/event: `{best_row['event_name']}` (`{best_row['selected_cycle_label']}`)",
            f"- Preprocessing choice: `{best_row['smoothing']}` smoothing",
            f"- Derivative method: `{best_row['derivative_method']}`",
            f"- Scaling choice: `{best_row['scaling']}`",
            f"- Library degree: `{int(best_row['library_degree'])}`",
            f"- Sparsity threshold: `{float(best_row['threshold']):.1e}`",
            "- Final discovered equations:",
            f"  - `{best_row['equation_tau']}`",
            f"  - `{best_row['equation_V']}`",
            f"- Interpretation quality: {report_strength(best_row)}",
            f"- Next bottleneck: {next_bottleneck(best_row)}",
            "",
            "## Notes",
            "- The current local Utah FORGE folder contains only `p5838_datatable.mat`, so the cross-experiment comparison is limited to the locally available raw file set.",
            "- The best equations are written in physical `tau` and `V` coordinates after converting the fitted scaled model back into original units.",
        ]
    )
    (out_dir / "model_improvement_report.md").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    layout = ensure_results_layout()
    for old_plot in layout["rollout_dir"].glob("*.png"):
        old_plot.unlink()
    event_rows: list[dict] = []
    state_frames: dict[str, pd.DataFrame] = {}
    available_experiments: list[str] = []
    missing_experiments: list[str] = []

    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        file_path = UTAH_FORGE_CONFIG.raw_dir / f"{experiment_id}_datatable.mat"
        if not file_path.exists():
            event_rows.append(make_experiment_row(experiment_id, status="missing_local_file"))
            missing_experiments.append(experiment_id)
            continue

        raw_df, load_summary = load_utah_forge_dataset(file_path)
        state_df, _ = build_utah_forge_state(raw_df, load_summary["column_mapping"])
        state_df = enforce_monotonic_time(state_df)
        state_frames[experiment_id] = state_df
        available_experiments.append(experiment_id)

        dataset_row = make_experiment_row(experiment_id, status="available_experiment", raw_file=str(file_path))
        dataset_row.update(
            {
                "dataset_n_samples": int(len(state_df)),
                "dataset_duration_s": float(state_df["time"].iloc[-1] - state_df["time"].iloc[0]),
                "dataset_tau_mean": float(state_df["tau"].mean()),
                "dataset_tau_std": float(state_df["tau"].std()),
                "dataset_V_mean": float(state_df["V"].mean()),
                "dataset_V_std": float(state_df["V"].std()),
            }
        )
        event_rows.append(dataset_row)
        for candidate in detect_candidate_events(experiment_id, state_df):
            candidate["raw_file"] = str(file_path)
            candidate["dataset_n_samples"] = int(len(state_df))
            candidate["dataset_duration_s"] = float(state_df["time"].iloc[-1] - state_df["time"].iloc[0])
            candidate["dataset_tau_mean"] = float(state_df["tau"].mean())
            candidate["dataset_tau_std"] = float(state_df["tau"].std())
            candidate["dataset_V_mean"] = float(state_df["V"].mean())
            candidate["dataset_V_std"] = float(state_df["V"].std())
            event_rows.append(candidate)

    event_quality_df = pd.DataFrame(event_rows)
    event_quality_df = compute_event_quality_scores(event_quality_df)
    event_quality_df.sort_values(["status", "experiment_id", "event_quality_score", "tau_drop"], ascending=[True, True, False, False], inplace=True, na_position="last")
    event_quality_df.to_csv(layout["results_dir"] / "event_quality_summary.csv", index=False)

    selections = select_best_events(event_quality_df)
    saved_cycles = save_selected_cycles(selections, state_frames)
    selected_events_by_name = {selection.label: pd.read_csv(selection.csv_path) for selection in saved_cycles}

    derivative_rows: list[dict] = []
    for label, event_df in selected_events_by_name.items():
        derivative_rows.extend(plot_derivative_diagnostics(label, event_df, layout["derivative_dir"]))
    pd.DataFrame(derivative_rows).to_csv(layout["results_dir"] / "derivative_diagnostics_summary.csv", index=False)

    sweep_rows: list[dict] = []
    for label, event_df in selected_events_by_name.items():
        for smoothing_name, derivative_method, scaling_name, degree in product(
            SMOOTHING_LEVELS.keys(), DERIVATIVE_METHODS, SCALING_METHODS, LIBRARY_DEGREES
        ):
            for threshold in THRESHOLD_GRID[scaling_name]:
                row = fit_sindy_candidate(label, event_df, smoothing_name, derivative_method, scaling_name, degree, threshold)
                selected_row = selections[label]
                row["experiment_id"] = selected_row["experiment_id"]
                row["event_id"] = selected_row["event_id"]
                row["selected_cycle_label"] = label
                sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.sort_values(["model_score", "rollout_relative_error"], inplace=True)
    sweep_df.to_csv(layout["results_dir"] / "sindy_sweep_results.csv", index=False)

    top_models = select_top_models(sweep_df, limit=3)
    for rank, (_, row) in enumerate(top_models.iterrows(), start=1):
        out_path = layout["rollout_dir"] / (
            f"rank_{rank:02d}__{row['selected_cycle_label']}__{row['smoothing']}__{row['derivative_method']}__{row['scaling']}__deg{int(row['library_degree'])}.png"
        )
        make_rollout_plot(str(row["selected_cycle_label"]), selected_events_by_name[str(row["selected_cycle_label"])], row, out_path)

    if top_models.empty:
        raise RuntimeError("No Utah FORGE models completed successfully.")

    best_row = top_models.iloc[0]
    write_best_artifacts(best_row, layout["results_dir"])
    write_report(layout["results_dir"], available_experiments, missing_experiments, selections, best_row)


if __name__ == "__main__":
    main()
