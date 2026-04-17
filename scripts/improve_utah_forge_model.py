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
from scipy.linalg import lstsq
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.derivatives import derivative_savgol, derivative_spline
from src.io.utah_forge import load_utah_forge_dataset
from src.preprocess.common import smooth_series
from src.preprocess.utah_forge import build_utah_forge_state
from src.utils.paths import ensure_directory


RAW_FILE = REPO_ROOT / "data" / "utah_forge" / "p5838_datatable.mat"
RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
DERIVATIVE_DIR = RESULTS_DIR / "p5838_derivative_diagnostics"
ROLLOUT_DIR = RESULTS_DIR / "p5838_delay_rollouts"
EVENT_DIR = RESULTS_DIR / "p5838_selected_cycles"

SIGNAL_WINDOWS = {"light": 31, "moderate": 61, "strong": 101}
DERIVATIVE_METHODS = ("savgol", "spline")
DELTAS = (1, 2, 3, 4, 5)
THRESHOLDS = (0.01, 0.02, 0.05, 0.1)
MODEL_MAX_POINTS = 3_000
MIN_EVENT_POINTS = 1_200
MAX_SELECTED_EVENTS = 3
TAU_PEAK_WINDOW = 301
V_EVENT_WINDOW = 101
PEAK_DISTANCE = 2_000
RIDGE_ALPHA = 1e-6


@dataclass(frozen=True)
class EventCandidate:
    event_id: str
    start_idx: int
    end_idx: int
    peak_idx: int
    trough_idx: int
    tau_drop: float
    duration_s: float
    positive_fraction: float
    velocity_range: float
    velocity_noise_ratio: float
    score: float


def ensure_layout() -> None:
    ensure_directory(RESULTS_DIR)
    ensure_directory(DERIVATIVE_DIR)
    ensure_directory(ROLLOUT_DIR)
    ensure_directory(EVENT_DIR)


def enforce_monotonic_time(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 1:
        return df.reset_index(drop=True).copy()
    time = df["time"].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    keep[1:] = np.diff(time) > 0
    return df.loc[keep].reset_index(drop=True).copy()


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True).copy()


def downsample_frame(df: pd.DataFrame, max_points: int = MODEL_MAX_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True).copy()
    indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return df.iloc[indices].reset_index(drop=True).copy()


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    split = np.where(np.diff(indices) > 1)[0]
    starts = np.concatenate(([indices[0]], indices[split + 1]))
    ends = np.concatenate((indices[split] + 1, [indices[-1] + 1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _safe_quantile(values: np.ndarray, q: float, fallback: float) -> float:
    if len(values) == 0:
        return fallback
    return float(np.quantile(values, q))


def load_p5838_state() -> tuple[pd.DataFrame, dict]:
    raw_df, summary = load_utah_forge_dataset(RAW_FILE)
    state_df, metadata = build_utah_forge_state(raw_df, summary["column_mapping"])
    state_df = enforce_monotonic_time(remove_invalid_rows(state_df))
    run_summary = {
        "raw_file": str(RAW_FILE),
        "n_rows": int(len(state_df)),
        "columns": list(state_df.columns),
        "time_start": float(state_df["time"].iloc[0]),
        "time_end": float(state_df["time"].iloc[-1]),
        "tau_mean": float(state_df["tau"].mean()),
        "tau_std": float(state_df["tau"].std()),
        "V_min": float(state_df["V"].min()),
        "V_max": float(state_df["V"].max()),
        "V_nonpositive_count": int((state_df["V"] <= 0).sum()),
        "column_mapping": summary["column_mapping"],
        "preserved_columns": metadata["preserved_columns"],
    }
    return state_df, run_summary


def detect_clean_cycles(state_df: pd.DataFrame) -> list[EventCandidate]:
    tau = state_df["tau"].to_numpy(dtype=float)
    velocity = state_df["V"].to_numpy(dtype=float)
    time = state_df["time"].to_numpy(dtype=float)

    tau_smoothed = smooth_series(tau, window=TAU_PEAK_WINDOW, polyorder=3)
    velocity_smoothed = smooth_series(velocity, window=V_EVENT_WINDOW, polyorder=3)

    tau_std = float(np.std(tau_smoothed))
    peaks, _ = find_peaks(tau_smoothed, prominence=max(0.12 * tau_std, 0.1), distance=PEAK_DISTANCE)
    troughs, _ = find_peaks(-tau_smoothed, prominence=max(0.12 * tau_std, 0.1), distance=PEAK_DISTANCE)

    candidates: list[EventCandidate] = []
    used_ranges: list[tuple[int, int]] = []
    for event_index, peak_idx in enumerate(peaks):
        future_troughs = troughs[troughs > peak_idx]
        if len(future_troughs) == 0:
            continue
        trough_idx = int(future_troughs[0])
        previous_troughs = troughs[troughs < peak_idx]
        next_peaks = peaks[peaks > trough_idx]
        previous_trough_idx = int(previous_troughs[-1]) if len(previous_troughs) else 0
        next_peak_idx = int(next_peaks[0]) if len(next_peaks) else len(state_df) - 1

        tau_drop = float(tau_smoothed[peak_idx] - tau_smoothed[trough_idx])
        if tau_drop < max(0.2 * tau_std, 0.25):
            continue

        search_start = max(previous_trough_idx, peak_idx - 20_000)
        search_end = min(next_peak_idx, trough_idx + 20_000)
        local_velocity = velocity_smoothed[search_start:search_end]
        positive_velocity = local_velocity[local_velocity > 0]
        if len(positive_velocity) < 100:
            continue

        high_threshold = max(
            _safe_quantile(positive_velocity, 0.70, 0.0),
            float(np.median(positive_velocity) + 0.25 * np.std(positive_velocity)),
            0.25,
        )
        low_threshold = max(0.35 * high_threshold, _safe_quantile(positive_velocity, 0.30, 0.1), 0.05)
        burst_runs = contiguous_runs(local_velocity > high_threshold)
        if not burst_runs:
            continue

        reference_slice_start = max(search_start, peak_idx - 2_000)
        reference_slice_end = min(search_end, trough_idx + 2_000)
        if reference_slice_end - reference_slice_start < 10:
            continue
        reference_idx = int(reference_slice_start + np.argmax(velocity_smoothed[reference_slice_start:reference_slice_end]))
        scored_runs: list[tuple[float, tuple[int, int]]] = []
        for run_start, run_end in burst_runs:
            global_start = search_start + run_start
            global_end = search_start + run_end
            midpoint = 0.5 * (global_start + global_end)
            area = float(np.trapezoid(np.clip(velocity_smoothed[global_start:global_end], 0.0, None)))
            distance_penalty = abs(midpoint - reference_idx)
            scored_runs.append((area - 0.05 * distance_penalty, (global_start, global_end)))
        run_start, run_end = max(scored_runs, key=lambda item: item[0])[1]

        start_idx = run_start
        while start_idx > search_start and velocity_smoothed[start_idx - 1] > low_threshold:
            start_idx -= 1
        end_idx = run_end
        while end_idx < search_end and velocity_smoothed[end_idx] > low_threshold:
            end_idx += 1

        start_idx = max(previous_trough_idx, start_idx - 250)
        end_idx = min(next_peak_idx, end_idx + 250)
        if end_idx - start_idx < MIN_EVENT_POINTS:
            continue

        segment_velocity = velocity_smoothed[start_idx:end_idx]
        segment_raw_velocity = velocity[start_idx:end_idx]
        positive_fraction = float(np.mean(segment_velocity > 0))
        if positive_fraction < 0.93:
            continue

        duration_s = float(time[end_idx - 1] - time[start_idx])
        velocity_range = float(np.max(segment_velocity) - np.min(segment_velocity))
        velocity_noise_ratio = float(np.std(segment_raw_velocity - segment_velocity) / (np.std(segment_velocity) + 1e-12))
        score = float(
            0.45 * (tau_drop / (tau_std + 1e-12))
            + 0.30 * positive_fraction
            + 0.15 * np.clip(np.log10(max(duration_s, 1e-3)) / 2.0, 0.0, 1.5)
            + 0.10 * np.clip(velocity_range / (np.std(velocity_smoothed) + 1e-12), 0.0, 6.0)
            - 0.10 * velocity_noise_ratio
        )

        overlap = False
        for used_start, used_end in used_ranges:
            if not (end_idx <= used_start or start_idx >= used_end):
                overlap = True
                break
        if overlap:
            continue

        used_ranges.append((start_idx, end_idx))
        candidates.append(
            EventCandidate(
                event_id=f"p5838_event_{event_index:03d}",
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                peak_idx=int(peak_idx),
                trough_idx=int(trough_idx),
                tau_drop=tau_drop,
                duration_s=duration_s,
                positive_fraction=positive_fraction,
                velocity_range=velocity_range,
                velocity_noise_ratio=velocity_noise_ratio,
                score=score,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def save_event_inventory(state_df: pd.DataFrame, candidates: list[EventCandidate]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict] = []
    cycle_frames: list[pd.DataFrame] = []
    short_candidates = [candidate for candidate in candidates if candidate.duration_s <= 5.0]
    chosen_candidates = short_candidates[:MAX_SELECTED_EVENTS] if len(short_candidates) >= MAX_SELECTED_EVENTS else candidates[:MAX_SELECTED_EVENTS]
    for candidate in chosen_candidates:
        cycle = state_df.iloc[candidate.start_idx : candidate.end_idx].reset_index(drop=True).copy()
        cycle.insert(0, "event_id", candidate.event_id)
        cycle.to_csv(EVENT_DIR / f"{candidate.event_id}.csv", index=False)
        cycle_frames.append(cycle)
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
    event_df.to_csv(RESULTS_DIR / "p5838_physics_informed_event_summary.csv", index=False)
    return event_df, cycle_frames


def prepare_cycle(cycle_df: pd.DataFrame, smoothing_name: str, delta: int) -> tuple[pd.DataFrame, dict]:
    window = SIGNAL_WINDOWS[smoothing_name]
    working = cycle_df[["time", "tau", "V"]].copy()
    working = downsample_frame(working, max_points=MODEL_MAX_POINTS)
    working = enforce_monotonic_time(working)

    tau_smooth = smooth_series(working["tau"].to_numpy(dtype=float), window=window, polyorder=3)
    v_smooth = smooth_series(working["V"].to_numpy(dtype=float), window=window, polyorder=3)
    positive_values = v_smooth[v_smooth > 0]
    velocity_floor = max(_safe_quantile(positive_values, 0.01, 1e-3) * 0.25, 1e-3)
    v_positive = np.clip(v_smooth, velocity_floor, None)
    log_v = np.log(v_positive)

    tau_mean = float(np.mean(tau_smooth))
    tau_std = float(np.std(tau_smooth))
    if tau_std == 0 or not np.isfinite(tau_std):
        tau_std = 1.0
    log_v_mean = float(np.mean(log_v))
    log_v_std = float(np.std(log_v))
    if log_v_std == 0 or not np.isfinite(log_v_std):
        log_v_std = 1.0

    prepared = pd.DataFrame(
        {
            "time": working["time"].to_numpy(dtype=float),
            "tau": tau_smooth,
            "V": v_positive,
            "logV": log_v,
            "tau_z": (tau_smooth - tau_mean) / tau_std,
            "logV_z": (log_v - log_v_mean) / log_v_std,
        }
    )
    prepared["tau_lag"] = prepared["tau"].shift(delta)
    prepared["V_lag"] = prepared["V"].shift(delta)
    prepared["tau_lag_z"] = prepared["tau_z"].shift(delta)
    prepared["logV_lag_z"] = prepared["logV_z"].shift(delta)
    prepared = remove_invalid_rows(prepared)
    metadata = {
        "smoothing": smoothing_name,
        "signal_window": window,
        "delta": int(delta),
        "velocity_floor": float(velocity_floor),
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "logV_mean": log_v_mean,
        "logV_std": log_v_std,
    }
    return prepared.reset_index(drop=True), metadata


def estimate_derivatives(prepared_df: pd.DataFrame, method: str, smoothing_name: str) -> tuple[np.ndarray, np.ndarray]:
    time = prepared_df["time"].to_numpy(dtype=float)
    tau = prepared_df["tau"].to_numpy(dtype=float)
    velocity = prepared_df["V"].to_numpy(dtype=float)
    if method == "savgol":
        window = max(SIGNAL_WINDOWS[smoothing_name], 31)
        tau_dot = derivative_savgol(tau, t=time, window=window, polyorder=3)
        velocity_dot = derivative_savgol(velocity, t=time, window=window, polyorder=3)
    elif method == "spline":
        tau_dot = derivative_spline(tau, t=time)
        velocity_dot = derivative_spline(velocity, t=time)
    else:
        raise ValueError(f"Unsupported derivative method: {method}")
    return tau_dot, velocity_dot


def save_derivative_diagnostics(selected_cycles: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for cycle_df in selected_cycles:
        event_id = str(cycle_df["event_id"].iloc[0])
        fig, axes = plt.subplots(len(SIGNAL_WINDOWS), 4, figsize=(18, 10), sharex="col")
        if len(SIGNAL_WINDOWS) == 1:
            axes = np.array([axes])
        for row_index, smoothing_name in enumerate(SIGNAL_WINDOWS):
            prepared, metadata = prepare_cycle(cycle_df, smoothing_name=smoothing_name, delta=1)
            time = prepared["time"].to_numpy(dtype=float) - float(prepared["time"].iloc[0])
            for method_index, method in enumerate(DERIVATIVE_METHODS):
                tau_dot, velocity_dot = estimate_derivatives(prepared, method=method, smoothing_name=smoothing_name)
                rows.append(
                    {
                        "event_id": event_id,
                        "smoothing": smoothing_name,
                        "method": method,
                        "signal_window": metadata["signal_window"],
                        "tau_std": float(np.std(prepared["tau"])),
                        "V_std": float(np.std(prepared["V"])),
                        "dtau_std": float(np.std(tau_dot)),
                        "dV_std": float(np.std(velocity_dot)),
                        "dV_roughness": float(np.std(np.diff(velocity_dot)) / (np.std(velocity_dot) + 1e-12)),
                    }
                )

                line_width = 0.9 if method_index == 0 else 0.7
                alpha = 0.95 if method_index == 0 else 0.75
                label = method
                axes[row_index, 0].plot(time, prepared["tau"], linewidth=line_width, alpha=alpha, label=label)
                axes[row_index, 1].plot(time, prepared["V"], linewidth=line_width, alpha=alpha, label=label)
                axes[row_index, 2].plot(time, tau_dot, linewidth=line_width, alpha=alpha, label=label)
                axes[row_index, 3].plot(time, velocity_dot, linewidth=line_width, alpha=alpha, label=label)

            axes[row_index, 0].set_ylabel(smoothing_name)
            for col in range(4):
                axes[row_index, col].grid(True, alpha=0.3)
            axes[row_index, 0].legend(loc="upper right", fontsize=8)
        axes[0, 0].set_title("tau")
        axes[0, 1].set_title("V (positive clipped)")
        axes[0, 2].set_title("dtau/dt")
        axes[0, 3].set_title("dV/dt")
        for col in range(4):
            axes[-1, col].set_xlabel("time since cycle start [s]")
        fig.suptitle(f"{event_id} derivative comparison", y=0.995)
        fig.tight_layout()
        fig.savefig(DERIVATIVE_DIR / f"{event_id}_derivative_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df.to_csv(RESULTS_DIR / "p5838_derivative_method_comparison.csv", index=False)
    return diagnostics_df


def build_libraries(prepared_df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    tau_library = np.column_stack(
        [
            np.ones(len(prepared_df)),
            prepared_df["tau_z"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
        ]
    )
    tau_terms = ["1", "tau_z", "V"]

    v_library = np.column_stack(
        [
            prepared_df["tau_z"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["logV_z"].to_numpy(dtype=float),
            prepared_df["tau_lag_z"].to_numpy(dtype=float),
        ]
    )
    v_terms = ["tau_z", "V", "logV_z", "tau_lag_z"]
    return tau_library, tau_terms, v_library, v_terms


def fit_sparse_equation(
    library: np.ndarray,
    target: np.ndarray,
    terms: list[str],
    threshold: float,
    mandatory_terms: set[str] | None = None,
) -> tuple[np.ndarray, float]:
    mandatory_terms = mandatory_terms or set()
    mandatory_mask = np.array([term in mandatory_terms for term in terms], dtype=bool)
    column_scales = np.linalg.norm(library, axis=0)
    column_scales[column_scales == 0] = 1.0
    scaled_library = library / column_scales
    active = np.ones(len(terms), dtype=bool)

    for _ in range(20):
        active_library = scaled_library[:, active]
        if active_library.size == 0:
            break
        lhs = np.vstack([active_library, math.sqrt(RIDGE_ALPHA) * np.eye(active_library.shape[1])])
        rhs = np.concatenate([target, np.zeros(active_library.shape[1])])
        coeff_active, _, _, _ = lstsq(lhs, rhs)
        coeff_scaled = np.zeros(len(terms))
        coeff_scaled[active] = coeff_active
        small = np.abs(coeff_scaled) < threshold
        small[mandatory_mask] = False
        updated_active = ~small
        if np.array_equal(updated_active, active):
            active = updated_active
            break
        active = updated_active

    active_library = scaled_library[:, active]
    lhs = np.vstack([active_library, math.sqrt(RIDGE_ALPHA) * np.eye(active_library.shape[1])])
    rhs = np.concatenate([target, np.zeros(active_library.shape[1])])
    coeff_active, _, _, _ = lstsq(lhs, rhs)
    coeff_scaled = np.zeros(len(terms))
    coeff_scaled[active] = coeff_active
    coefficients = coeff_scaled / column_scales
    prediction = library @ coefficients
    residual = float(np.linalg.norm(target - prediction) / (np.linalg.norm(target) + 1e-12))
    return coefficients, residual


def active_terms(coefficients: np.ndarray, terms: list[str], tolerance: float = 1e-12) -> list[str]:
    return [term for term, coefficient in zip(terms, coefficients) if abs(coefficient) > tolerance]


def equation_string(name: str, coefficients: np.ndarray, terms: list[str]) -> str:
    pieces: list[str] = []
    for term, coefficient in zip(terms, coefficients):
        if abs(coefficient) <= 1e-12:
            continue
        sign = "+" if coefficient >= 0 else "-"
        piece = f"{sign} {abs(coefficient):.3e}*{term}"
        pieces.append(piece if pieces else piece.lstrip("+ ").strip())
    if not pieces:
        return f"d{name}/dt = 0"
    return f"d{name}/dt = " + " ".join(pieces)


def rollout_cycle(
    prepared_df: pd.DataFrame,
    delta: int,
    tau_coefficients: np.ndarray,
    tau_terms: list[str],
    v_coefficients: np.ndarray,
    v_terms: list[str],
    tau_mean: float,
    tau_std: float,
    log_v_mean: float,
    log_v_std: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    time = prepared_df["time"].to_numpy(dtype=float)
    tau_true = prepared_df["tau"].to_numpy(dtype=float)
    v_true = prepared_df["V"].to_numpy(dtype=float)
    velocity_floor = float(np.min(v_true))
    tau_std = tau_std or 1.0
    log_v_std = log_v_std or 1.0

    tau_pred = tau_true.copy()
    v_pred = v_true.copy()
    max_tau_allowed = 5.0 * max(float(np.max(np.abs(tau_true))), 1.0)
    max_v_allowed = 5.0 * max(float(np.max(np.abs(v_true))), 1.0)
    stable = True
    for index in range(delta, len(prepared_df) - 1):
        dt = float(time[index + 1] - time[index])
        current_tau = tau_pred[index]
        current_v = max(float(v_pred[index]), velocity_floor)
        current_tau_z = (current_tau - tau_mean) / tau_std
        current_log_v_z = (math.log(current_v) - log_v_mean) / log_v_std
        lag_tau = tau_pred[index - delta]
        lag_v = max(float(v_pred[index - delta]), velocity_floor)
        lag_tau_z = (lag_tau - tau_mean) / tau_std

        tau_features = {
            "1": 1.0,
            "tau_z": current_tau_z,
            "V": current_v,
            "tau_lag_z": lag_tau_z,
            "V_lag": lag_v,
        }
        v_features = {
            "tau_z": current_tau_z,
            "V": current_v,
            "logV_z": current_log_v_z,
            "tau_z*logV_z": current_tau_z * current_log_v_z,
            "tau_z*V": current_tau_z * current_v,
            "tau_lag_z": lag_tau_z,
            "V_lag": lag_v,
        }
        tau_dot = float(sum(coefficient * tau_features[term] for coefficient, term in zip(tau_coefficients, tau_terms)))
        v_dot = float(sum(coefficient * v_features[term] for coefficient, term in zip(v_coefficients, v_terms)))

        tau_pred[index + 1] = current_tau + dt * tau_dot
        v_pred[index + 1] = max(current_v + dt * v_dot, velocity_floor)
        if (
            not np.isfinite(tau_pred[index + 1])
            or not np.isfinite(v_pred[index + 1])
            or abs(tau_pred[index + 1]) > max_tau_allowed
            or abs(v_pred[index + 1]) > max_v_allowed
        ):
            stable = False
            tau_pred[index + 1 :] = np.nan
            v_pred[index + 1 :] = np.nan
            break

    if np.isfinite(tau_pred).all() and np.isfinite(v_pred).all() and stable:
        tau_rmse = float(np.sqrt(np.mean((tau_pred - tau_true) ** 2)))
        v_rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
        tau_rel = float(np.linalg.norm(tau_pred - tau_true) / (np.linalg.norm(tau_true) + 1e-12))
        v_rel = float(np.linalg.norm(v_pred - v_true) / (np.linalg.norm(v_true) + 1e-12))
    else:
        tau_rmse = float("inf")
        v_rmse = float("inf")
        tau_rel = float("inf")
        v_rel = float("inf")
        stable = False
    metrics = {
        "tau_rmse": tau_rmse,
        "V_rmse": v_rmse,
        "tau_relative_error": tau_rel,
        "V_relative_error": v_rel,
        "combined_relative_error": 0.5 * (tau_rel + v_rel),
        "stable": stable,
    }
    return tau_pred, v_pred, metrics


def fit_configuration(selected_cycles: list[pd.DataFrame], smoothing_name: str, derivative_method: str, delta: int, threshold: float) -> dict:
    prepared_cycles: list[pd.DataFrame] = []
    cycle_metadata: list[dict] = []
    for cycle_df in selected_cycles:
        prepared_df, metadata = prepare_cycle(cycle_df, smoothing_name=smoothing_name, delta=delta)
        prepared_cycles.append(prepared_df)
        cycle_metadata.append(
            {
                "event_id": str(cycle_df["event_id"].iloc[0]),
                "n_points": int(len(prepared_df)),
                **metadata,
            }
        )

    global_tau = np.concatenate([prepared_df["tau"].to_numpy(dtype=float) for prepared_df in prepared_cycles])
    global_log_v = np.concatenate([prepared_df["logV"].to_numpy(dtype=float) for prepared_df in prepared_cycles])
    global_tau_mean = float(np.mean(global_tau))
    global_tau_std = float(np.std(global_tau))
    if global_tau_std == 0 or not np.isfinite(global_tau_std):
        global_tau_std = 1.0
    global_log_v_mean = float(np.mean(global_log_v))
    global_log_v_std = float(np.std(global_log_v))
    if global_log_v_std == 0 or not np.isfinite(global_log_v_std):
        global_log_v_std = 1.0

    tau_target_parts: list[np.ndarray] = []
    v_target_parts: list[np.ndarray] = []
    tau_library_parts: list[np.ndarray] = []
    v_library_parts: list[np.ndarray] = []
    scaled_cycles: list[pd.DataFrame] = []
    tau_terms: list[str] | None = None
    v_terms: list[str] | None = None
    for prepared_df in prepared_cycles:
        scaled_df = prepared_df.copy()
        scaled_df["tau_z"] = (scaled_df["tau"] - global_tau_mean) / global_tau_std
        scaled_df["logV_z"] = (scaled_df["logV"] - global_log_v_mean) / global_log_v_std
        scaled_df["tau_lag_z"] = (scaled_df["tau_lag"] - global_tau_mean) / global_tau_std
        scaled_cycles.append(scaled_df)

        tau_dot, v_dot = estimate_derivatives(scaled_df, method=derivative_method, smoothing_name=smoothing_name)
        tau_library, cycle_tau_terms, v_library, cycle_v_terms = build_libraries(scaled_df)
        tau_terms = cycle_tau_terms
        v_terms = cycle_v_terms
        tau_target_parts.append(tau_dot)
        v_target_parts.append(v_dot)
        tau_library_parts.append(tau_library)
        v_library_parts.append(v_library)

    tau_target = np.concatenate(tau_target_parts)
    v_target = np.concatenate(v_target_parts)
    tau_library = np.vstack(tau_library_parts)
    v_library = np.vstack(v_library_parts)

    tau_coefficients, tau_residual = fit_sparse_equation(
        tau_library,
        tau_target,
        tau_terms,
        threshold=threshold,
        mandatory_terms={"V"},
    )
    v_coefficients, v_residual = fit_sparse_equation(
        v_library,
        v_target,
        v_terms,
        threshold=threshold,
        mandatory_terms={"tau_z"},
    )

    tau_active = active_terms(tau_coefficients, tau_terms)
    v_active = active_terms(v_coefficients, v_terms)
    has_tau_v_coupling = "V" in tau_active
    has_v_tau_coupling = "tau_z" in v_active
    has_hidden_state_proxy = any(term in v_active for term in ("logV_z", "tau_lag_z", "V_lag"))

    rollout_rows: list[dict] = []
    stable_all = True
    for prepared_df, metadata in zip(scaled_cycles, cycle_metadata):
        tau_pred, v_pred, rollout_metrics = rollout_cycle(
            prepared_df,
            delta=delta,
            tau_coefficients=tau_coefficients,
            tau_terms=tau_terms,
            v_coefficients=v_coefficients,
            v_terms=v_terms,
            tau_mean=global_tau_mean,
            tau_std=global_tau_std,
            log_v_mean=global_log_v_mean,
            log_v_std=global_log_v_std,
        )
        stable_all = stable_all and rollout_metrics["stable"]
        rollout_rows.append({"event_id": metadata["event_id"], **rollout_metrics, "tau_prediction": tau_pred, "V_prediction": v_pred})

    rollout_error = float(np.mean([row["combined_relative_error"] for row in rollout_rows]))
    tau_error = float(np.mean([row["tau_relative_error"] for row in rollout_rows]))
    v_error = float(np.mean([row["V_relative_error"] for row in rollout_rows]))
    if not np.isfinite(rollout_error):
        rollout_error = float("inf")
    if not np.isfinite(tau_error):
        tau_error = float("inf")
    if not np.isfinite(v_error):
        v_error = float("inf")
    total_terms = len(tau_active) + len(v_active)
    physical_valid = bool(has_tau_v_coupling and has_v_tau_coupling and has_hidden_state_proxy and stable_all)
    if np.isfinite(rollout_error):
        physical_score = (
            3.0 * float(has_tau_v_coupling)
            + 3.0 * float(has_v_tau_coupling)
            + 2.0 * float(has_hidden_state_proxy)
            + 2.0 * float(stable_all)
            - 2.0 * rollout_error
            - 0.15 * total_terms
        )
    else:
        physical_score = -1e6

    return {
        "smoothing": smoothing_name,
        "derivative_method": derivative_method,
        "delta": int(delta),
        "threshold": float(threshold),
        "tau_terms_active": tau_active,
        "V_terms_active": v_active,
        "tau_equation": equation_string("tau", tau_coefficients, tau_terms),
        "V_equation": equation_string("V", v_coefficients, v_terms),
        "tau_coefficients": tau_coefficients.tolist(),
        "V_coefficients": v_coefficients.tolist(),
        "tau_terms": tau_terms,
        "V_terms": v_terms,
        "tau_residual": float(tau_residual),
        "V_residual": float(v_residual),
        "tau_rollout_error": tau_error,
        "V_rollout_error": v_error,
        "rollout_error": rollout_error,
        "stable_all_cycles": stable_all,
        "has_tau_v_coupling": has_tau_v_coupling,
        "has_v_tau_coupling": has_v_tau_coupling,
        "has_hidden_state_proxy": has_hidden_state_proxy,
        "physical_valid": physical_valid,
        "physical_score": float(physical_score),
        "total_terms": int(total_terms),
        "global_tau_mean": global_tau_mean,
        "global_tau_std": global_tau_std,
        "global_logV_mean": global_log_v_mean,
        "global_logV_std": global_log_v_std,
        "prepared_cycles": cycle_metadata,
        "rollouts": rollout_rows,
    }


def save_rollout_plots(best_result: dict, selected_cycles: list[pd.DataFrame]) -> None:
    for cycle_df in selected_cycles:
        event_id = str(cycle_df["event_id"].iloc[0])
        prepared_df, _ = prepare_cycle(cycle_df, smoothing_name=best_result["smoothing"], delta=int(best_result["delta"]))
        matching_rollout = next(row for row in best_result["rollouts"] if row["event_id"] == event_id)
        tau_pred = np.asarray(matching_rollout["tau_prediction"], dtype=float)
        v_pred = np.asarray(matching_rollout["V_prediction"], dtype=float)
        time = prepared_df["time"].to_numpy(dtype=float) - float(prepared_df["time"].iloc[0])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(time, prepared_df["tau"], label="true", linewidth=1.0)
        axes[0].plot(time, tau_pred, label="rollout", linewidth=0.9, alpha=0.85)
        axes[0].set_ylabel("tau")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(time, prepared_df["V"], label="true", linewidth=1.0)
        axes[1].plot(time, v_pred, label="rollout", linewidth=0.9, alpha=0.85)
        axes[1].set_ylabel("V")
        axes[1].set_xlabel("time since cycle start [s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        fig.suptitle(
            f"{event_id} rollout | {best_result['derivative_method']} | {best_result['smoothing']} | delta={best_result['delta']}"
        )
        fig.tight_layout()
        fig.savefig(ROLLOUT_DIR / f"{event_id}_rollout.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_best_model_artifacts(best_result: dict, event_df: pd.DataFrame, diagnostics_df: pd.DataFrame, load_summary: dict) -> None:
    coefficient_rows: list[dict] = []
    for equation_name, terms_key, coefficients_key in (
        ("tau", "tau_terms", "tau_coefficients"),
        ("V", "V_terms", "V_coefficients"),
    ):
        for term, coefficient in zip(best_result[terms_key], best_result[coefficients_key]):
            coefficient_rows.append(
                {
                    "equation": equation_name,
                    "term": term,
                    "coefficient": float(coefficient),
                    "active": bool(abs(coefficient) > 1e-12),
                }
            )
    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_df.to_csv(RESULTS_DIR / "p5838_physics_informed_coefficients_best.csv", index=False)

    equations_text = "\n".join(
        [
            "Best physics-informed delay-embedded model for Utah FORGE p5838",
            f"smoothing={best_result['smoothing']}",
            f"derivative_method={best_result['derivative_method']}",
            f"delta={best_result['delta']}",
            f"threshold={best_result['threshold']}",
            "",
            best_result["tau_equation"],
            best_result["V_equation"],
        ]
    )
    (RESULTS_DIR / "p5838_physics_informed_equations.txt").write_text(equations_text, encoding="utf-8")

    summary_payload = {
        "dataset": "utah_forge",
        "experiment_id": "p5838",
        "raw_summary": load_summary,
        "selected_cycles": event_df.to_dict(orient="records"),
        "best_model": {
            key: value
            for key, value in best_result.items()
            if key not in {"rollouts", "tau_coefficients", "V_coefficients"}
        },
        "success": bool(best_result["physical_valid"]),
        "success_criteria": {
            "tau_dot_includes_V": bool(best_result["has_tau_v_coupling"]),
            "V_dot_includes_tau": bool(best_result["has_v_tau_coupling"]),
            "V_dot_includes_log_or_delay": bool(best_result["has_hidden_state_proxy"]),
            "stable_rollout_multiple_cycles": bool(best_result["stable_all_cycles"]),
        },
        "derivative_diagnostics_rows": int(len(diagnostics_df)),
        "artifacts": {
            "equations": str(RESULTS_DIR / "p5838_physics_informed_equations.txt"),
            "coefficients": str(RESULTS_DIR / "p5838_physics_informed_coefficients_best.csv"),
            "diagnostics": str(RESULTS_DIR / "p5838_derivative_method_comparison.csv"),
            "rollouts_dir": str(ROLLOUT_DIR),
        },
    }
    (RESULTS_DIR / "p5838_physics_informed_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Utah FORGE p5838 physics-informed SINDy report",
        "",
        "## Best configuration",
        f"- Smoothing: `{best_result['smoothing']}`",
        f"- Derivative method: `{best_result['derivative_method']}`",
        f"- Delay delta: `{best_result['delta']}` samples",
        f"- Sparsity threshold: `{best_result['threshold']}`",
        "",
        "## Recovered equations",
        f"- `{best_result['tau_equation']}`",
        f"- `{best_result['V_equation']}`",
        "",
        "## Success criteria",
        f"- `tau_dot` includes `V`: `{best_result['has_tau_v_coupling']}`",
        f"- `V_dot` includes `tau`: `{best_result['has_v_tau_coupling']}`",
        f"- `V_dot` includes `log(V)` or delayed terms: `{best_result['has_hidden_state_proxy']}`",
        f"- Stable rollout across selected cycles: `{best_result['stable_all_cycles']}`",
        f"- Pipeline successful under requested criteria: `{best_result['physical_valid']}`",
        "",
        "## Errors",
        f"- `tau` residual: `{best_result['tau_residual']:.4f}`",
        f"- `V` residual: `{best_result['V_residual']:.4f}`",
        f"- Mean rollout relative error: `{best_result['rollout_error']:.4f}`",
        "",
        "## Notes",
        "- `tau_z` and `logV_z` denote zero-mean, unit-variance transformed variables used for conditioning.",
        "- Delay terms are sample delays over the downsampled cycle grid and serve as hidden-state surrogates for missing rate-and-state memory.",
        "- Rollouts are evaluated on multiple selected clean cycles using the same fitted model.",
    ]
    (RESULTS_DIR / "p5838_physics_informed_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    state_df, load_summary = load_p5838_state()
    candidates = detect_clean_cycles(state_df)
    if not candidates:
        raise RuntimeError("No clean p5838 stick-slip cycles were detected with positive-slip windows.")

    event_df, selected_cycles = save_event_inventory(state_df, candidates)
    if len(selected_cycles) < 2:
        raise RuntimeError("At least two clean p5838 cycles are required for multi-cycle stability assessment.")

    diagnostics_df = save_derivative_diagnostics(selected_cycles)

    model_rows: list[dict] = []
    best_result: dict | None = None
    for smoothing_name in SIGNAL_WINDOWS:
        for derivative_method in DERIVATIVE_METHODS:
            for delta in DELTAS:
                for threshold in THRESHOLDS:
                    result = fit_configuration(
                        selected_cycles=selected_cycles,
                        smoothing_name=smoothing_name,
                        derivative_method=derivative_method,
                        delta=delta,
                        threshold=threshold,
                    )
                    model_rows.append(
                        {
                            "smoothing": result["smoothing"],
                            "derivative_method": result["derivative_method"],
                            "delta": result["delta"],
                            "threshold": result["threshold"],
                            "tau_terms_active": "|".join(result["tau_terms_active"]),
                            "V_terms_active": "|".join(result["V_terms_active"]),
                            "tau_residual": result["tau_residual"],
                            "V_residual": result["V_residual"],
                            "tau_rollout_error": result["tau_rollout_error"],
                            "V_rollout_error": result["V_rollout_error"],
                            "rollout_error": result["rollout_error"],
                            "stable_all_cycles": result["stable_all_cycles"],
                            "has_tau_v_coupling": result["has_tau_v_coupling"],
                            "has_v_tau_coupling": result["has_v_tau_coupling"],
                            "has_hidden_state_proxy": result["has_hidden_state_proxy"],
                            "physical_valid": result["physical_valid"],
                            "physical_score": result["physical_score"],
                            "total_terms": result["total_terms"],
                            "tau_equation": result["tau_equation"],
                            "V_equation": result["V_equation"],
                        }
                    )
                    if best_result is None:
                        best_result = result
                        continue
                    current_key = (
                        int(result["physical_valid"]),
                        int(result["stable_all_cycles"]),
                        result["physical_score"],
                        -result["rollout_error"],
                        -result["total_terms"],
                    )
                    best_key = (
                        int(best_result["physical_valid"]),
                        int(best_result["stable_all_cycles"]),
                        best_result["physical_score"],
                        -best_result["rollout_error"],
                        -best_result["total_terms"],
                    )
                    if current_key > best_key:
                        best_result = result

    sweep_df = pd.DataFrame(model_rows).sort_values(
        ["physical_valid", "stable_all_cycles", "physical_score", "rollout_error", "total_terms"],
        ascending=[False, False, False, True, True],
    )
    sweep_df.to_csv(RESULTS_DIR / "p5838_physics_informed_model_results.csv", index=False)

    if best_result is None:
        raise RuntimeError("No model results were produced for p5838.")

    save_rollout_plots(best_result, selected_cycles)
    save_best_model_artifacts(best_result, event_df, diagnostics_df, load_summary)

    print(
        json.dumps(
            {
                "selected_event_ids": list(event_df["event_id"]),
                "best_physical_valid": bool(best_result["physical_valid"]),
                "best_smoothing": best_result["smoothing"],
                "best_derivative_method": best_result["derivative_method"],
                "best_delta": int(best_result["delta"]),
                "best_threshold": float(best_result["threshold"]),
                "best_tau_equation": best_result["tau_equation"],
                "best_V_equation": best_result["V_equation"],
                "best_rollout_error": float(best_result["rollout_error"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
