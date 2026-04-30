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
from scipy.integrate import odeint
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_proposal_equation_recovery as proposal
from src.preprocess.common import smooth_series
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
CHECKPOINT_PATH = RESULTS_DIR / "proposal_equation_checkpoints" / "prepared_segments.pkl"
STEP_NAMES = [
    "p5838_step2",
    "p5838_step3",
    "p5838_step4",
    "p5838_step5",
    "p5838_step7",
    "p5838_step8",
    "p5838_step9",
    "p5838_step10",
]
CURRENT_HOLDOUT = ["p5838_step2", "p5838_step7"]
CURRENT_TRAIN = ["p5838_step3", "p5838_step4", "p5838_step5", "p5838_step8", "p5838_step9", "p5838_step10"]
MIN_STEP_ROWS = 1500
MIN_POSITIVE_FRACTION = 0.98
SMOOTHING_WINDOW = proposal.SMOOTHING_WINDOW
SMOOTHING_POLYORDER = proposal.SMOOTHING_POLYORDER
DERIV_WINDOW = proposal.DERIV_WINDOW
DERIV_POLYORDER = proposal.DERIV_POLYORDER
EPS = 1e-12


@dataclass
class StepArtifacts:
    step_name: str
    raw_segment: pd.DataFrame
    raw_downsampled: pd.DataFrame
    prepared: pd.DataFrame
    inclusion_row: dict


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return json_ready(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_ready(value.to_dict())
    return value


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    working = frame.copy()
    columns = [str(column) for column in working.columns]
    rows = [[str(value) for value in row] for row in working.astype(object).fillna("").to_numpy().tolist()]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body]) if body else "\n".join([header, divider])


def load_checkpoint_payload() -> dict:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Prepared-segment checkpoint not found: {CHECKPOINT_PATH}")
    return pd.read_pickle(CHECKPOINT_PATH)


def build_raw_segments() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    state_df, _ = base.load_p5838_state()
    steps = delay_ref.load_rsfit_steps()
    segments: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for step_name in STEP_NAMES:
        step = steps[step_name]
        mask = (state_df["time"] >= float(step.time[0])) & (state_df["time"] <= float(step.time[-1]))
        segment = state_df.loc[mask].reset_index(drop=True).copy()
        if segment.empty:
            continue
        v_drive = np.where(
            segment["time"].to_numpy(dtype=float) < float(step.params["TimeOfStep"]),
            float(step.params["InitialVelocity"]),
            float(step.params["FinalVelocity"]),
        )
        segment.insert(0, "step_name", step_name)
        segment["V_drive"] = v_drive
        segments[step_name] = segment
        rows.append(
            {
                "step_name": step_name,
                "time_start": float(segment["time"].iloc[0]),
                "time_end": float(segment["time"].iloc[-1]),
                "duration_s": float(segment["time"].iloc[-1] - segment["time"].iloc[0]),
                "n_rows_raw": int(len(segment)),
                "positive_fraction": float((segment["V"] > 0).mean()),
                "tau_std_raw": float(segment["tau"].std()),
                "step_r2": float(step.params["R2"]),
                "initial_velocity": float(step.params["InitialVelocity"]),
                "final_velocity": float(step.params["FinalVelocity"]),
            }
        )
    inventory = pd.DataFrame(rows).sort_values("time_start").reset_index(drop=True)
    eligible = inventory.loc[
        (inventory["n_rows_raw"] >= MIN_STEP_ROWS) & (inventory["positive_fraction"] >= MIN_POSITIVE_FRACTION)
    ].copy()
    eligible = eligible.sort_values(["duration_s", "n_rows_raw"], ascending=[False, False]).reset_index(drop=True)
    return state_df, segments, eligible


def build_step_artifacts() -> tuple[dict[str, StepArtifacts], pd.DataFrame]:
    payload = load_checkpoint_payload()
    outputs = payload["outputs"]
    inclusion_df = pd.DataFrame(payload["inclusion_rows"])
    prepared_map = {}
    for key in ("all_train", "all_holdout"):
        for df in outputs[key]:
            prepared_map[str(df["step_name"].iloc[0])] = df.copy()

    _, raw_segments, eligible = build_raw_segments()
    artifacts: dict[str, StepArtifacts] = {}
    for step_name in STEP_NAMES:
        raw_segment = raw_segments[step_name].copy()
        raw_downsampled = base.downsample_frame(raw_segment[["step_name", "time", "tau", "V", "V_drive"]], max_points=base.MODEL_MAX_POINTS)
        raw_downsampled = base.enforce_monotonic_time(raw_downsampled)
        inclusion_row = (
            inclusion_df.loc[inclusion_df["step_name"] == step_name].iloc[0].to_dict()
            if step_name in inclusion_df["step_name"].values
            else {"step_name": step_name}
        )
        artifacts[step_name] = StepArtifacts(
            step_name=step_name,
            raw_segment=raw_segment,
            raw_downsampled=raw_downsampled,
            prepared=prepared_map[step_name].copy(),
            inclusion_row=inclusion_row,
        )
    return artifacts, eligible


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def series_stats(time: np.ndarray, values: np.ndarray) -> dict:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {key: float("nan") for key in ("mean", "std", "variance", "min", "max", "range", "iqr", "total_variation", "avg_abs_derivative", "peak_to_peak", "dominant_frequency_hz")}
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    diffs = np.diff(values)
    dt = np.diff(time)
    finite_dt = dt[np.isfinite(dt) & (np.abs(dt) > EPS)]
    avg_abs_derivative = float(np.mean(np.abs(diffs / dt))) if len(diffs) and len(finite_dt) == len(dt) else float("nan")
    demeaned = values - float(np.mean(values))
    dominant_frequency = float("nan")
    if len(values) >= 8 and duration > EPS:
        uniform_time = np.linspace(0.0, duration, len(values))
        uniform_values = np.interp(uniform_time, time - time[0], values)
        spectrum = np.abs(np.fft.rfft(uniform_values - np.mean(uniform_values)))
        freqs = np.fft.rfftfreq(len(uniform_values), d=duration / max(len(uniform_values) - 1, 1))
        if len(spectrum) > 1:
            peak_index = 1 + int(np.argmax(spectrum[1:]))
            dominant_frequency = float(freqs[peak_index])
    prominence = max(0.05 * float(np.ptp(values)), 0.10 * float(np.std(values)), 1e-6)
    peaks, _ = find_peaks(values, prominence=prominence)
    troughs, _ = find_peaks(-values, prominence=prominence)
    oscillation_count = int(min(len(peaks), len(troughs)))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "total_variation": float(np.sum(np.abs(diffs))),
        "avg_abs_derivative": avg_abs_derivative,
        "peak_to_peak": float(np.ptp(values)),
        "dominant_frequency_hz": dominant_frequency,
        "rough_oscillation_count": oscillation_count,
        "duration_s": duration,
        "sample_count": int(len(values)),
    }


def derive_raw_downsampled_dtau(raw_downsampled: pd.DataFrame) -> np.ndarray:
    time = raw_downsampled["time"].to_numpy(dtype=float)
    tau = raw_downsampled["tau"].to_numpy(dtype=float)
    return proposal.derivative_savgol(tau, t=time, window=DERIV_WINDOW, polyorder=DERIV_POLYORDER)


def metric_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if not np.isfinite(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def compute_step_metrics(artifacts: dict[str, StepArtifacts]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    for step_name, artifact in artifacts.items():
        raw_time = artifact.raw_downsampled["time"].to_numpy(dtype=float)
        raw_tau = artifact.raw_downsampled["tau"].to_numpy(dtype=float)
        raw_v = artifact.raw_downsampled["V"].to_numpy(dtype=float)
        raw_gap = artifact.raw_downsampled["V_drive"].to_numpy(dtype=float) - raw_v
        raw_dtau = derive_raw_downsampled_dtau(artifact.raw_downsampled)

        prep_time = artifact.prepared["time"].to_numpy(dtype=float)
        prep_tau = artifact.prepared["tau"].to_numpy(dtype=float)
        prep_v = artifact.prepared["V"].to_numpy(dtype=float)
        prep_gap = artifact.prepared["V_drive"].to_numpy(dtype=float) - prep_v
        prep_dtau = artifact.prepared["dtau_dt"].to_numpy(dtype=float)

        raw_tau_stats = series_stats(raw_time, raw_tau)
        prep_tau_stats = series_stats(prep_time, prep_tau)
        raw_dtau_stats = series_stats(raw_time, raw_dtau)
        prep_dtau_stats = series_stats(prep_time, prep_dtau)
        raw_v_stats = series_stats(raw_time, raw_v)
        prep_v_stats = series_stats(prep_time, prep_v)
        raw_gap_stats = series_stats(raw_time, raw_gap)
        prep_gap_stats = series_stats(prep_time, prep_gap)

        smoothing_effect = {
            "tau_std_ratio_prepared_over_raw": metric_ratio(prep_tau_stats["std"], raw_tau_stats["std"]),
            "tau_range_ratio_prepared_over_raw": metric_ratio(prep_tau_stats["range"], raw_tau_stats["range"]),
            "tau_total_variation_ratio_prepared_over_raw": metric_ratio(prep_tau_stats["total_variation"], raw_tau_stats["total_variation"]),
            "tau_avg_abs_derivative_ratio_prepared_over_raw": metric_ratio(prep_tau_stats["avg_abs_derivative"], raw_tau_stats["avg_abs_derivative"]),
            "dtau_std_ratio_prepared_over_raw": metric_ratio(prep_dtau_stats["std"], raw_dtau_stats["std"]),
        }

        row = {
            "step_name": step_name,
            "raw_rows": int(len(artifact.raw_segment)),
            "prepared_rows": int(len(artifact.prepared)),
            "retained_fraction_vs_raw": float(len(artifact.prepared) / max(len(artifact.raw_segment), 1)),
            "downsampled_fraction_vs_raw": float(len(artifact.raw_downsampled) / max(len(artifact.raw_segment), 1)),
            "duration_s": prep_tau_stats["duration_s"],
            "tau_mean": prep_tau_stats["mean"],
            "tau_std": prep_tau_stats["std"],
            "tau_variance": prep_tau_stats["variance"],
            "tau_min": prep_tau_stats["min"],
            "tau_max": prep_tau_stats["max"],
            "tau_range": prep_tau_stats["range"],
            "tau_iqr": prep_tau_stats["iqr"],
            "tau_total_variation": prep_tau_stats["total_variation"],
            "tau_total_variation_per_s": metric_ratio(prep_tau_stats["total_variation"], max(prep_tau_stats["duration_s"], EPS)),
            "tau_avg_abs_derivative": prep_tau_stats["avg_abs_derivative"],
            "tau_peak_to_peak": prep_tau_stats["peak_to_peak"],
            "tau_dominant_frequency_hz": prep_tau_stats["dominant_frequency_hz"],
            "tau_rough_oscillation_count": prep_tau_stats["rough_oscillation_count"],
            "dtau_mean": prep_dtau_stats["mean"],
            "dtau_std": prep_dtau_stats["std"],
            "dtau_range": prep_dtau_stats["range"],
            "V_mean": prep_v_stats["mean"],
            "V_std": prep_v_stats["std"],
            "V_range": prep_v_stats["range"],
            "V_drive_minus_V_mean": prep_gap_stats["mean"],
            "V_drive_minus_V_std": prep_gap_stats["std"],
            "V_drive_minus_V_range": prep_gap_stats["range"],
            "theta_event_valid": bool(artifact.inclusion_row.get("theta_event_valid", False)),
            "theta_sample_valid": bool(artifact.inclusion_row.get("theta_sample_valid", False)),
            "theta_log_correlation": float(artifact.inclusion_row.get("theta_log_correlation", float("nan"))),
            "theta_reason": str(artifact.inclusion_row.get("theta_reason", "")),
            "tau_std_ratio_prepared_over_raw": smoothing_effect["tau_std_ratio_prepared_over_raw"],
            "tau_range_ratio_prepared_over_raw": smoothing_effect["tau_range_ratio_prepared_over_raw"],
            "tau_total_variation_ratio_prepared_over_raw": smoothing_effect["tau_total_variation_ratio_prepared_over_raw"],
            "tau_avg_abs_derivative_ratio_prepared_over_raw": smoothing_effect["tau_avg_abs_derivative_ratio_prepared_over_raw"],
            "dtau_std_ratio_prepared_over_raw": smoothing_effect["dtau_std_ratio_prepared_over_raw"],
        }
        rows.append(row)
        detail[step_name] = {
            "raw_tau_stats": raw_tau_stats,
            "prepared_tau_stats": prep_tau_stats,
            "raw_dtau_stats": raw_dtau_stats,
            "prepared_dtau_stats": prep_dtau_stats,
            "raw_v_stats": raw_v_stats,
            "prepared_v_stats": prep_v_stats,
            "raw_v_drive_minus_v_stats": raw_gap_stats,
            "prepared_v_drive_minus_v_stats": prep_gap_stats,
            "smoothing_effect": smoothing_effect,
            "preprocessing_audit": {
                "raw_rows": int(len(artifact.raw_segment)),
                "rows_after_downsampling_before_smoothing": int(len(artifact.raw_downsampled)),
                "rows_after_preparation": int(len(artifact.prepared)),
                "raw_duration_s": float(artifact.raw_segment["time"].iloc[-1] - artifact.raw_segment["time"].iloc[0]),
                "prepared_duration_s": float(artifact.prepared["time"].iloc[-1] - artifact.prepared["time"].iloc[0]),
                "tau_window": SMOOTHING_WINDOW,
                "derivative_window": DERIV_WINDOW,
                "v_clipped_fraction_after_smoothing": float(np.mean(artifact.prepared["V"].to_numpy(dtype=float) <= proposal.EPS)),
                "sample_theta_ok_fraction": float(np.mean(artifact.prepared["sample_theta_ok"].to_numpy(dtype=bool))),
                "theta_run_start_time": artifact.inclusion_row.get("theta_run_start_time"),
                "theta_run_end_time": artifact.inclusion_row.get("theta_run_end_time"),
            },
        }
    frame = pd.DataFrame(rows)
    rank_cols = [
        "tau_std",
        "tau_range",
        "tau_iqr",
        "tau_total_variation_per_s",
        "tau_avg_abs_derivative",
        "dtau_std",
    ]
    z_cols = []
    for col in rank_cols:
        z_col = f"{col}_z"
        z_cols.append(z_col)
        denom = float(frame[col].std(ddof=0)) or 1.0
        frame[z_col] = (frame[col] - float(frame[col].mean())) / denom
    frame["tau_variability_score"] = frame[z_cols].mean(axis=1)
    frame = frame.sort_values(["tau_variability_score", "tau_total_variation_per_s"], ascending=[False, False]).reset_index(drop=True)
    frame["tau_variability_rank"] = np.arange(1, len(frame) + 1)
    return frame, detail


def standardized_distances(feature_df: pd.DataFrame, train_names: list[str], holdout_names: list[str], feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    train = feature_df.loc[feature_df["step_name"].isin(train_names), ["step_name", *feature_cols]].reset_index(drop=True)
    holdout = feature_df.loc[feature_df["step_name"].isin(holdout_names), ["step_name", *feature_cols]].reset_index(drop=True)
    means = train[feature_cols].mean(axis=0)
    stds = train[feature_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
    z_rows = []
    train_scaled = (train[feature_cols] - means) / stds
    centroid = train_scaled.mean(axis=0).to_numpy(dtype=float)
    for _, row in holdout.iterrows():
        z = ((row[feature_cols] - means) / stds).to_dict()
        z["step_name"] = row["step_name"]
        z["distance_to_train_centroid"] = float(np.linalg.norm(((row[feature_cols] - means) / stds).to_numpy(dtype=float) - centroid))
        z_rows.append(z)
    return pd.DataFrame(z_rows), {"means": means.to_dict(), "stds": stds.to_dict(), "train_centroid": centroid.tolist()}


def pca_coordinates(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = feature_df[feature_cols].to_numpy(dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    x = x / std
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    components = vh[:2].T
    coords = x @ components
    out = feature_df[["step_name"]].copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    return out


def select_regime_balanced_holdout(feature_df: pd.DataFrame) -> tuple[list[str], list[str], dict]:
    work = feature_df[["step_name", "tau_variability_score", "tau_std", "tau_total_variation_per_s", "V_drive_minus_V_std"]].copy()
    median_score = float(work["tau_variability_score"].median())
    work["regime"] = np.where(work["tau_variability_score"] <= median_score, "flatter", "more_variable")
    holdout_names = []
    rationale = {"median_variability_score": median_score, "per_step_regime": work.set_index("step_name")["regime"].to_dict()}
    for regime in ("flatter", "more_variable"):
        group = work.loc[work["regime"] == regime].copy().reset_index(drop=True)
        cols = ["tau_variability_score", "tau_std", "tau_total_variation_per_s", "V_drive_minus_V_std"]
        center = group[cols].mean(axis=0)
        scale = group[cols].std(axis=0, ddof=0).replace(0.0, 1.0)
        group["distance"] = np.sqrt(np.sum(((group[cols] - center) / scale) ** 2, axis=1))
        chosen = str(group.sort_values(["distance", "step_name"]).iloc[0]["step_name"])
        holdout_names.append(chosen)
    train_names = [name for name in feature_df["step_name"].tolist() if name not in holdout_names]
    rationale["holdout_names"] = holdout_names
    rationale["train_names"] = train_names
    return train_names, holdout_names, rationale


def tau_derivative_metrics(coefficients: dict[str, float], segment_df: pd.DataFrame) -> dict:
    target = segment_df["dtau_dt"].to_numpy(dtype=float)
    prediction = (
        coefficients.get("1", 0.0)
        + coefficients.get("V", 0.0) * segment_df["V"].to_numpy(dtype=float)
        + coefficients.get("V_drive_minus_V", 0.0) * (segment_df["V_drive"].to_numpy(dtype=float) - segment_df["V"].to_numpy(dtype=float))
    )
    residual = prediction - target
    mse = float(np.mean(residual ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": safe_r2(target, prediction),
        "prediction": prediction,
    }


def tau_rollout_metrics(coefficients: dict[str, float], segment_df: pd.DataFrame) -> dict:
    time = segment_df["time"].to_numpy(dtype=float)
    tau_true = segment_df["tau"].to_numpy(dtype=float)
    v = segment_df["V"].to_numpy(dtype=float)
    v_drive = segment_df["V_drive"].to_numpy(dtype=float)

    def rhs(state: np.ndarray, t_val: float) -> list[float]:
        v_now = float(np.interp(t_val, time, v))
        v_drive_now = float(np.interp(t_val, time, v_drive))
        return [coefficients.get("1", 0.0) + coefficients.get("V", 0.0) * v_now + coefficients.get("V_drive_minus_V", 0.0) * (v_drive_now - v_now)]

    tau_pred = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), [float(tau_true[0])], time).reshape(-1)
    error = tau_pred - tau_true
    mse = float(np.mean(error ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "r2": safe_r2(tau_true, tau_pred),
        "tau_prediction": tau_pred,
    }


def evaluate_strategy(name: str, label: str, train_names: list[str], eval_names: list[str], prepared_map: dict[str, pd.DataFrame], honest_holdout: bool) -> dict:
    train_segments = [prepared_map[step].copy() for step in train_names]
    eval_segments = [prepared_map[step].copy() for step in eval_names]
    tau_model = proposal.fit_tau_recovery(train_segments, eval_segments)
    coefficients = tau_model["coefficients_physical"]
    derivative_rows = []
    rollout_rows = []
    for step_name in eval_names:
        segment_df = prepared_map[step_name]
        derivative = tau_derivative_metrics(coefficients, segment_df)
        rollout = tau_rollout_metrics(coefficients, segment_df)
        derivative_rows.append(
            {
                "step_name": step_name,
                "derivative_mse": derivative["mse"],
                "derivative_rmse": derivative["rmse"],
                "derivative_mae": derivative["mae"],
                "derivative_r2": derivative["r2"],
            }
        )
        rollout_rows.append(
            {
                "step_name": step_name,
                "tau_rollout_mse": rollout["mse"],
                "tau_rollout_rmse": rollout["rmse"],
                "tau_rollout_mae": rollout["mae"],
                "tau_rollout_r2": rollout["r2"],
                "max_abs_tau_error": rollout["max_abs_error"],
            }
        )
    return {
        "strategy_name": name,
        "strategy_label": label,
        "train_steps": train_names,
        "eval_steps": eval_names,
        "honest_holdout": honest_holdout,
        "tau_model": tau_model,
        "coefficients_physical": coefficients,
        "equation": tau_model["exact_equation"],
        "one_term_equation": tau_model["one_term_equation"],
        "derivative_rows": derivative_rows,
        "rollout_rows": rollout_rows,
        "mean_derivative_mse": float(np.mean([row["derivative_mse"] for row in derivative_rows])),
        "mean_derivative_rmse": float(np.mean([row["derivative_rmse"] for row in derivative_rows])),
        "mean_derivative_mae": float(np.mean([row["derivative_mae"] for row in derivative_rows])),
        "mean_derivative_r2": float(np.nanmean([row["derivative_r2"] for row in derivative_rows])),
        "mean_tau_rollout_mse": float(np.mean([row["tau_rollout_mse"] for row in rollout_rows])),
        "mean_tau_rollout_rmse": float(np.mean([row["tau_rollout_rmse"] for row in rollout_rows])),
        "mean_tau_rollout_mae": float(np.mean([row["tau_rollout_mae"] for row in rollout_rows])),
        "mean_tau_rollout_r2": float(np.nanmean([row["tau_rollout_r2"] for row in rollout_rows])),
    }


def leave_two_out_experiments(prepared_map: dict[str, pd.DataFrame], feature_df: pd.DataFrame) -> tuple[list[dict], dict]:
    names = feature_df["step_name"].tolist()
    rows = []
    regime_map = feature_df.set_index("step_name")["tau_variability_score"].to_dict()
    global_center = feature_df[["tau_variability_score", "tau_std", "tau_total_variation_per_s", "V_drive_minus_V_std"]].mean(axis=0)
    global_scale = feature_df[["tau_variability_score", "tau_std", "tau_total_variation_per_s", "V_drive_minus_V_std"]].std(axis=0, ddof=0).replace(0.0, 1.0)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            holdout = [names[i], names[j]]
            train = [name for name in names if name not in holdout]
            result = evaluate_strategy("C_leave_two_out", f"Leave-two-out: {holdout[0]} + {holdout[1]}", train, holdout, prepared_map, honest_holdout=True)
            pair_rows = feature_df.loc[feature_df["step_name"].isin(holdout), ["tau_variability_score", "tau_std", "tau_total_variation_per_s", "V_drive_minus_V_std"]]
            pair_center = pair_rows.mean(axis=0)
            shift_distance = float(np.linalg.norm(((pair_center - global_center) / global_scale).to_numpy(dtype=float)))
            variability_gap = float(abs(regime_map[holdout[0]] - regime_map[holdout[1]]))
            result["pair_shift_distance_from_global_center"] = shift_distance
            result["pair_internal_variability_gap"] = variability_gap
            result["is_current_pair"] = set(holdout) == set(CURRENT_HOLDOUT)
            rows.append(result)
    current = next(row for row in rows if row["is_current_pair"])
    representative = sorted(rows, key=lambda row: (row["pair_shift_distance_from_global_center"], row["mean_tau_rollout_rmse"]))[0]
    summary = {
        "n_pairs": len(rows),
        "current_pair_rank_by_rollout_rmse": 1 + sum(row["mean_tau_rollout_rmse"] < current["mean_tau_rollout_rmse"] for row in rows),
        "current_pair_rank_by_shift_distance": 1 + sum(row["pair_shift_distance_from_global_center"] < current["pair_shift_distance_from_global_center"] for row in rows),
        "mean_leave_two_out_rollout_rmse": float(np.mean([row["mean_tau_rollout_rmse"] for row in rows])),
        "median_leave_two_out_rollout_rmse": float(np.median([row["mean_tau_rollout_rmse"] for row in rows])),
        "best_representative_pair": representative["eval_steps"],
        "best_representative_pair_rollout_rmse": representative["mean_tau_rollout_rmse"],
        "current_pair_rollout_rmse": current["mean_tau_rollout_rmse"],
        "current_pair_shift_distance": current["pair_shift_distance_from_global_center"],
        "representative_pair_shift_distance": representative["pair_shift_distance_from_global_center"],
    }
    return rows, summary


def plot_step_overlays(artifacts: dict[str, StepArtifacts]) -> None:
    ordered = STEP_NAMES
    fig, ax = plt.subplots(figsize=(11, 7))
    for step_name in ordered:
        df = artifacts[step_name].prepared
        rel = df["time"].to_numpy(dtype=float) - float(df["time"].iloc[0])
        normalized = rel / max(float(rel[-1]), EPS)
        ax.plot(normalized, df["tau"], linewidth=1.0, label=step_name)
    ax.set_xlabel("normalized time")
    ax.set_ylabel("tau")
    ax.set_title("Prepared tau(t) on normalized time axis")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tau_step_overlay_normalized.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for step_name in ordered:
        df = artifacts[step_name].prepared
        rel = df["time"].to_numpy(dtype=float) - float(df["time"].iloc[0])
        ax.plot(rel, df["tau"], linewidth=1.0, label=step_name)
    ax.set_xlabel("time since step start [s]")
    ax.set_ylabel("tau")
    ax.set_title("Prepared tau(t) on raw step time axis")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tau_step_overlay_rawtime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for column, ylabel, filename in (
        ("dtau_dt", "dtau/dt", "dtau_step_overlay.png"),
        ("V", "V", "v_step_overlay.png"),
    ):
        fig, ax = plt.subplots(figsize=(11, 7))
        for step_name in ordered:
            df = artifacts[step_name].prepared
            rel = df["time"].to_numpy(dtype=float) - float(df["time"].iloc[0])
            ax.plot(rel, df[column], linewidth=1.0, label=step_name)
        ax.set_xlabel("time since step start [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} overlays")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for step_name in ordered:
        df = artifacts[step_name].prepared
        rel = df["time"].to_numpy(dtype=float) - float(df["time"].iloc[0])
        ax.plot(rel, df["V_drive"] - df["V"], linewidth=1.0, label=step_name)
    ax.set_xlabel("time since step start [s]")
    ax.set_ylabel("V_drive - V")
    ax.set_title("V_drive(t) - V(t) overlays")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "vdrive_minus_v_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_focused_comparisons(artifacts: dict[str, StepArtifacts]) -> None:
    pairs = [("p5838_step2", "p5838_step5"), ("p5838_step7", "p5838_step9"), ("p5838_step2", "p5838_step10")]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 12), sharex=False)
    for row_index, (left_name, right_name) in enumerate(pairs):
        for col_index, step_name in enumerate((left_name, right_name)):
            artifact = artifacts[step_name]
            raw = artifact.raw_downsampled
            prepared = artifact.prepared
            rel_raw = raw["time"].to_numpy(dtype=float) - float(raw["time"].iloc[0])
            rel_prep = prepared["time"].to_numpy(dtype=float) - float(prepared["time"].iloc[0])
            ax = axes[row_index, col_index]
            ax.plot(rel_raw, raw["tau"], color="0.7", linewidth=0.9, label="raw/downsampled tau")
            ax.plot(rel_prep, prepared["tau"], color="tab:blue", linewidth=1.2, label="smoothed/prepared tau")
            ax.set_title(step_name)
            ax.set_xlabel("time since step start [s]")
            ax.set_ylabel("tau")
            ax.grid(True, alpha=0.3)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Focused raw vs smoothed tau comparisons", y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "holdout_vs_train_step_comparisons.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_space(feature_df: pd.DataFrame, current_holdout: list[str]) -> None:
    cols = ["tau_std", "tau_total_variation_per_s", "dtau_std", "V_std", "V_drive_minus_V_std"]
    coords = pca_coordinates(feature_df, cols)
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in coords.iterrows():
        step_name = row["step_name"]
        is_holdout = step_name in current_holdout
        ax.scatter(row["pc1"], row["pc2"], s=80, color=("tab:red" if is_holdout else "tab:blue"), alpha=0.85)
        ax.text(row["pc1"] + 0.02, row["pc2"] + 0.02, step_name.replace("p5838_", ""), fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Step-level summary feature space")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "holdout_feature_space.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_split_strategy_comparison(strategy_rows: list[dict], representative_cv: dict, prepared_map: dict[str, pd.DataFrame]) -> None:
    selected = []
    selected.append(next(row for row in strategy_rows if row["strategy_name"] == "A_current_split"))
    selected.append(next(row for row in strategy_rows if row["strategy_name"] == "B_all_step_descriptive"))
    selected.append(next(row for row in strategy_rows if row["strategy_name"] == "D_regime_balanced"))
    labels = ["Current", "All-step", "Balanced"]
    derivative_rmse = [row["mean_derivative_rmse"] for row in selected]
    rollout_rmse = [row["mean_tau_rollout_rmse"] for row in selected]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(labels, derivative_rmse, color=["tab:red", "tab:green", "tab:blue"], alpha=0.8)
    axes[0, 0].set_ylabel("mean derivative RMSE")
    axes[0, 0].set_title("Strategy derivative accuracy")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].bar(labels, rollout_rmse, color=["tab:red", "tab:green", "tab:blue"], alpha=0.8)
    axes[0, 1].set_ylabel("mean tau rollout RMSE")
    axes[0, 1].set_title("Strategy rollout accuracy")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    current = selected[0]
    current_holdout = current["eval_steps"][0]
    current_segment = prepared_map[current_holdout]
    current_roll = tau_rollout_metrics(current["coefficients_physical"], current_segment)
    rel_time = current_segment["time"].to_numpy(dtype=float) - float(current_segment["time"].iloc[0])
    axes[1, 0].plot(rel_time, current_segment["tau"], label=f"{current_holdout} observed", linewidth=1.2)
    axes[1, 0].plot(rel_time, current_roll["tau_prediction"], label="current predicted", linewidth=1.0)
    axes[1, 0].set_title(f"Current split example: {current_holdout}")
    axes[1, 0].set_xlabel("time since step start [s]")
    axes[1, 0].set_ylabel("tau")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    rep_holdout = representative_cv["eval_steps"][0]
    rep_segment = prepared_map[rep_holdout]
    rep_roll = tau_rollout_metrics(representative_cv["coefficients_physical"], rep_segment)
    rel_time = rep_segment["time"].to_numpy(dtype=float) - float(rep_segment["time"].iloc[0])
    axes[1, 1].plot(rel_time, rep_segment["tau"], label=f"{rep_holdout} observed", linewidth=1.2)
    axes[1, 1].plot(rel_time, rep_roll["tau_prediction"], label="representative CV predicted", linewidth=1.0)
    axes[1, 1].set_title(f"Representative leave-two-out: {', '.join(representative_cv['eval_steps'])}")
    axes[1, 1].set_xlabel("time since step start [s]")
    axes[1, 1].set_ylabel("tau")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tau_split_strategy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_markdown_reports(
    variability_df: pd.DataFrame,
    detail: dict,
    shift_df: pd.DataFrame,
    shift_summary: dict,
    pca_df: pd.DataFrame,
    strategy_table: pd.DataFrame,
    leave_two_out_summary: dict,
    balanced_rationale: dict,
) -> tuple[str, str, str]:
    top = variability_df[["tau_variability_rank", "step_name", "tau_variability_score", "tau_std", "tau_total_variation_per_s", "tau_avg_abs_derivative", "tau_range"]].copy()
    holdout_rows = variability_df.loc[variability_df["step_name"].isin(CURRENT_HOLDOUT), ["step_name", "tau_variability_rank", "tau_variability_score", "tau_std", "tau_total_variation_per_s", "tau_range", "tau_std_ratio_prepared_over_raw", "tau_total_variation_ratio_prepared_over_raw"]]

    variability_md = [
        "# Step Variability Diagnostics",
        "",
        "## Variability ranking",
        markdown_table(top),
        "",
        "## Holdout focus",
        markdown_table(holdout_rows),
        "",
        "## Preprocessing audit",
    ]
    for step_name in CURRENT_HOLDOUT:
        audit = detail[step_name]["preprocessing_audit"]
        effect = detail[step_name]["smoothing_effect"]
        variability_md.extend(
            [
                f"### {step_name}",
                f"- Raw rows: `{audit['raw_rows']}`",
                f"- Rows after downsampling before smoothing: `{audit['rows_after_downsampling_before_smoothing']}`",
                f"- Rows after preparation: `{audit['rows_after_preparation']}`",
                f"- Sample-theta-ok fraction: `{audit['sample_theta_ok_fraction']:.3f}`",
                f"- Prepared/raw tau std ratio: `{effect['tau_std_ratio_prepared_over_raw']:.3f}`",
                f"- Prepared/raw tau total-variation ratio: `{effect['tau_total_variation_ratio_prepared_over_raw']:.3f}`",
                f"- Prepared/raw dtau std ratio: `{effect['dtau_std_ratio_prepared_over_raw']:.3f}`",
            ]
        )
    variability_md.append("")

    shift_md = [
        "# Holdout Shift Assessment",
        "",
        "## Holdout z-scores against current training set",
        markdown_table(shift_df),
        "",
        "## PCA coordinates",
        markdown_table(pca_df),
        "",
        "## Notes",
        f"- Current holdout pair rank by leave-two-out rollout RMSE: `{leave_two_out_summary['current_pair_rank_by_rollout_rmse']}` / `{leave_two_out_summary['n_pairs']}`",
        f"- Current holdout pair rank by shift distance: `{leave_two_out_summary['current_pair_rank_by_shift_distance']}` / `{leave_two_out_summary['n_pairs']}`",
        f"- Most representative leave-two-out pair by feature-space distance: `{', '.join(leave_two_out_summary['best_representative_pair'])}`",
        f"- Regime-balanced holdout picked one `{balanced_rationale['per_step_regime'][balanced_rationale['holdout_names'][0]]}` and one `{balanced_rationale['per_step_regime'][balanced_rationale['holdout_names'][1]]}` step.",
        "",
    ]

    strategy_md = [
        "# Tau Split Strategy Comparison",
        "",
        "## Strategy table",
        markdown_table(strategy_table),
        "",
        "## Leave-two-out summary",
        f"- Mean leave-two-out rollout RMSE: `{leave_two_out_summary['mean_leave_two_out_rollout_rmse']:.6f}`",
        f"- Median leave-two-out rollout RMSE: `{leave_two_out_summary['median_leave_two_out_rollout_rmse']:.6f}`",
        f"- Current pair rollout RMSE: `{leave_two_out_summary['current_pair_rollout_rmse']:.6f}`",
        f"- Best representative pair: `{', '.join(leave_two_out_summary['best_representative_pair'])}` with rollout RMSE `{leave_two_out_summary['best_representative_pair_rollout_rmse']:.6f}`",
        "",
        "## Honesty note",
        "- `B_all_step_descriptive` is in-sample and optimistic by construction.",
        "- The leave-two-out rows are honest holdout evaluations.",
        "",
    ]
    return "\n".join(variability_md) + "\n", "\n".join(shift_md) + "\n", "\n".join(strategy_md) + "\n"


def final_assessment(
    variability_df: pd.DataFrame,
    detail: dict,
    shift_df: pd.DataFrame,
    strategy_summary_rows: list[dict],
    leave_two_out_summary: dict,
    balanced_result: dict,
) -> dict:
    current = next(row for row in strategy_summary_rows if row["strategy_name"] == "A_current_split")
    all_step = next(row for row in strategy_summary_rows if row["strategy_name"] == "B_all_step_descriptive")
    balanced = next(row for row in strategy_summary_rows if row["strategy_name"] == "D_regime_balanced")
    holdout_ranks = variability_df.set_index("step_name")["tau_variability_rank"].to_dict()
    holdout_scores = variability_df.set_index("step_name")["tau_variability_score"].to_dict()
    atypical_tau = []
    for _, row in shift_df.iterrows():
        flags = [col for col in shift_df.columns if col not in {"step_name", "distance_to_train_centroid"} and abs(float(row[col])) >= 1.5]
        if flags:
            atypical_tau.append({"step_name": row["step_name"], "feature_flags": flags, "distance_to_train_centroid": float(row["distance_to_train_centroid"])})
    smoothing_flags = {}
    for step_name in CURRENT_HOLDOUT:
        effect = detail[step_name]["smoothing_effect"]
        smoothing_flags[step_name] = {
            "tau_std_ratio_prepared_over_raw": effect["tau_std_ratio_prepared_over_raw"],
            "tau_total_variation_ratio_prepared_over_raw": effect["tau_total_variation_ratio_prepared_over_raw"],
            "dtau_std_ratio_prepared_over_raw": effect["dtau_std_ratio_prepared_over_raw"],
            "substantially_flattened": bool(
                effect["tau_total_variation_ratio_prepared_over_raw"] < 0.80
                or effect["dtau_std_ratio_prepared_over_raw"] < 0.80
            ),
        }
    return {
        "step2_and_step7_tau_variability_ranks": {step: int(holdout_ranks[step]) for step in CURRENT_HOLDOUT},
        "step2_and_step7_tau_variability_scores": {step: float(holdout_scores[step]) for step in CURRENT_HOLDOUT},
        "holdout_steps_atypical_feature_flags": atypical_tau,
        "smoothing_flags": smoothing_flags,
        "current_split_rollout_rmse": current["mean_tau_rollout_rmse"],
        "all_step_descriptive_rollout_rmse": all_step["mean_tau_rollout_rmse"],
        "balanced_split_rollout_rmse": balanced["mean_tau_rollout_rmse"],
        "current_split_derivative_rmse": current["mean_derivative_rmse"],
        "all_step_descriptive_derivative_rmse": all_step["mean_derivative_rmse"],
        "balanced_split_derivative_rmse": balanced["mean_derivative_rmse"],
        "leave_two_out_summary": leave_two_out_summary,
        "balanced_split_steps": balanced_result["eval_steps"],
    }


def main() -> None:
    artifacts, eligible = build_step_artifacts()
    eligible_holdout = eligible.head(2)["step_name"].tolist()
    if eligible_holdout != CURRENT_HOLDOUT:
        raise RuntimeError(f"Reconstructed duration-based holdout does not match expected current split: {eligible_holdout}")

    variability_df, detail = compute_step_metrics(artifacts)
    feature_cols = ["tau_std", "tau_range", "tau_total_variation_per_s", "dtau_std", "V_std", "V_drive_minus_V_std"]
    shift_df, shift_summary = standardized_distances(variability_df, CURRENT_TRAIN, CURRENT_HOLDOUT, feature_cols)
    pca_df = pca_coordinates(variability_df, feature_cols)

    prepared_map = {step: artifact.prepared.copy() for step, artifact in artifacts.items()}
    current_result = evaluate_strategy("A_current_split", "Current split", CURRENT_TRAIN, CURRENT_HOLDOUT, prepared_map, honest_holdout=True)
    all_step_result = evaluate_strategy("B_all_step_descriptive", "All-step descriptive", STEP_NAMES, STEP_NAMES, prepared_map, honest_holdout=False)
    l2o_rows, l2o_summary = leave_two_out_experiments(prepared_map, variability_df)
    balanced_train, balanced_holdout, balanced_rationale = select_regime_balanced_holdout(variability_df)
    balanced_result = evaluate_strategy("D_regime_balanced", "Regime-balanced split", balanced_train, balanced_holdout, prepared_map, honest_holdout=True)
    representative_cv = next(row for row in l2o_rows if row["eval_steps"] == l2o_summary["best_representative_pair"])

    plot_step_overlays(artifacts)
    plot_focused_comparisons(artifacts)
    plot_feature_space(variability_df, CURRENT_HOLDOUT)
    plot_split_strategy_comparison([current_result, all_step_result, balanced_result], representative_cv, prepared_map)

    strategy_rows = []
    for row in [current_result, all_step_result, *l2o_rows, balanced_result]:
        strategy_rows.append(
            {
                "strategy_name": row["strategy_name"],
                "strategy_label": row["strategy_label"],
                "train_steps": ",".join(row["train_steps"]),
                "eval_steps": ",".join(row["eval_steps"]),
                "honest_holdout": row["honest_holdout"],
                "equation": row["equation"],
                "mean_derivative_rmse": row["mean_derivative_rmse"],
                "mean_derivative_mae": row["mean_derivative_mae"],
                "mean_derivative_r2": row["mean_derivative_r2"],
                "mean_tau_rollout_rmse": row["mean_tau_rollout_rmse"],
                "mean_tau_rollout_mae": row["mean_tau_rollout_mae"],
                "mean_tau_rollout_r2": row["mean_tau_rollout_r2"],
                "pair_shift_distance_from_global_center": row.get("pair_shift_distance_from_global_center", float("nan")),
                "pair_internal_variability_gap": row.get("pair_internal_variability_gap", float("nan")),
                "is_current_pair": row.get("is_current_pair", False),
            }
        )
    strategy_table = pd.DataFrame(strategy_rows).sort_values(["strategy_name", "mean_tau_rollout_rmse"]).reset_index(drop=True)

    assessment = final_assessment(
        variability_df=variability_df,
        detail=detail,
        shift_df=shift_df,
        strategy_summary_rows=[current_result, all_step_result, balanced_result],
        leave_two_out_summary=l2o_summary,
        balanced_result=balanced_result,
    )

    variability_md, shift_md, strategy_md = build_markdown_reports(
        variability_df=variability_df,
        detail=detail,
        shift_df=shift_df,
        shift_summary=shift_summary,
        pca_df=pca_df,
        strategy_table=strategy_table,
        leave_two_out_summary=l2o_summary,
        balanced_rationale=balanced_rationale,
    )

    (RESULTS_DIR / "step_variability_diagnostics.md").write_text(variability_md, encoding="utf-8")
    (RESULTS_DIR / "holdout_shift_assessment.md").write_text(shift_md, encoding="utf-8")
    (RESULTS_DIR / "tau_split_strategy_comparison.md").write_text(strategy_md, encoding="utf-8")

    variability_df.to_csv(RESULTS_DIR / "step_variability_table.csv", index=False)
    strategy_table.to_csv(RESULTS_DIR / "tau_split_strategy_table.csv", index=False)

    variability_json = {
        "current_split": {"train_steps": CURRENT_TRAIN, "holdout_steps": CURRENT_HOLDOUT},
        "duration_based_reconstruction": {"eligible_steps_in_duration_order": eligible["step_name"].tolist()},
        "variability_table": variability_df,
        "per_step_detail": detail,
        "assessment": assessment,
    }
    shift_json = {
        "feature_columns": feature_cols,
        "holdout_vs_train_zscores": shift_df,
        "train_standardization": shift_summary,
        "pca_coordinates": pca_df,
        "leave_two_out_summary": l2o_summary,
        "balanced_split_rationale": balanced_rationale,
        "assessment": assessment,
    }
    strategy_json = {
        "strategy_rows": strategy_rows,
        "current_split_result": current_result,
        "all_step_descriptive_result": all_step_result,
        "leave_two_out_rows": l2o_rows,
        "regime_balanced_result": balanced_result,
        "representative_leave_two_out_pair": representative_cv["eval_steps"],
        "leave_two_out_summary": l2o_summary,
        "assessment": assessment,
    }
    write_json(RESULTS_DIR / "step_variability_diagnostics.json", variability_json)
    write_json(RESULTS_DIR / "holdout_shift_assessment.json", shift_json)
    write_json(RESULTS_DIR / "tau_split_strategy_comparison.json", strategy_json)

    print(json.dumps(json_ready(assessment), indent=2))


if __name__ == "__main__":
    main()
