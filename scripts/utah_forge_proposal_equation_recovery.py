from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import lsq_linear


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_memory_refinement as memory_ref
from scripts import utah_forge_reviewer_ablation as reviewer_ablation
from src.derivatives import derivative_savgol
from src.io.utah_forge import load_utah_forge_dataset
from src.preprocess.common import smooth_series
from src.preprocess.utah_forge import build_utah_forge_state
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
CHECKPOINT_DIR = ensure_directory(RESULTS_DIR / "proposal_equation_checkpoints")
CHECKPOINT_VERSION = 1
SMOOTHING_WINDOW = 61
SMOOTHING_POLYORDER = 3
DERIV_WINDOW = 15
DERIV_POLYORDER = 3
EPS = 1e-6

TAU_THRESHOLDS = (0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
VELOCITY_THRESHOLDS = (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2)

THETA_CORR_MIN = 0.20
THETA_SAMPLE_MIN = 200
THETA_LOGSTD_MIN = 1e-4
THETA_DIRECT_LOGSTD_MIN = 1e-4
THETA_LONGEST_RUN_MIN = 200
THETA_ERROR_SIGMAS = 2.5
THETA_ERROR_FLOOR = 0.35
THETA_HARD_LOG_ERROR = 1.50

ACOUSTIC_CANDIDATES = ("timeshift", "avg_timeshift", "avg_RmsAmp", "RmsAmp")
MODEL_ORDER = ("A_exact_rsf", "B_reduced_rsf", "C_local_memory", "D_acoustic_augmented")
PREVIOUS_MODEL_PATTERNS = {
    "memory_divergence": (RESULTS_DIR / "p5838_memory_model_report.md", r"Mean holdout divergence time: `([^`]+)` s"),
    "delay_theta_corr": (RESULTS_DIR / "p5838_refinement_report.md", r"Mean theta correlation: `([^`]+)`"),
    "model_b_div": (RESULTS_DIR / "p5838_final_report.md", r"### Model B.*?Mean holdout divergence: `([^`]+)` s"),
    "model_c_div": (RESULTS_DIR / "p5838_final_report.md", r"### Model C.*?Mean holdout divergence: `([^`]+)` s"),
}


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
    if isinstance(value, Path):
        return str(value)
    return value


def checkpoint_payload_path(stage_name: str) -> Path:
    return CHECKPOINT_DIR / f"{stage_name}.pkl"


def checkpoint_meta_path(stage_name: str) -> Path:
    return CHECKPOINT_DIR / f"{stage_name}.json"


def save_pickle_checkpoint(stage_name: str, payload, summary: dict | None = None) -> None:
    payload_path = checkpoint_payload_path(stage_name)
    meta_path = checkpoint_meta_path(stage_name)
    pd.to_pickle(payload, payload_path)
    meta = {
        "stage": stage_name,
        "version": CHECKPOINT_VERSION,
        "payload_path": payload_path.name,
        "updated_epoch_s": time.time(),
        "summary": json_ready(summary or {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_pickle_checkpoint(stage_name: str):
    meta_path = checkpoint_meta_path(stage_name)
    payload_path = checkpoint_payload_path(stage_name)
    if not meta_path.exists() or not payload_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("version") != CHECKPOINT_VERSION:
        return None
    return pd.read_pickle(payload_path)


def write_json_artifact(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    working = frame.copy()
    if index:
        working = working.reset_index()
    columns = [str(column) for column in working.columns]
    rows = [[str(value) for value in row] for row in working.astype(object).fillna("").to_numpy().tolist()]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body]) if body else "\n".join([header, divider])


def timing_start(context: str, stage: str) -> float:
    print(f"[timing:{context}] start {stage}", flush=True)
    return time.perf_counter()


def timing_end(context: str, stage: str, started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    print(f"[timing:{context}] end {stage} elapsed_s={elapsed:.3f}", flush=True)
    return elapsed


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if np.std(x_valid) <= 1e-12 or np.std(y_valid) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    split = np.where(np.diff(indices) > 1)[0]
    starts = np.concatenate(([indices[0]], indices[split + 1]))
    ends = np.concatenate((indices[split] + 1, [indices[-1] + 1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def longest_true_run(mask: np.ndarray) -> tuple[int, tuple[int, int] | None]:
    runs = contiguous_true_runs(mask)
    if not runs:
        return 0, None
    lengths = [end - start for start, end in runs]
    index = int(np.argmax(lengths))
    return int(lengths[index]), runs[index]


def cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    if len(values) <= 1:
        return result
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    result[1:] = np.cumsum(increments)
    return result


def fill_missing_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if np.isfinite(values).all():
        return values
    indices = np.arange(len(values), dtype=float)
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.zeros_like(values)
    return np.interp(indices, indices[mask], values[mask])


def zscore_frame(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    scaled = frame.copy()
    scaling: dict[str, dict[str, float]] = {}
    for column in columns:
        mean = float(frame[column].mean())
        std = float(frame[column].std())
        if not np.isfinite(std) or std == 0:
            std = 1.0
        scaled[column] = (frame[column] - mean) / std
        scaling[column] = {"mean": mean, "std": std}
    return scaled, scaling


def denormalize_coefficients(
    normalized_coefficients: np.ndarray,
    feature_names: list[str],
    scaling: dict[str, dict[str, float]],
) -> dict[str, float]:
    physical: dict[str, float] = {}
    intercept = 0.0
    for coefficient, feature_name in zip(normalized_coefficients, feature_names):
        coefficient = float(coefficient)
        stats = scaling.get(feature_name, {"mean": 0.0, "std": 1.0})
        beta = coefficient / stats["std"]
        intercept -= coefficient * stats["mean"] / stats["std"]
        physical[feature_name] = beta
    physical["1"] = physical.get("1", 0.0) + intercept
    return physical


def constrained_stlsq(
    design: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    threshold: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    mandatory_terms: set[str],
) -> np.ndarray:
    active = np.ones(len(feature_names), dtype=bool)
    mandatory_mask = np.array([name in mandatory_terms for name in feature_names], dtype=bool)
    for _ in range(20):
        result = lsq_linear(
            design[:, active],
            target,
            bounds=(lower_bounds[active], upper_bounds[active]),
            method="trf",
            lsmr_tol="auto",
        )
        coefficients = np.zeros(len(feature_names), dtype=float)
        coefficients[active] = result.x
        small = np.abs(coefficients) < threshold
        small[mandatory_mask] = False
        updated_active = ~small
        if np.array_equal(updated_active, active):
            active = updated_active
            break
        active = updated_active

    result = lsq_linear(
        design[:, active],
        target,
        bounds=(lower_bounds[active], upper_bounds[active]),
        method="trf",
        lsmr_tol="auto",
    )
    coefficients = np.zeros(len(feature_names), dtype=float)
    coefficients[active] = result.x
    return coefficients


def format_equation(lhs: str, coefficient_map: dict[str, float], ordered_terms: list[str]) -> str:
    pieces: list[str] = []
    intercept = coefficient_map.get("1", 0.0)
    if abs(intercept) > 1e-14:
        pieces.append(f"{intercept:.6e}")
    for term in ordered_terms:
        coefficient = coefficient_map.get(term, 0.0)
        if abs(coefficient) <= 1e-14:
            continue
        sign = "+" if coefficient >= 0 else "-"
        pieces.append(f"{sign} {abs(coefficient):.6e}*{term}")
    if not pieces:
        return f"{lhs} = 0"
    if pieces[0].startswith("+ "):
        pieces[0] = pieces[0][2:]
    return f"{lhs} = " + " ".join(pieces)


def parse_previous_reports() -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, (path, pattern) in PREVIOUS_MODEL_PATTERNS.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(pattern, text, flags=re.S)
        if match:
            payload[key] = match.group(1)
    return payload


def load_enriched_utah_forge_state() -> tuple[pd.DataFrame, dict]:
    raw_df, summary = load_utah_forge_dataset(base.RAW_FILE)
    state_df, metadata = build_utah_forge_state(raw_df, summary["column_mapping"])
    state_df = base.enforce_monotonic_time(base.remove_invalid_rows(state_df))
    time_column = summary["column_mapping"]["time"]
    extra_columns = [column for column in ["sigmaN", "v_ext", *ACOUSTIC_CANDIDATES] if column in raw_df.columns]
    if extra_columns:
        extra_df = (
            raw_df[[time_column, *extra_columns]]
            .rename(columns={time_column: "time"})
            .drop_duplicates(subset=["time"])
            .sort_values("time")
        )
        state_df = state_df.merge(extra_df, on="time", how="left")
    return state_df, metadata


def feature_value(feature_name: str, row: pd.Series) -> float:
    v_value = max(float(row["V"]), EPS)
    sigma = float(row["sigmaN"])
    theta = max(float(row["theta"]), EPS)
    v0 = max(float(row["V0"]), EPS)
    dc = max(float(row["Dc"]), EPS)
    if feature_name == "1":
        return 1.0
    if feature_name == "tau":
        return float(row["tau"])
    if feature_name == "sigmaN":
        return sigma
    if feature_name == "sigmaN_logV":
        return sigma * math.log(v_value / v0)
    if feature_name == "sigmaN_logTheta":
        return sigma * math.log(theta * v0 / dc)
    if feature_name == "deltaS_local":
        return float(row["deltaS_local"])
    if feature_name == "deltaS_orth":
        return float(row["deltaS_orth"])
    if feature_name == "acoustic_feature":
        return float(row["acoustic_feature"])
    raise KeyError(f"Unsupported feature: {feature_name}")


def build_feature_table(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{feature: feature_value(feature, row) for feature in feature_names} for _, row in df.iterrows()])


def residualize_feature(
    frame: pd.DataFrame,
    source_column: str,
    regressors: list[str],
    output_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    working = frame.copy()
    design = np.column_stack([np.ones(len(working))] + [working[column].to_numpy(dtype=float) for column in regressors])
    target = working[source_column].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    working[output_column] = target - prediction
    metadata = {"intercept": float(coefficients[0])}
    metadata.update({column: float(value) for column, value in zip(regressors, coefficients[1:])})
    return working, metadata


def detect_acoustic_feature_name(state_df: pd.DataFrame) -> str | None:
    for column in ACOUSTIC_CANDIDATES:
        if column in state_df.columns:
            values = state_df[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) > 100 and float(np.nanstd(finite)) > 0:
                return column
    return None


def prepare_segment_with_rsf(
    segment_df: pd.DataFrame,
    step: delay_ref.RSFitStep,
    rsfit_globals: dict[str, np.ndarray],
    acoustic_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    requested_columns = ["step_name", "time", "tau", "V", "V_drive"]
    if acoustic_column and acoustic_column in segment_df.columns:
        requested_columns.append(acoustic_column)
    working = segment_df[requested_columns].copy()
    working = base.downsample_frame(working, max_points=base.MODEL_MAX_POINTS)
    working = base.enforce_monotonic_time(working)

    time = working["time"].to_numpy(dtype=float)
    tau = smooth_series(working["tau"].to_numpy(dtype=float), window=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER)
    velocity = smooth_series(working["V"].to_numpy(dtype=float), window=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER)
    velocity = np.clip(velocity, EPS, None)
    tau_dot = derivative_savgol(tau, t=time, window=DERIV_WINDOW, polyorder=DERIV_POLYORDER)
    v_dot = derivative_savgol(velocity, t=time, window=DERIV_WINDOW, polyorder=DERIV_POLYORDER)

    theta_direct, theta_status = reviewer_ablation.reconstruct_theta(
        pd.DataFrame({"time": time, "tau": tau, "V": velocity}),
        step,
        rsfit_globals,
    )
    theta_interp = np.interp(time, step.time, step.theta_eff)
    sigma_interp = np.interp(time, rsfit_globals["time"], rsfit_globals["sigmaN"])
    mu_interp = np.interp(time, rsfit_globals["time"], rsfit_globals["mu"])
    params = reviewer_ablation.effective_step_params(step)

    acoustic_values = np.full(len(time), np.nan, dtype=float)
    if acoustic_column and acoustic_column in working.columns:
        raw_acoustic = fill_missing_1d(working[acoustic_column].to_numpy(dtype=float))
        acoustic_values = smooth_series(raw_acoustic, window=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER)

    base_mask = np.isfinite(theta_interp) & (theta_interp > EPS) & np.isfinite(sigma_interp) & (sigma_interp > EPS)
    theta_corr = float("nan")
    theta_direct_logstd = 0.0
    theta_interp_logstd = float(np.std(np.log(np.clip(theta_interp[base_mask], EPS, None)))) if np.any(base_mask) else 0.0
    theta_log_error = np.full(len(time), np.nan, dtype=float)
    abs_theta_log_error_limit = float("nan")
    if theta_direct is not None:
        direct_mask = base_mask & np.isfinite(theta_direct) & (theta_direct > EPS)
        if np.any(direct_mask):
            log_direct = np.log(np.clip(theta_direct[direct_mask], EPS, None))
            log_interp = np.log(np.clip(theta_interp[direct_mask], EPS, None))
            theta_corr = safe_corr(log_direct, log_interp)
            theta_direct_logstd = float(np.std(log_direct))
            theta_log_error[direct_mask] = log_direct - log_interp
            abs_error = np.abs(theta_log_error[direct_mask])
            median_error = float(np.median(abs_error))
            mad_error = float(np.median(np.abs(abs_error - median_error)))
            robust_sigma = 1.4826 * mad_error
            abs_theta_log_error_limit = min(
                THETA_HARD_LOG_ERROR,
                max(THETA_ERROR_FLOOR, median_error + THETA_ERROR_SIGMAS * robust_sigma),
            )
            sample_mask = direct_mask & (np.abs(theta_log_error) <= abs_theta_log_error_limit)
        else:
            sample_mask = np.zeros(len(time), dtype=bool)
    else:
        sample_mask = np.zeros(len(time), dtype=bool)

    sample_keep_fraction = float(np.mean(sample_mask)) if len(sample_mask) else 0.0
    usable_samples = int(np.sum(sample_mask))
    longest_run, run_slice = longest_true_run(sample_mask)
    event_valid = bool(
        theta_direct is not None
        and np.any(base_mask)
        and float(theta_status["clipped_fraction"]) <= 0.25
        and theta_interp_logstd > THETA_LOGSTD_MIN
        and theta_direct_logstd > THETA_DIRECT_LOGSTD_MIN
        and np.isfinite(theta_corr)
        and theta_corr >= THETA_CORR_MIN
    )
    if event_valid and theta_direct is not None:
        direct_mask = base_mask & np.isfinite(theta_direct) & (theta_direct > EPS)
        sample_mask = direct_mask.copy()
        usable_samples = int(np.sum(sample_mask))
        longest_run, run_slice = longest_true_run(sample_mask)
        sample_keep_fraction = float(np.mean(sample_mask)) if len(sample_mask) else 0.0
    sample_valid = bool(
        theta_direct is not None
        and float(theta_status["clipped_fraction"]) <= 0.25
        and usable_samples >= THETA_SAMPLE_MIN
        and longest_run >= THETA_LONGEST_RUN_MIN
    )

    reason = []
    if theta_direct is None:
        reason.append("theta_direct_invalid")
    if theta_interp_logstd <= THETA_LOGSTD_MIN:
        reason.append("theta_interp_low_variation")
    if theta_direct_logstd <= THETA_DIRECT_LOGSTD_MIN:
        reason.append("theta_direct_low_variation")
    if np.isfinite(theta_corr) and theta_corr < THETA_CORR_MIN:
        reason.append("theta_alignment_low_corr")
    if not np.isfinite(theta_corr):
        reason.append("theta_alignment_undefined")
    if float(theta_status["clipped_fraction"]) > 0.25:
        reason.append("theta_clipped")
    if usable_samples < THETA_SAMPLE_MIN:
        reason.append("too_few_high_quality_samples")
    if longest_run < THETA_LONGEST_RUN_MIN:
        reason.append("short_high_quality_run")

    prepared = pd.DataFrame(
        {
            "step_name": working["step_name"].to_numpy(),
            "time": time,
            "tau": tau,
            "V": velocity,
            "V_drive": working["V_drive"].to_numpy(dtype=float),
            "sigmaN": sigma_interp,
            "mu": mu_interp,
            "theta": theta_interp,
            "theta_log_error": theta_log_error,
            "dtau_dt": tau_dot,
            "dV_dt": v_dot,
            "V0": float(params["V0"]),
            "Dc": float(params["Dc"]),
            "deltaS": cumulative_trapezoid(velocity, time),
            "acoustic_feature": acoustic_values,
            "acoustic_name": acoustic_column or "",
        }
    )
    prepared["deltaS_local"] = prepared["deltaS"] - float(prepared["deltaS"].iloc[0])
    prepared["sample_theta_ok"] = sample_mask
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "tau", "V", "sigmaN", "dtau_dt", "dV_dt"]).reset_index(drop=True)
    if np.any(prepared["sample_theta_ok"].to_numpy(dtype=bool)):
        masked_prepared = prepared.loc[prepared["sample_theta_ok"]].reset_index(drop=True)
    else:
        masked_prepared = None

    stats = {
        "step_name": str(segment_df["step_name"].iloc[0]),
        "n_rows": int(len(working)),
        "n_rows_after_clean": int(len(prepared)),
        "theta_event_valid": event_valid,
        "theta_sample_valid": sample_valid,
        "theta_direct_ok": bool(theta_direct is not None),
        "theta_log_correlation": theta_corr,
        "theta_interp_logstd": theta_interp_logstd,
        "theta_direct_logstd": theta_direct_logstd,
        "theta_invalid_fraction": float(theta_status["invalid_fraction"]) if theta_direct is not None else 1.0,
        "theta_clipped_fraction": float(theta_status["clipped_fraction"]) if theta_direct is not None else 1.0,
        "theta_usable_samples": usable_samples,
        "theta_keep_fraction": sample_keep_fraction,
        "theta_longest_run": int(longest_run),
        "theta_abs_log_error_q90": float(np.nanquantile(np.abs(theta_log_error), 0.90)) if np.isfinite(theta_log_error).any() else float("nan"),
        "theta_abs_log_error_limit": abs_theta_log_error_limit,
        "sigma_mean": float(np.mean(sigma_interp)),
        "sigma_cv": float(np.std(sigma_interp) / (np.mean(sigma_interp) + 1e-12)),
        "step_r2": float(step.params["R2"]),
        "V0": float(params["V0"]),
        "Dc": float(params["Dc"]),
        "a_ref": float(params["a"]),
        "b_ref": float(params["b"]),
        "mu0_ref": float(params["mu0"]),
        "theta_reason": ",".join(reason) if reason else "ok",
        "acoustic_name": acoustic_column or "",
        "acoustic_std": float(np.nanstd(acoustic_values)) if np.isfinite(acoustic_values).any() else float("nan"),
    }
    if run_slice is not None:
        stats["theta_run_start_time"] = float(time[run_slice[0]])
        stats["theta_run_end_time"] = float(time[run_slice[1] - 1])
    return prepared, masked_prepared, stats


def prepare_all_segments() -> tuple[dict[str, list[pd.DataFrame]], list[dict], dict[str, str], str | None]:
    state_df, _ = load_enriched_utah_forge_state()
    acoustic_name = detect_acoustic_feature_name(state_df)
    inventory_df, segments, steps = memory_ref.segment_step_windows(state_df)
    train_segments, holdout_segments, train_names, holdout_names = memory_ref.split_train_holdout_segments(inventory_df, segments)
    rsfit_globals = reviewer_ablation.load_rsfit_globals()

    outputs = {
        "tau_train": [],
        "tau_holdout": [],
        "theta_event_train": [],
        "theta_event_holdout": [],
        "theta_sample_train": [],
        "theta_sample_holdout": [],
        "all_train": [],
        "all_holdout": [],
    }
    inclusion_rows: list[dict] = []

    for source, split_name in ((train_segments, "train"), (holdout_segments, "holdout")):
        for segment_df in source:
            step_name = str(segment_df["step_name"].iloc[0])
            try:
                prepared, masked_prepared, stats = prepare_segment_with_rsf(segment_df, steps[step_name], rsfit_globals, acoustic_name)
                stats["split"] = split_name
                inclusion_rows.append(stats)
                outputs[f"tau_{split_name}"].append(prepared.copy())
                outputs[f"all_{split_name}"].append(prepared.copy())
                if stats["theta_event_valid"]:
                    outputs[f"theta_event_{split_name}"].append(prepared.copy())
                if stats["theta_sample_valid"] and masked_prepared is not None:
                    outputs[f"theta_sample_{split_name}"].append(masked_prepared.copy())
                print(
                    f"[prepare] {step_name} split={split_name} rows={len(prepared)} "
                    f"theta_event={stats['theta_event_valid']} theta_sample={stats['theta_sample_valid']} "
                    f"corr={stats['theta_log_correlation']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Skipping {step_name}: proposal preparation failed: {exc}", flush=True)
                inclusion_rows.append({"step_name": step_name, "split": split_name, "theta_event_valid": False, "theta_sample_valid": False, "error": str(exc)})

    insertion_note = {
        "best_insertion_point": "new standalone proposal-equation script using RSFit-aligned step segments",
        "why_current_models_miss_target": (
            "Current Utah FORGE paths either use delay proxies (tau_lag), memory surrogates (tau_avg, tau_ema, S), "
            "or theta-informed libraries with extra interaction terms, and many reported equations stay in normalized coordinates or without sign constraints. "
            "That makes them useful surrogates, but not clean recoveries of the two proposal equations."
        ),
        "train_steps": ", ".join(train_names),
        "holdout_steps": ", ".join(holdout_names),
    }
    return outputs, inclusion_rows, insertion_note, acoustic_name


def compute_design_diagnostics(feature_df: pd.DataFrame, target: np.ndarray) -> dict:
    diagnostics: dict[str, object] = {"feature_stats": [], "pairwise_correlations": []}
    for column in feature_df.columns:
        values = feature_df[column].to_numpy(dtype=float)
        diagnostics["feature_stats"].append(
            {
                "feature": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
            }
        )

    corr = feature_df.corr()
    diagnostics["correlation_matrix"] = corr.round(6).to_dict()
    for row_name in corr.index:
        for column_name in corr.columns:
            if row_name < column_name:
                diagnostics["pairwise_correlations"].append(
                    {"feature_a": row_name, "feature_b": column_name, "correlation": float(corr.loc[row_name, column_name])}
                )

    design = np.column_stack([np.ones(len(feature_df))] + [feature_df[column].to_numpy(dtype=float) for column in feature_df.columns])
    diagnostics["condition_number"] = float(np.linalg.cond(design))

    vif_rows = []
    for index, column in enumerate(feature_df.columns, start=1):
        y = design[:, index]
        regressors = np.delete(design, index, axis=1)
        coefficients, *_ = np.linalg.lstsq(regressors, y, rcond=None)
        prediction = regressors @ coefficients
        ss_res = float(np.sum((y - prediction) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        vif = float("inf") if 1.0 - r2 <= 1e-12 else float(1.0 / (1.0 - r2))
        vif_rows.append({"feature": column, "vif": vif, "r2_against_others": r2})
    diagnostics["vif"] = vif_rows

    diagnostics["sigma_cv_mean"] = float(np.mean(feature_df["sigmaN"].to_numpy(dtype=float) / (np.mean(feature_df["sigmaN"].to_numpy(dtype=float)) + 1e-12) * 0 + 0)) if "sigmaN" not in feature_df.columns else None
    if "sigmaN" in feature_df.columns:
        sigma_values = feature_df["sigmaN"].to_numpy(dtype=float)
        diagnostics["sigma_cv"] = float(np.std(sigma_values) / (np.mean(sigma_values) + 1e-12))
        intercept_only = np.ones((len(feature_df), 1), dtype=float)
        beta_sigma, *_ = np.linalg.lstsq(intercept_only, sigma_values, rcond=None)
        sigma_resid = sigma_values - intercept_only @ beta_sigma
        diagnostics["sigma_residual_fraction_after_intercept"] = float(np.std(sigma_resid) / (np.std(sigma_values) + 1e-12))

    if "sigmaN_logTheta" in feature_df.columns:
        theta_values = feature_df["sigmaN_logTheta"].to_numpy(dtype=float)
        theta_intercept = np.ones((len(feature_df), 1), dtype=float)
        beta_theta, *_ = np.linalg.lstsq(theta_intercept, theta_values, rcond=None)
        theta_resid_int = theta_values - theta_intercept @ beta_theta
        diagnostics["theta_residual_fraction_after_intercept"] = float(np.std(theta_resid_int) / (np.std(theta_values) + 1e-12))

        regressors = [np.ones(len(feature_df), dtype=float)]
        for column in ["tau", "sigmaN", "sigmaN_logV"]:
            if column in feature_df.columns:
                regressors.append(feature_df[column].to_numpy(dtype=float))
        theta_design = np.column_stack(regressors)
        beta_theta_full, *_ = np.linalg.lstsq(theta_design, theta_values, rcond=None)
        theta_resid = theta_values - theta_design @ beta_theta_full
        diagnostics["theta_residual_fraction_after_baseline"] = float(np.std(theta_resid) / (np.std(theta_values) + 1e-12))
        target_design = np.column_stack(regressors)
        beta_target, *_ = np.linalg.lstsq(target_design, target, rcond=None)
        target_resid = target - target_design @ beta_target
        diagnostics["theta_partial_correlation_with_dVdt"] = safe_corr(theta_resid, target_resid)
        diagnostics["theta_baseline_projection"] = {
            "intercept": float(beta_theta_full[0]),
            **{
                column: float(value)
                for column, value in zip([col for col in ["tau", "sigmaN", "sigmaN_logV"] if col in feature_df.columns], beta_theta_full[1:])
            },
        }
    return diagnostics


def exact_rsf_design_table(df: pd.DataFrame) -> pd.DataFrame:
    velocity = np.clip(df["V"].to_numpy(dtype=float), EPS, None)
    theta = np.clip(df["theta"].to_numpy(dtype=float), EPS, None)
    v0 = np.clip(df["V0"].to_numpy(dtype=float), EPS, None)
    dc = np.clip(df["Dc"].to_numpy(dtype=float), EPS, None)
    sigma = df["sigmaN"].to_numpy(dtype=float)
    log_v = np.log(velocity / v0)
    log_theta = np.log(theta * v0 / dc)
    return pd.DataFrame(
        {
            "tau": df["tau"].to_numpy(dtype=float),
            "sigmaN": sigma,
            "logV_over_V0": log_v,
            "logThetaV0_over_Dc": log_theta,
            "sigmaN_logV": sigma * log_v,
            "sigmaN_logTheta": sigma * log_theta,
        }
    )


def column_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "variance": float(np.var(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "range": float(np.max(finite) - np.min(finite)),
    }


def assess_primary_diagnosis(
    inclusion_rows: list[dict],
    exact_design_df: pd.DataFrame,
    target: np.ndarray,
    diagnostics: dict,
) -> dict:
    inclusion_df = pd.DataFrame(inclusion_rows)
    train_df = inclusion_df.loc[inclusion_df.get("split", pd.Series(dtype=str)) == "train"].copy()
    theta_event_steps = int(train_df.get("theta_event_valid", pd.Series(dtype=bool)).fillna(False).sum())
    theta_sample_steps = int(train_df.get("theta_sample_valid", pd.Series(dtype=bool)).fillna(False).sum())
    theta_sample_rows = int(train_df.get("theta_usable_samples", pd.Series(dtype=float)).fillna(0).sum())
    theta_event_rows = int(train_df.loc[train_df.get("theta_event_valid", pd.Series(dtype=bool)).fillna(False), "n_rows_after_clean"].sum()) if "n_rows_after_clean" in train_df.columns else 0
    event_sample_gap = max(theta_event_rows - theta_sample_rows, 0)
    event_sample_gap_fraction = float(event_sample_gap / max(theta_event_rows, 1))

    theta_feature = exact_design_df["sigmaN_logTheta"].to_numpy(dtype=float)
    baseline_columns = ["tau", "sigmaN", "sigmaN_logV"]
    baseline_design = np.column_stack([np.ones(len(exact_design_df))] + [exact_design_df[column].to_numpy(dtype=float) for column in baseline_columns])
    theta_beta, *_ = np.linalg.lstsq(baseline_design, theta_feature, rcond=None)
    theta_resid = theta_feature - baseline_design @ theta_beta
    target_beta, *_ = np.linalg.lstsq(baseline_design, target, rcond=None)
    target_resid = target - baseline_design @ target_beta

    mean_theta_corr = float(train_df.get("theta_log_correlation", pd.Series(dtype=float)).dropna().mean()) if not train_df.empty else float("nan")
    max_theta_invalid = float(train_df.get("theta_invalid_fraction", pd.Series(dtype=float)).fillna(1.0).max()) if not train_df.empty else 1.0
    sigma_cv = float(diagnostics.get("sigma_cv", float("nan")))
    theta_residual_fraction = float(diagnostics.get("theta_residual_fraction_after_baseline", float("nan")))
    partial_corr = float(diagnostics.get("theta_partial_correlation_with_dVdt", float("nan")))
    condition_number = float(diagnostics.get("condition_number", float("nan")))
    max_vif = float(max((row["vif"] for row in diagnostics.get("vif", [])), default=float("nan")))
    rank_ratio = float(diagnostics.get("smallest_to_largest_singular_ratio", float("nan")))

    flags = {
        "implementation_bug": bool(theta_sample_rows == 0 and theta_event_steps == 0 and (not np.isfinite(mean_theta_corr) or mean_theta_corr < 0.05)),
        "alignment_or_units_bug": bool(np.isfinite(mean_theta_corr) and mean_theta_corr < 0.2 and max_theta_invalid > 0.5),
        "over_filtering": bool(event_sample_gap_fraction > 0.25),
        "insufficient_theta_variation": bool(theta_residual_fraction < 0.10 or exact_design_df["logThetaV0_over_Dc"].std() < 0.15),
        "multicollinearity_or_structural_non_identifiability": bool(
            (np.isfinite(condition_number) and condition_number > 1e5)
            or (np.isfinite(max_vif) and max_vif > 10.0)
            or (np.isfinite(rank_ratio) and rank_ratio < 1e-8)
            or sigma_cv < 0.01
        ),
    }

    primary = [name for name, active in flags.items() if active]
    if not primary:
        primary = ["no_single_failure_mode_detected"]
    if len(primary) == 1:
        summary = primary[0]
    else:
        summary = "combination: " + ", ".join(primary)

    diagnostics_summary = {
        "mean_theta_log_correlation_train": mean_theta_corr,
        "max_theta_invalid_fraction_train": max_theta_invalid,
        "event_valid_train_steps": theta_event_steps,
        "sample_valid_train_steps": theta_sample_steps,
        "event_valid_train_rows": theta_event_rows,
        "sample_valid_train_rows": theta_sample_rows,
        "event_minus_sample_gap_rows": event_sample_gap,
        "event_minus_sample_gap_fraction": event_sample_gap_fraction,
        "theta_residual_fraction_after_baseline": theta_residual_fraction,
        "theta_partial_correlation_with_dVdt": partial_corr,
        "condition_number": condition_number,
        "max_vif": max_vif,
        "sigma_cv": sigma_cv,
    }
    return {"flags": flags, "summary": summary, "support": diagnostics_summary}


def add_deltaS_orth(train_segments: list[pd.DataFrame], holdout_segments: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, float]]:
    train_df = pd.concat(train_segments, ignore_index=True)
    base_features = build_feature_table(train_df, ["tau", "sigmaN_logV"])
    design = np.column_stack([np.ones(len(train_df)), base_features["tau"], base_features["sigmaN_logV"]])
    target = train_df["deltaS_local"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    def transform(segment_df: pd.DataFrame) -> pd.DataFrame:
        feature_df = build_feature_table(segment_df, ["tau", "sigmaN_logV"])
        prediction = coefficients[0] + coefficients[1] * feature_df["tau"].to_numpy(dtype=float) + coefficients[2] * feature_df["sigmaN_logV"].to_numpy(dtype=float)
        updated = segment_df.copy()
        updated["deltaS_orth"] = updated["deltaS_local"].to_numpy(dtype=float) - prediction
        return updated

    train_out = [transform(df) for df in train_segments]
    holdout_out = [transform(df) for df in holdout_segments]
    metadata = {"intercept": float(coefficients[0]), "tau": float(coefficients[1]), "sigmaN_logV": float(coefficients[2])}
    return train_out, holdout_out, metadata


def acoustic_incremental_diagnostic(train_df: pd.DataFrame) -> dict[str, float] | None:
    if "acoustic_feature" not in train_df.columns:
        return None
    acoustic = train_df["acoustic_feature"].to_numpy(dtype=float)
    if not np.isfinite(acoustic).any() or np.nanstd(acoustic) <= 0:
        return None
    feature_df = build_feature_table(train_df, ["tau", "sigmaN", "sigmaN_logV"])
    base_design = np.column_stack([np.ones(len(feature_df))] + [feature_df[column].to_numpy(dtype=float) for column in feature_df.columns])
    acoustic_beta, *_ = np.linalg.lstsq(base_design, acoustic, rcond=None)
    acoustic_resid = acoustic - base_design @ acoustic_beta
    target = train_df["dV_dt"].to_numpy(dtype=float)
    target_beta, *_ = np.linalg.lstsq(base_design, target, rcond=None)
    target_resid = target - base_design @ target_beta
    return {
        "acoustic_residual_std_fraction": float(np.std(acoustic_resid) / (np.std(acoustic) + 1e-12)),
        "acoustic_partial_correlation_with_dVdt": safe_corr(acoustic_resid, target_resid),
    }


def model_feature_definitions(acoustic_available: bool) -> list[dict]:
    definitions = [
        {
            "label": "A_exact_rsf",
            "feature_names": ["1", "tau", "sigmaN_logV", "sigmaN_logTheta"],
            "mandatory_terms": {"tau", "sigmaN_logV", "sigmaN_logTheta"},
            "bounds": {"tau": (0.0, np.inf), "sigmaN_logV": (-np.inf, 0.0), "sigmaN_logTheta": (-np.inf, 0.0)},
            "ordered_terms": ["tau", "sigmaN_logV", "sigmaN_logTheta"],
            "dataset_key": "theta_sample",
            "timing_mode": "observed_theta",
        },
        {
            "label": "B_reduced_rsf",
            "feature_names": ["1", "tau", "sigmaN", "sigmaN_logV"],
            "mandatory_terms": {"tau", "sigmaN", "sigmaN_logV"},
            "bounds": {"tau": (0.0, np.inf), "sigmaN": (-np.inf, 0.0), "sigmaN_logV": (-np.inf, 0.0)},
            "ordered_terms": ["tau", "sigmaN", "sigmaN_logV"],
            "dataset_key": "all",
            "timing_mode": "observed_sigma",
        },
        {
            "label": "C_local_memory",
            "feature_names": ["1", "tau", "sigmaN_logV", "deltaS_orth"],
            "mandatory_terms": {"tau", "sigmaN_logV", "deltaS_orth"},
            "bounds": {"tau": (0.0, np.inf), "sigmaN_logV": (-np.inf, 0.0), "deltaS_orth": (-np.inf, np.inf)},
            "ordered_terms": ["tau", "sigmaN_logV", "deltaS_orth"],
            "dataset_key": "all_orth",
            "timing_mode": "memory",
        },
    ]
    if acoustic_available:
        definitions.append(
            {
                "label": "D_acoustic_augmented",
                "feature_names": ["1", "tau", "sigmaN", "sigmaN_logV", "acoustic_feature"],
                "mandatory_terms": {"tau", "sigmaN", "sigmaN_logV", "acoustic_feature"},
                "bounds": {"tau": (0.0, np.inf), "sigmaN": (-np.inf, 0.0), "sigmaN_logV": (-np.inf, 0.0), "acoustic_feature": (-np.inf, np.inf)},
                "ordered_terms": ["tau", "sigmaN", "sigmaN_logV", "acoustic_feature"],
                "dataset_key": "all",
                "timing_mode": "observed_sigma",
            }
        )
    return definitions


def build_design_and_scaling(train_df: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, dict[str, dict[str, float]], pd.DataFrame]:
    raw_features = build_feature_table(train_df, feature_names)
    scaled = raw_features.copy()
    scale_columns = [column for column in feature_names if column != "1"]
    scaled, scaling = zscore_frame(scaled, scale_columns)
    if "1" in feature_names:
        scaled["1"] = 1.0
        scaling["1"] = {"mean": 0.0, "std": 1.0}
    return scaled[feature_names].to_numpy(dtype=float), scaling, raw_features


def predict_from_feature_table(feature_df: pd.DataFrame, coefficients: dict[str, float], feature_names: list[str]) -> np.ndarray:
    prediction = np.full(len(feature_df), coefficients.get("1", 0.0), dtype=float)
    for feature_name in feature_names:
        if feature_name == "1":
            continue
        prediction += coefficients.get(feature_name, 0.0) * feature_df[feature_name].to_numpy(dtype=float)
    return prediction


def predict_with_coefficients(segment_df: pd.DataFrame, coefficients: dict[str, float], feature_names: list[str]) -> np.ndarray:
    feature_df = build_feature_table(segment_df, feature_names)
    return predict_from_feature_table(feature_df, coefficients, feature_names)


def peak_time(values: np.ndarray, time: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(time[int(np.argmax(values))] - time[0])


def onset_time(values: np.ndarray, time: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    threshold = v_min + 0.15 * (v_max - v_min)
    indices = np.flatnonzero(values >= threshold)
    if len(indices) == 0:
        return float("nan")
    return float(time[int(indices[0])] - time[0])


def rollout_velocity_model(model_row: dict, segment_df: pd.DataFrame) -> dict[str, float]:
    coefficients = model_row["coefficients_physical"]
    feature_names = model_row["feature_names"]
    time = segment_df["time"].to_numpy(dtype=float)
    observed = segment_df["V"].to_numpy(dtype=float)
    observed_tau = segment_df["tau"].to_numpy(dtype=float)
    observed_sigma = segment_df["sigmaN"].to_numpy(dtype=float)
    observed_theta = segment_df["theta"].to_numpy(dtype=float) if "theta" in segment_df.columns else np.full(len(segment_df), np.nan)
    observed_acoustic = segment_df["acoustic_feature"].to_numpy(dtype=float) if "acoustic_feature" in segment_df.columns else np.full(len(segment_df), np.nan)
    v0 = segment_df["V0"].to_numpy(dtype=float)
    dc = segment_df["Dc"].to_numpy(dtype=float)

    use_memory = "deltaS_orth" in feature_names
    orth_map = model_row.get("deltaS_orth_map", {})

    def exogenous_row(t_value: float, current_v: float, current_s: float) -> dict[str, float]:
        tau = float(np.interp(t_value, time, observed_tau))
        sigma = float(np.interp(t_value, time, observed_sigma))
        theta = float(np.interp(t_value, time, observed_theta)) if np.isfinite(observed_theta).any() else 1.0
        acoustic = float(np.interp(t_value, time, observed_acoustic)) if np.isfinite(observed_acoustic).any() else 0.0
        v0_now = max(float(np.interp(t_value, time, v0)), EPS)
        dc_now = max(float(np.interp(t_value, time, dc)), EPS)
        delta_s_orth = current_s - (
            orth_map.get("intercept", 0.0)
            + orth_map.get("tau", 0.0) * tau
            + orth_map.get("sigmaN_logV", 0.0) * sigma * math.log(max(current_v, EPS) / v0_now)
        )
        return {
            "tau": tau,
            "sigmaN": sigma,
            "theta": max(theta, EPS),
            "V": max(current_v, EPS),
            "V0": v0_now,
            "Dc": dc_now,
            "deltaS_local": current_s,
            "deltaS_orth": delta_s_orth,
            "acoustic_feature": acoustic,
        }

    def rhs(state: np.ndarray, t_value: float) -> list[float]:
        current_v = max(float(state[0]), EPS)
        current_s = float(state[1]) if use_memory else 0.0
        row = exogenous_row(t_value, current_v, current_s)
        dvdt = coefficients.get("1", 0.0)
        for feature_name in feature_names:
            if feature_name == "1":
                continue
            dvdt += coefficients.get(feature_name, 0.0) * feature_value(feature_name, pd.Series(row))
        if use_memory:
            return [dvdt, current_v]
        return [dvdt]

    try:
        initial_state = [float(observed[0]), 0.0] if use_memory else [float(observed[0])]
        solution = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), initial_state, time)
        predicted = solution[:, 0]
    except Exception:
        predicted = np.full_like(observed, np.nan)

    finite = np.isfinite(predicted)
    if not np.any(finite):
        return {"rollout_mse": float("inf"), "stable_fraction": 0.0, "peak_timing_error_s": float("inf"), "onset_timing_error_s": float("inf")}

    sigma_obs = float(np.std(observed))
    threshold = 3.0 * max(sigma_obs, 1e-6)
    divergence_index = len(observed)
    for index, (predicted_value, observed_value) in enumerate(zip(predicted, observed)):
        if (not np.isfinite(predicted_value)) or abs(predicted_value - observed_value) > threshold:
            divergence_index = index
            break
    stable_fraction = float(divergence_index / len(observed))
    rollout_mse = float(np.mean((predicted[finite] - observed[finite]) ** 2)) if np.any(finite) else float("inf")
    peak_error = abs(peak_time(predicted[finite], time[finite]) - peak_time(observed[finite], time[finite])) if np.any(finite) else float("inf")
    onset_error = abs(onset_time(predicted[finite], time[finite]) - onset_time(observed[finite], time[finite])) if np.any(finite) else float("inf")
    return {
        "rollout_mse": rollout_mse,
        "stable_fraction": stable_fraction,
        "peak_timing_error_s": float(peak_error),
        "onset_timing_error_s": float(onset_error),
    }


def fit_velocity_model(
    train_segments: list[pd.DataFrame],
    holdout_segments: list[pd.DataFrame],
    model_def: dict,
    *,
    fast_mode: bool = False,
) -> dict:
    context = f"fit_velocity_model:{model_def['label']}" + (":fast" if fast_mode else "")
    overall_started = timing_start(context, "overall")

    feature_started = timing_start(context, "feature_construction")
    train_df = pd.concat(train_segments, ignore_index=True)
    design, scaling, raw_features = build_design_and_scaling(train_df, model_def["feature_names"])
    target = train_df["dV_dt"].to_numpy(dtype=float)
    acoustic_diag = acoustic_incremental_diagnostic(train_df) if "acoustic_feature" in model_def["feature_names"] else None
    holdout_feature_cache = []
    for segment_df in holdout_segments:
        holdout_feature_cache.append(
            {
                "step_name": str(segment_df["step_name"].iloc[0]),
                "segment_df": segment_df,
                "feature_df": build_feature_table(segment_df, model_def["feature_names"]),
                "target": segment_df["dV_dt"].to_numpy(dtype=float),
            }
        )
    timing_end(context, "feature_construction", feature_started)

    lower = []
    upper = []
    for feature_name in model_def["feature_names"]:
        bounds = model_def["bounds"].get(feature_name, (-np.inf, np.inf))
        lower.append(bounds[0])
        upper.append(bounds[1])
    lower_bounds = np.array(lower, dtype=float)
    upper_bounds = np.array(upper, dtype=float)

    candidates: list[dict] = []
    thresholds = (VELOCITY_THRESHOLDS[0],) if fast_mode else VELOCITY_THRESHOLDS
    regression_total = 0.0
    prediction_total = 0.0
    rollout_total = 0.0
    packaging_total = 0.0
    print(f"[timing:{context}] start regression_threshold_sweep", flush=True)
    print(f"[timing:{context}] start holdout_prediction", flush=True)
    print(f"[timing:{context}] start rollout_timing_validation", flush=True)
    print(f"[timing:{context}] start report_packaging", flush=True)
    for threshold in thresholds:
        regression_started = time.perf_counter()
        coefficients_z = constrained_stlsq(
            design,
            target,
            model_def["feature_names"],
            threshold=threshold,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            mandatory_terms=model_def["mandatory_terms"],
        )
        regression_total += time.perf_counter() - regression_started

        packaging_started = time.perf_counter()
        coefficients = denormalize_coefficients(coefficients_z, model_def["feature_names"], scaling)
        active_terms = [name for name in model_def["feature_names"] if abs(coefficients.get(name, 0.0)) > 1e-10]
        train_prediction = predict_from_feature_table(raw_features, coefficients, model_def["feature_names"])
        train_mse = float(np.mean((train_prediction - target) ** 2))
        packaging_total += time.perf_counter() - packaging_started

        holdout_derivative_rows = []
        rollout_rows = []
        prediction_started = time.perf_counter()
        for cache_row in holdout_feature_cache:
            segment_df = cache_row["segment_df"]
            prediction = predict_from_feature_table(cache_row["feature_df"], coefficients, model_def["feature_names"])
            observed_dvdt = cache_row["target"]
            residual = prediction - observed_dvdt
            derivative_mse = float(np.mean(residual ** 2))
            derivative_mae = float(np.mean(np.abs(residual)))
            derivative_rmse = float(np.sqrt(derivative_mse))
            ss_tot = float(np.sum((observed_dvdt - np.mean(observed_dvdt)) ** 2))
            derivative_r2 = float(1.0 - np.sum(residual ** 2) / ss_tot) if ss_tot > 1e-12 else float("nan")
            rollout_row = rollout_velocity_model(
                {**model_def, "feature_names": model_def["feature_names"], "coefficients_physical": coefficients, "deltaS_orth_map": model_def.get("deltaS_orth_map", {})},
                segment_df,
            )
            holdout_derivative_rows.append(
                {
                    "step_name": str(segment_df["step_name"].iloc[0]),
                    "derivative_mse": derivative_mse,
                    "derivative_mae": derivative_mae,
                    "derivative_rmse": derivative_rmse,
                    "derivative_r2": derivative_r2,
                }
            )
            if not fast_mode:
                rollout_started = time.perf_counter()
                rollout_row = rollout_velocity_model(
                    {**model_def, "feature_names": model_def["feature_names"], "coefficients_physical": coefficients, "deltaS_orth_map": model_def.get("deltaS_orth_map", {})},
                    segment_df,
                )
                rollout_total += time.perf_counter() - rollout_started
                rollout_rows.append({"step_name": str(segment_df["step_name"].iloc[0]), **rollout_row})
            else:
                rollout_rows.append(
                    {
                        "step_name": str(segment_df["step_name"].iloc[0]),
                        "rollout_mse": float("nan"),
                        "stable_fraction": float("nan"),
                        "peak_timing_error_s": float("nan"),
                        "onset_timing_error_s": float("nan"),
                    }
                )
        prediction_total += time.perf_counter() - prediction_started

        holdout_mse = float(np.mean([row["derivative_mse"] for row in holdout_derivative_rows]))
        holdout_mae = float(np.mean([row["derivative_mae"] for row in holdout_derivative_rows]))
        holdout_rmse = float(np.mean([row["derivative_rmse"] for row in holdout_derivative_rows]))
        holdout_r2 = float(np.nanmean([row["derivative_r2"] for row in holdout_derivative_rows]))
        if fast_mode:
            mean_rollout_mse = float("nan")
            mean_stable_fraction = float("nan")
            mean_peak_error = float("nan")
            mean_onset_error = float("nan")
            mean_timing_error = float("nan")
        else:
            mean_rollout_mse = float(np.mean([row["rollout_mse"] for row in rollout_rows]))
            mean_stable_fraction = float(np.mean([row["stable_fraction"] for row in rollout_rows]))
            mean_peak_error = float(np.mean([row["peak_timing_error_s"] for row in rollout_rows]))
            mean_onset_error = float(np.mean([row["onset_timing_error_s"] for row in rollout_rows]))
            mean_timing_error = float(np.mean([0.5 * (row["peak_timing_error_s"] + row["onset_timing_error_s"]) for row in rollout_rows]))

        sign_checks: dict[str, bool] = {}
        for feature_name, bounds in model_def["bounds"].items():
            if feature_name == "1":
                continue
            coefficient = coefficients.get(feature_name, 0.0)
            if bounds[0] == 0.0 and math.isinf(bounds[1]):
                sign_checks[f"{feature_name}_positive"] = bool(coefficient > 0)
            elif math.isinf(bounds[0]) and bounds[1] == 0.0:
                sign_checks[f"{feature_name}_negative"] = bool(coefficient < -1e-10)

        beta_tau = coefficients.get("tau", 0.0)
        beta_sigma = coefficients.get("sigmaN", 0.0)
        beta_sigma_v = coefficients.get("sigmaN_logV", 0.0)
        beta_sigma_theta = coefficients.get("sigmaN_logTheta", 0.0)
        m_hat = float("nan")
        mu0_hat = float("nan")
        a_hat = float("nan")
        b_hat = float("nan")
        if beta_tau > 0:
            m_hat = 1.0 / beta_tau
            if "sigmaN" in model_def["feature_names"]:
                mu0_hat = -beta_sigma / beta_tau
            a_hat = -beta_sigma_v / beta_tau if "sigmaN_logV" in model_def["feature_names"] else float("nan")
            b_hat = -beta_sigma_theta / beta_tau if "sigmaN_logTheta" in model_def["feature_names"] else float("nan")

        theta_term_active = bool(abs(beta_sigma_theta) > 1e-8) if "sigmaN_logTheta" in model_def["feature_names"] else False
        acoustic_term_active = bool(abs(coefficients.get("acoustic_feature", 0.0)) > 1e-8) if "acoustic_feature" in model_def["feature_names"] else False
        physics_consistent = all(sign_checks.values()) if sign_checks else True

        candidate = {
            "label": model_def["label"],
            "threshold": threshold,
            "feature_names": model_def["feature_names"],
            "coefficients_z": coefficients_z.tolist(),
            "coefficients_physical": coefficients,
            "active_terms": active_terms,
            "train_mse": train_mse,
            "holdout_mse": holdout_mse,
            "holdout_mae": holdout_mae,
            "holdout_rmse": holdout_rmse,
            "holdout_r2": holdout_r2,
            "holdout_derivative_rows": holdout_derivative_rows,
            "rollout_rows": rollout_rows,
            "mean_rollout_mse": mean_rollout_mse,
            "mean_stable_fraction": mean_stable_fraction,
            "mean_peak_timing_error_s": mean_peak_error,
            "mean_onset_timing_error_s": mean_onset_error,
            "mean_timing_error_s": mean_timing_error,
            "sign_checks": sign_checks,
            "physics_consistent": physics_consistent,
            "theta_term_active": theta_term_active,
            "acoustic_term_active": acoustic_term_active,
            "m_hat": m_hat,
            "mu0_hat": mu0_hat,
            "a_hat": a_hat,
            "b_hat": b_hat,
            "equation": format_equation("dV/dt", coefficients, model_def["ordered_terms"]),
            "deltaS_orth_map": model_def.get("deltaS_orth_map", {}),
            "acoustic_incremental_diagnostic": acoustic_diag,
            "fast_mode": fast_mode,
        }
        candidates.append(candidate)
        print(
            f"[velocity:{model_def['label']}] threshold={threshold:.5f} active={','.join(active_terms)} "
            f"holdout_mse={holdout_mse:.6e} stable={mean_stable_fraction:.3f} peak_err={mean_peak_error:.3f}",
            flush=True,
        )
    print(f"[timing:{context}] end regression_threshold_sweep elapsed_s={regression_total:.3f}", flush=True)
    print(f"[timing:{context}] end holdout_prediction elapsed_s={prediction_total:.3f}", flush=True)
    print(f"[timing:{context}] end rollout_timing_validation elapsed_s={rollout_total:.3f}", flush=True)

    packaging_started = time.perf_counter()
    best = sorted(
        candidates,
        key=lambda row: (
            -int(row["physics_consistent"]),
            -int(row["theta_term_active"]) if row["label"] == "A_exact_rsf" else 0,
            row["mean_timing_error_s"] if np.isfinite(row["mean_timing_error_s"]) else float("inf"),
            -row["mean_stable_fraction"] if np.isfinite(row["mean_stable_fraction"]) else float("inf"),
            row["holdout_mse"],
            len(row["active_terms"]),
        ),
    )[0]
    packaging_total += time.perf_counter() - packaging_started
    print(f"[timing:{context}] end report_packaging elapsed_s={packaging_total:.3f}", flush=True)
    timing_end(context, "overall", overall_started)
    return {"best": best, "all": candidates}


def fit_tau_recovery(train_segments: list[pd.DataFrame], holdout_segments: list[pd.DataFrame]) -> dict:
    context = "fit_tau_recovery"
    overall_started = timing_start(context, "overall")
    feature_started = timing_start(context, "feature_construction")
    train_df = pd.concat(train_segments, ignore_index=True)
    tau_feature_names = ["1", "V", "V_drive_minus_V"]
    tau_features = pd.DataFrame(
        {
            "1": 1.0,
            "V": train_df["V"].to_numpy(dtype=float),
            "V_drive_minus_V": (train_df["V_drive"] - train_df["V"]).to_numpy(dtype=float),
        }
    )
    scaled_features, scaling = zscore_frame(tau_features, ["V", "V_drive_minus_V"])
    scaled_features["1"] = 1.0
    design = scaled_features[tau_feature_names].to_numpy(dtype=float)
    holdout_cache = [
        {
            "step_name": str(segment_df["step_name"].iloc[0]),
            "V": segment_df["V"].to_numpy(dtype=float),
            "V_drive_minus_V": (segment_df["V_drive"] - segment_df["V"]).to_numpy(dtype=float),
            "target": segment_df["dtau_dt"].to_numpy(dtype=float),
        }
        for segment_df in holdout_segments
    ]
    timing_end(context, "feature_construction", feature_started)
    lower = np.array([-np.inf, -np.inf, 0.0], dtype=float)
    upper = np.array([np.inf, np.inf, np.inf], dtype=float)

    candidates = []
    print(f"[timing:{context}] start threshold_sweep", flush=True)
    sweep_started = time.perf_counter()
    for threshold in TAU_THRESHOLDS:
        coefficients_z = constrained_stlsq(
            design,
            train_df["dtau_dt"].to_numpy(dtype=float),
            tau_feature_names,
            threshold=threshold,
            lower_bounds=lower,
            upper_bounds=upper,
            mandatory_terms={"V_drive_minus_V"},
        )
        coefficients = denormalize_coefficients(coefficients_z, tau_feature_names, scaling)
        active_terms = [name for name in tau_feature_names if abs(coefficients.get(name, 0.0)) > 1e-10]
        train_prediction = (
            coefficients.get("1", 0.0)
            + coefficients.get("V", 0.0) * train_df["V"].to_numpy(dtype=float)
            + coefficients.get("V_drive_minus_V", 0.0) * (train_df["V_drive"] - train_df["V"]).to_numpy(dtype=float)
        )
        train_mse = float(np.mean((train_prediction - train_df["dtau_dt"].to_numpy(dtype=float)) ** 2))
        holdout_rows = []
        for cache_row in holdout_cache:
            prediction = (
                coefficients.get("1", 0.0)
                + coefficients.get("V", 0.0) * cache_row["V"]
                + coefficients.get("V_drive_minus_V", 0.0) * cache_row["V_drive_minus_V"]
            )
            holdout_rows.append(
                {
                    "step_name": cache_row["step_name"],
                    "mse": float(np.mean((prediction - cache_row["target"]) ** 2)),
                }
            )
        holdout_mse = float(np.mean([row["mse"] for row in holdout_rows]))
        candidates.append(
            {
                "threshold": threshold,
                "coefficients_z": coefficients_z.tolist(),
                "coefficients_physical": coefficients,
                "active_terms": active_terms,
                "train_mse": train_mse,
                "holdout_rows": holdout_rows,
                "holdout_mse": holdout_mse,
                "sign_ok": bool(coefficients.get("V_drive_minus_V", 0.0) > 0),
            }
        )
        print(
            f"[tau] threshold={threshold:.5f} active={','.join(active_terms)} holdout_mse={holdout_mse:.6e}",
            flush=True,
        )
    timing_end(context, "threshold_sweep", sweep_started)

    packaging_started = timing_start(context, "report_packaging")
    best = sorted(
        candidates,
        key=lambda row: (
            -int(row["sign_ok"]),
            -int("V_drive_minus_V" in row["active_terms"]),
            len(row["active_terms"]),
            row["holdout_mse"],
        ),
    )[0]

    drive_mismatch = (train_df["V_drive"] - train_df["V"]).to_numpy(dtype=float)
    one_term_k = float(np.dot(train_df["dtau_dt"].to_numpy(dtype=float), drive_mismatch) / (np.dot(drive_mismatch, drive_mismatch) + 1e-12))
    if one_term_k < 0:
        one_term_k = 0.0
    one_term_rows = []
    for segment_df in holdout_segments:
        prediction = one_term_k * (segment_df["V_drive"] - segment_df["V"]).to_numpy(dtype=float)
        one_term_rows.append(
            {
                "step_name": str(segment_df["step_name"].iloc[0]),
                "mse": float(np.mean((prediction - segment_df["dtau_dt"].to_numpy(dtype=float)) ** 2)),
            }
        )
    best["one_term_k_hat"] = one_term_k
    best["one_term_holdout_mse"] = float(np.mean([row["mse"] for row in one_term_rows]))
    best["exact_equation"] = format_equation("dtau/dt", best["coefficients_physical"], ["V", "V_drive_minus_V"])
    best["one_term_equation"] = f"dtau/dt ~= {one_term_k:.6e}*(V_drive - V)"
    timing_end(context, "report_packaging", packaging_started)
    timing_end(context, "overall", overall_started)
    return best


def semi_observed_tau_rollout(tau_model: dict, holdout_segments: list[pd.DataFrame]) -> list[dict]:
    coefficients = tau_model["coefficients_physical"]
    rows = []
    for segment_df in holdout_segments:
        time = segment_df["time"].to_numpy(dtype=float)
        observed_tau = segment_df["tau"].to_numpy(dtype=float)
        observed_v = segment_df["V"].to_numpy(dtype=float)
        observed_v_drive = segment_df["V_drive"].to_numpy(dtype=float)

        def rhs(state: np.ndarray, t_value: float) -> list[float]:
            v_now = float(np.interp(t_value, time, observed_v))
            v_drive_now = float(np.interp(t_value, time, observed_v_drive))
            return [coefficients.get("1", 0.0) + coefficients.get("V", 0.0) * v_now + coefficients.get("V_drive_minus_V", 0.0) * (v_drive_now - v_now)]

        try:
            tau_roll = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), [float(observed_tau[0])], time).reshape(-1)
            mse = float(np.mean((tau_roll - observed_tau) ** 2))
        except Exception:
            mse = float("inf")
        rows.append({"step_name": str(segment_df["step_name"].iloc[0]), "tau_rollout_mse": mse})
    return rows


def choose_final_velocity_model(model_results: dict[str, dict]) -> tuple[dict, str]:
    exact = model_results["A_exact_rsf"]["best"]
    reduced = model_results["B_reduced_rsf"]["best"]
    memory = model_results["C_local_memory"]["best"]
    acoustic = model_results.get("D_acoustic_augmented", {}).get("best")

    if exact["physics_consistent"] and exact["theta_term_active"] and exact["b_hat"] > 1e-5:
        return exact, "Exact equation (2) partially recovered but still weak" if exact["mean_stable_fraction"] < 0.9 else "Exact equation (2) recovered credibly"

    fallback = reduced
    if acoustic is not None:
        acoustic_better = (
            acoustic["physics_consistent"]
            and acoustic["holdout_mse"] < 0.98 * reduced["holdout_mse"]
            and acoustic["mean_peak_timing_error_s"] <= reduced["mean_peak_timing_error_s"]
        )
        if acoustic_better:
            fallback = acoustic
    memory_better = (
        memory["physics_consistent"]
        and memory["mean_peak_timing_error_s"] < 0.90 * fallback["mean_peak_timing_error_s"]
        and memory["mean_stable_fraction"] >= fallback["mean_stable_fraction"]
    )
    if memory_better:
        fallback = memory

    return fallback, f"Exact equation (2) not identifiable from current data; best fallback model is {fallback['label']}"


def build_identifiability_payload(
    exact_train_segments: list[pd.DataFrame],
    inclusion_rows: list[dict],
) -> dict:
    context = "build_identifiability_payload"
    overall_started = timing_start(context, "overall")
    if not exact_train_segments:
        timing_end(context, "overall", overall_started)
        return {"reason": "no theta-usable training segments"}
    feature_started = timing_start(context, "feature_construction")
    train_df = pd.concat(exact_train_segments, ignore_index=True)
    feature_df = build_feature_table(train_df, ["tau", "sigmaN", "sigmaN_logV", "sigmaN_logTheta"])
    exact_design_df = exact_rsf_design_table(train_df)
    target = train_df["dV_dt"].to_numpy(dtype=float)
    timing_end(context, "feature_construction", feature_started)
    diagnostics_started = timing_start(context, "diagnostics")
    diagnostics = compute_design_diagnostics(feature_df, target)
    correlation_df = exact_design_df.corr().round(6)
    design = np.column_stack(
        [
            np.ones(len(exact_design_df), dtype=float),
            exact_design_df["tau"].to_numpy(dtype=float),
            exact_design_df["sigmaN_logV"].to_numpy(dtype=float),
            exact_design_df["sigmaN_logTheta"].to_numpy(dtype=float),
        ]
    )
    singular_values = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    rank_default = int(np.linalg.matrix_rank(design))
    rank_tight = int(np.linalg.matrix_rank(design, tol=max(singular_values[0] * 1e-8, 1e-10)))
    sigma_design = np.column_stack([np.ones(len(exact_design_df), dtype=float), exact_design_df["sigmaN"].to_numpy(dtype=float)])
    sigma_singular = np.linalg.svd(sigma_design, full_matrices=False, compute_uv=False)
    sigma_rank = int(np.linalg.matrix_rank(sigma_design))

    train_inclusion_df = pd.DataFrame(inclusion_rows)
    train_inclusion_df = train_inclusion_df.loc[train_inclusion_df.get("split", pd.Series(dtype=str)) == "train"].copy()
    event_valid_steps = int(train_inclusion_df.get("theta_event_valid", pd.Series(dtype=bool)).fillna(False).sum())
    sample_valid_steps = int(train_inclusion_df.get("theta_sample_valid", pd.Series(dtype=bool)).fillna(False).sum())
    event_valid_rows = int(train_inclusion_df.loc[train_inclusion_df.get("theta_event_valid", pd.Series(dtype=bool)).fillna(False), "n_rows_after_clean"].sum()) if "n_rows_after_clean" in train_inclusion_df.columns else 0
    sample_valid_rows = int(train_inclusion_df.get("theta_usable_samples", pd.Series(dtype=float)).fillna(0).sum())

    diagnostics["requested_feature_stats"] = {column: column_stats(exact_design_df[column].to_numpy(dtype=float)) for column in exact_design_df.columns}
    diagnostics["exact_rsf_correlation_matrix"] = correlation_df.to_dict()
    diagnostics["exact_rsf_pairwise_correlations"] = [
        {"feature_a": row_name, "feature_b": column_name, "correlation": float(correlation_df.loc[row_name, column_name])}
        for row_name in correlation_df.index
        for column_name in correlation_df.columns
        if row_name < column_name
    ]
    diagnostics["exact_rsf_condition_number"] = float(np.linalg.cond(design))
    diagnostics["exact_rsf_rank_default_tol"] = rank_default
    diagnostics["exact_rsf_rank_tight_tol"] = rank_tight
    diagnostics["exact_rsf_n_columns"] = int(design.shape[1])
    diagnostics["exact_rsf_singular_values"] = [float(value) for value in singular_values]
    diagnostics["smallest_to_largest_singular_ratio"] = float(singular_values[-1] / singular_values[0]) if singular_values[0] > 0 else float("nan")
    diagnostics["near_rank_deficient"] = bool(rank_tight < design.shape[1] or diagnostics["smallest_to_largest_singular_ratio"] < 1e-8)
    diagnostics["intercept_sigmaN_condition_number"] = float(np.linalg.cond(sigma_design))
    diagnostics["intercept_sigmaN_rank"] = sigma_rank
    diagnostics["intercept_sigmaN_redundant"] = bool(sigma_rank < sigma_design.shape[1] or diagnostics.get("sigma_cv", float("inf")) < 0.01)
    diagnostics["theta_variation_too_weak_after_filtering"] = bool(
        exact_design_df["logThetaV0_over_Dc"].std() < 0.15
        or diagnostics.get("theta_residual_fraction_after_baseline", float("inf")) < 0.10
    )
    diagnostics["event_vs_sample_theta_screening"] = {
        "event_valid_train_steps": event_valid_steps,
        "sample_valid_train_steps": sample_valid_steps,
        "event_valid_train_rows": event_valid_rows,
        "sample_valid_train_rows": sample_valid_rows,
        "event_minus_sample_gap_rows": max(event_valid_rows - sample_valid_rows, 0),
        "event_minus_sample_gap_fraction": float(max(event_valid_rows - sample_valid_rows, 0) / max(event_valid_rows, 1)),
        "event_level_rejection_too_coarse": bool(max(event_valid_rows - sample_valid_rows, 0) / max(event_valid_rows, 1) > 0.25),
    }
    timing_end(context, "diagnostics", diagnostics_started)
    packaging_started = timing_start(context, "report_packaging")
    diagnostics["usable_samples_per_event"] = [
        {
            "step_name": str(df["step_name"].iloc[0]),
            "usable_samples": int(len(df)),
            "time_start": float(df["time"].iloc[0]),
            "time_end": float(df["time"].iloc[-1]),
        }
        for df in exact_train_segments
    ]
    diagnostics["total_usable_samples"] = int(len(train_df))
    diagnostics["inclusion_rows"] = inclusion_rows
    diagnostics["hard_diagnosis"] = assess_primary_diagnosis(inclusion_rows, exact_design_df, target, diagnostics)
    timing_end(context, "report_packaging", packaging_started)
    timing_end(context, "overall", overall_started)
    return diagnostics


def write_identifiability_report(payload: dict) -> None:
    if payload.get("reason"):
        text = "# Proposal Equation Identifiability Report\n\nNo identifiability diagnostics were computed because no theta-usable training segments were available.\n"
        (RESULTS_DIR / "proposal_equation_identifiability_report.md").write_text(text, encoding="utf-8")
        return

    per_event_df = pd.DataFrame(payload["usable_samples_per_event"])
    feature_stats_df = pd.DataFrame(
        [{"feature": column, **stats} for column, stats in payload.get("requested_feature_stats", {}).items()]
    )
    corr_df = pd.DataFrame(payload["exact_rsf_correlation_matrix"])
    vif_df = pd.DataFrame(payload["vif"])
    hard_diagnosis = payload.get("hard_diagnosis", {})
    diagnosis_flags = hard_diagnosis.get("flags", {})
    screening = payload.get("event_vs_sample_theta_screening", {})

    summary_lines = [
        "# Proposal Equation Identifiability Report",
        "",
        "## Usable sample counts",
        markdown_table(per_event_df, index=False),
        "",
        f"Total usable theta-mask training samples: `{payload['total_usable_samples']}`",
        "",
        "## Feature dynamic range",
        markdown_table(feature_stats_df, index=False),
        "",
        "## Pairwise correlations",
        markdown_table(corr_df, index=True),
        "",
        "## Multicollinearity diagnostics",
        f"- Condition number for [1, tau, sigmaN*log(V/V0), sigmaN*log(theta*V0/Dc)]: `{payload['exact_rsf_condition_number']:.6e}`",
        f"- Singular values: `{json.dumps(payload.get('exact_rsf_singular_values', []))}`",
        f"- Default numerical rank: `{payload.get('exact_rsf_rank_default_tol', 'NA')}` / `{payload.get('exact_rsf_n_columns', 'NA')}` columns",
        f"- Tight-tolerance numerical rank: `{payload.get('exact_rsf_rank_tight_tol', 'NA')}` / `{payload.get('exact_rsf_n_columns', 'NA')}` columns",
        f"- Near-rank-deficient: `{payload.get('near_rank_deficient')}`",
        markdown_table(vif_df, index=False),
        "",
        "## Intercept and sigmaN redundancy",
        f"- Intercept + sigmaN condition number: `{payload.get('intercept_sigmaN_condition_number', float('nan')):.6e}`",
        f"- Intercept + sigmaN rank: `{payload.get('intercept_sigmaN_rank', 'NA')}`",
        f"- Intercept + sigmaN redundant: `{payload.get('intercept_sigmaN_redundant')}`",
        "",
        "## Theta screening comparison",
        f"- Event-valid training steps: `{screening.get('event_valid_train_steps', 'NA')}`",
        f"- Sample-valid training steps: `{screening.get('sample_valid_train_steps', 'NA')}`",
        f"- Event-valid training rows: `{screening.get('event_valid_train_rows', 'NA')}`",
        f"- Sample-valid training rows: `{screening.get('sample_valid_train_rows', 'NA')}`",
        f"- Event-to-sample row gap fraction: `{screening.get('event_minus_sample_gap_fraction', float('nan')):.6e}`",
        f"- Event-level rejection too coarse: `{screening.get('event_level_rejection_too_coarse')}`",
        "",
        "## Interpretation",
        f"- SigmaN coefficient of variation within the theta-usable training matrix: `{payload.get('sigma_cv', float('nan')):.6e}`",
        f"- Residual sigmaN variation after projecting onto the intercept: `{payload.get('sigma_residual_fraction_after_intercept', float('nan')):.6e}` of original std",
        f"- Residual theta-feature variation after projecting onto the intercept: `{payload.get('theta_residual_fraction_after_intercept', float('nan')):.6e}` of original std",
        f"- Residual theta-feature variation after projecting onto [1, tau, sigmaN, sigmaN_logV]: `{payload.get('theta_residual_fraction_after_baseline', float('nan')):.6e}` of original std",
        f"- Partial correlation between residualized theta feature and residualized dV/dt: `{payload.get('theta_partial_correlation_with_dVdt', float('nan')):.6e}`",
        f"- Theta variation too weak after filtering: `{payload.get('theta_variation_too_weak_after_filtering')}`",
        "",
        "## Hard diagnosis",
        f"- Summary: `{hard_diagnosis.get('summary', 'NA')}`",
        f"- Implementation bug: `{diagnosis_flags.get('implementation_bug')}`",
        f"- Alignment / units bug: `{diagnosis_flags.get('alignment_or_units_bug')}`",
        f"- Over-filtering: `{diagnosis_flags.get('over_filtering')}`",
        f"- Insufficient theta variation: `{diagnosis_flags.get('insufficient_theta_variation')}`",
        f"- Multicollinearity / structural non-identifiability: `{diagnosis_flags.get('multicollinearity_or_structural_non_identifiability')}`",
        "",
        "## Diagnosis notes",
        "- The theta term collapses when its residual variation, after removing the baseline RSF predictors, is too small to support a stable independent coefficient.",
        "- If sigmaN is nearly constant, then sigmaN behaves almost like an intercept and cannot be separated cleanly from mu0-driven terms.",
        "- Large VIF, large condition number, or an effectively low-rank exact-RSF matrix indicate structural non-identifiability on the current Utah FORGE subsets.",
    ]
    (RESULTS_DIR / "proposal_equation_identifiability_report.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def write_model_comparison_csv(model_results: dict[str, dict], acoustic_name: str | None) -> pd.DataFrame:
    rows = []
    for label in MODEL_ORDER:
        if label not in model_results:
            continue
        best = model_results[label]["best"]
        rows.append(
            {
                "model": label,
                "equation": best["equation"],
                "n_terms": len(best["active_terms"]),
                "holdout_derivative_mse": best["holdout_mse"],
                "holdout_derivative_rmse": best["holdout_rmse"],
                "holdout_derivative_mae": best["holdout_mae"],
                "holdout_derivative_r2": best["holdout_r2"],
                "rollout_mse": best["mean_rollout_mse"],
                "stable_fraction": best["mean_stable_fraction"],
                "peak_timing_error_s": best["mean_peak_timing_error_s"],
                "onset_timing_error_s": best["mean_onset_timing_error_s"],
                "timing_error_s": best["mean_timing_error_s"],
                "physics_consistent": best["physics_consistent"],
                "theta_term_active": best["theta_term_active"],
                "acoustic_term_active": best["acoustic_term_active"],
                "acoustic_name": acoustic_name or "",
            }
        )
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(RESULTS_DIR / "proposal_equation_model_comparison.csv", index=False)
    return comparison_df


def write_dataset_summary(outputs: dict[str, list[pd.DataFrame]], inclusion_rows: list[dict], acoustic_name: str | None) -> None:
    inclusion_df = pd.DataFrame(inclusion_rows)
    inclusion_df.to_csv(RESULTS_DIR / "proposal_equation_theta_quality_diagnostics.csv", index=False)
    split_counts = {}
    for split_name in ["tau_train", "tau_holdout", "theta_event_train", "theta_event_holdout", "theta_sample_train", "theta_sample_holdout", "all_train", "all_holdout"]:
        split_counts[split_name] = {
            "n_segments": int(len(outputs.get(split_name, []))),
            "n_rows": int(sum(len(df) for df in outputs.get(split_name, []))),
        }
    summary = {
        "acoustic_feature_candidate": acoustic_name,
        "split_counts": split_counts,
        "theta_quality_summary": {
            "n_steps": int(len(inclusion_df)),
            "train_theta_event_valid_steps": int(inclusion_df.loc[inclusion_df.get("split", pd.Series(dtype=str)) == "train", "theta_event_valid"].fillna(False).sum()) if "theta_event_valid" in inclusion_df.columns else 0,
            "train_theta_sample_valid_steps": int(inclusion_df.loc[inclusion_df.get("split", pd.Series(dtype=str)) == "train", "theta_sample_valid"].fillna(False).sum()) if "theta_sample_valid" in inclusion_df.columns else 0,
            "train_theta_usable_samples": int(inclusion_df.loc[inclusion_df.get("split", pd.Series(dtype=str)) == "train", "theta_usable_samples"].fillna(0).sum()) if "theta_usable_samples" in inclusion_df.columns else 0,
        },
    }
    write_json_artifact(RESULTS_DIR / "proposal_equation_dataset_summary.json", summary)


def write_rollout_validation_json(
    tau_model: dict,
    tau_rollout_rows: list[dict],
    model_results: dict[str, dict],
) -> None:
    payload = {
        "tau_model": tau_model,
        "tau_rollout_rows": tau_rollout_rows,
        "velocity_rollouts": {label: result["best"]["rollout_rows"] for label, result in model_results.items()},
        "velocity_derivative_rows": {label: result["best"]["holdout_derivative_rows"] for label, result in model_results.items()},
    }
    write_json_artifact(RESULTS_DIR / "proposal_equation_rollout_validation.json", payload)


def build_diagnostic_figure(
    tau_model: dict,
    final_velocity_model: dict,
    tau_train_df: pd.DataFrame,
    velocity_train_df: pd.DataFrame,
    identifiability_payload: dict,
    comparison_df: pd.DataFrame,
) -> None:
    tau_coeff = tau_model["coefficients_physical"]
    tau_prediction = (
        tau_coeff.get("1", 0.0)
        + tau_coeff.get("V", 0.0) * tau_train_df["V"].to_numpy(dtype=float)
        + tau_coeff.get("V_drive_minus_V", 0.0) * (tau_train_df["V_drive"] - tau_train_df["V"]).to_numpy(dtype=float)
    )
    tau_observed = tau_train_df["dtau_dt"].to_numpy(dtype=float)

    velocity_prediction = predict_with_coefficients(velocity_train_df, final_velocity_model["coefficients_physical"], final_velocity_model["feature_names"])
    velocity_observed = velocity_train_df["dV_dt"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].scatter(tau_observed, tau_prediction, s=8, alpha=0.4)
    tau_min = float(min(np.min(tau_observed), np.min(tau_prediction)))
    tau_max = float(max(np.max(tau_observed), np.max(tau_prediction)))
    axes[0, 0].plot([tau_min, tau_max], [tau_min, tau_max], color="tab:red", linestyle="--")
    axes[0, 0].set_title("Tau derivative fit")
    axes[0, 0].set_xlabel("observed dtau/dt")
    axes[0, 0].set_ylabel("fitted dtau/dt")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(velocity_observed, velocity_prediction, s=8, alpha=0.4)
    v_min = float(min(np.min(velocity_observed), np.min(velocity_prediction)))
    v_max = float(max(np.max(velocity_observed), np.max(velocity_prediction)))
    axes[0, 1].plot([v_min, v_max], [v_min, v_max], color="tab:red", linestyle="--")
    axes[0, 1].set_title(f"Velocity derivative fit ({final_velocity_model['label']})")
    axes[0, 1].set_xlabel("observed dV/dt")
    axes[0, 1].set_ylabel("fitted dV/dt")
    axes[0, 1].grid(True, alpha=0.3)

    if "correlation_matrix" in identifiability_payload:
        corr_df = pd.DataFrame(identifiability_payload["correlation_matrix"])
        image = axes[1, 0].imshow(corr_df.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axes[1, 0].set_xticks(range(len(corr_df.columns)))
        axes[1, 0].set_xticklabels(corr_df.columns, rotation=20)
        axes[1, 0].set_yticks(range(len(corr_df.index)))
        axes[1, 0].set_yticklabels(corr_df.index)
        axes[1, 0].set_title("Exact-RSF feature correlations")
        fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        axes[1, 0].axis("off")

    axes[1, 1].axis("off")
    summary_lines = [
        f"Final velocity model: {final_velocity_model['label']}",
        final_velocity_model["equation"],
        f"stable_fraction={final_velocity_model['mean_stable_fraction']:.3f}",
        f"peak_timing_error_s={final_velocity_model['mean_peak_timing_error_s']:.3f}",
        f"theta_term_active={final_velocity_model['theta_term_active']}",
        f"physics_consistent={final_velocity_model['physics_consistent']}",
        "",
        "Model comparison:",
    ]
    for _, row in comparison_df.iterrows():
        summary_lines.append(
            f"{row['model']}: mse={row['holdout_derivative_mse']:.3e}, stable={row['stable_fraction']:.3f}, peak={row['peak_timing_error_s']:.3f}s"
        )
    axes[1, 1].text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "proposal_equation_recovery_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    insertion_note: dict[str, str],
    inclusion_rows: list[dict],
    tau_model: dict,
    model_results: dict[str, dict],
    final_velocity_model: dict,
    final_conclusion: str,
    tau_rollout_rows: list[dict],
    comparison_df: pd.DataFrame,
    identifiability_payload: dict,
    acoustic_name: str | None,
) -> None:
    previous_models = parse_previous_reports()
    inclusion_df = pd.DataFrame(inclusion_rows)
    total_steps = int(len(inclusion_df))
    theta_event_steps = int(inclusion_df.get("theta_event_valid", pd.Series(dtype=bool)).fillna(False).sum())
    theta_sample_steps = int(inclusion_df.get("theta_sample_valid", pd.Series(dtype=bool)).fillna(False).sum())
    theta_sample_rows = int(inclusion_df.get("theta_usable_samples", pd.Series(dtype=float)).fillna(0).sum())

    exact_model = model_results["A_exact_rsf"]["best"]
    reduced_model = model_results["B_reduced_rsf"]["best"]
    memory_model = model_results["C_local_memory"]["best"]
    acoustic_model = model_results["D_acoustic_augmented"]["best"] if "D_acoustic_augmented" in model_results else None

    theta_collapse_reason = {
        "theta_term_active": exact_model["theta_term_active"],
        "theta_coefficient": exact_model["coefficients_physical"].get("sigmaN_logTheta", 0.0),
        "theta_residual_fraction_after_baseline": identifiability_payload.get("theta_residual_fraction_after_baseline"),
        "theta_partial_correlation_with_dVdt": identifiability_payload.get("theta_partial_correlation_with_dVdt"),
        "condition_number": identifiability_payload.get("condition_number"),
        "sigma_cv": identifiability_payload.get("sigma_cv"),
    }

    output_json = {
        "insertion_note": insertion_note,
        "inclusion_rows": inclusion_rows,
        "tau_model": tau_model,
        "velocity_models": model_results,
        "final_velocity_model": final_velocity_model,
        "tau_rollout_rows": tau_rollout_rows,
        "identifiability": identifiability_payload,
        "previous_models": previous_models,
        "summary": {
            "total_steps": total_steps,
            "theta_event_valid_steps": theta_event_steps,
            "theta_sample_valid_steps": theta_sample_steps,
            "theta_sample_valid_rows": theta_sample_rows,
            "theta_collapse_reason": theta_collapse_reason,
            "acoustic_feature_used": acoustic_name,
            "final_conclusion": final_conclusion,
        },
    }
    (RESULTS_DIR / "proposal_equation_recovery.json").write_text(json.dumps(output_json, indent=2), encoding="utf-8")

    equation_lines = [
        tau_model["exact_equation"],
        tau_model["one_term_equation"],
        f"Final velocity model ({final_velocity_model['label']}): {final_velocity_model['equation']}",
        f"Exact-RSF attempt: {exact_model['equation']}",
    ]
    (RESULTS_DIR / "proposal_equation_recovery_equations.txt").write_text("\n".join(equation_lines) + "\n", encoding="utf-8")

    sign_rows = []
    sign_rows.append(("tau_drive_positive", "positive", tau_model["coefficients_physical"].get("V_drive_minus_V", 0.0), tau_model["coefficients_physical"].get("V_drive_minus_V", 0.0) > 0))
    for name, ok in exact_model["sign_checks"].items():
        value_name = name.replace("_positive", "").replace("_negative", "")
        value = exact_model["coefficients_physical"].get(value_name, 0.0)
        expected = "positive" if name.endswith("_positive") else "negative"
        sign_rows.append((name, expected, value, ok))

    sign_lines = [
        "| Check | Expected | Result | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name, expected, value, ok in sign_rows:
        sign_lines.append(f"| {name} | {expected} | {value:.6e} | {'✅' if ok else '❌'} |")

    comparison_md = comparison_df.copy()
    if not comparison_md.empty:
        comparison_md["holdout_derivative_mse"] = comparison_md["holdout_derivative_mse"].map(lambda value: f"{value:.6e}")
        comparison_md["holdout_derivative_rmse"] = comparison_md["holdout_derivative_rmse"].map(lambda value: f"{value:.6e}")
        comparison_md["holdout_derivative_mae"] = comparison_md["holdout_derivative_mae"].map(lambda value: f"{value:.6e}")
        comparison_md["holdout_derivative_r2"] = comparison_md["holdout_derivative_r2"].map(lambda value: f"{value:.6f}")
        comparison_md["rollout_mse"] = comparison_md["rollout_mse"].map(lambda value: f"{value:.6e}")
        comparison_md["stable_fraction"] = comparison_md["stable_fraction"].map(lambda value: f"{value:.3f}")
        comparison_md["peak_timing_error_s"] = comparison_md["peak_timing_error_s"].map(lambda value: f"{value:.3f}")
        comparison_md["onset_timing_error_s"] = comparison_md["onset_timing_error_s"].map(lambda value: f"{value:.3f}")
        comparison_md["timing_error_s"] = comparison_md["timing_error_s"].map(lambda value: f"{value:.3f}")

    if final_conclusion.startswith("Exact equation (2) recovered credibly"):
        closing_line = "Exact equation (2) recovered credibly"
    elif final_conclusion.startswith("Exact equation (2) partially recovered"):
        closing_line = "Exact equation (2) partially recovered but still weak"
    else:
        closing_line = final_conclusion

    report_lines = [
        "# Proposal Equation Recovery Report",
        "",
        "## Best insertion point",
        f"- {insertion_note['best_insertion_point']}",
        f"- Train steps: `{insertion_note['train_steps']}`",
        f"- Holdout steps: `{insertion_note['holdout_steps']}`",
        f"- Why the current models miss the proposal target cleanly: {insertion_note['why_current_models_miss_target']}",
        "",
        "## Data inclusion / exclusion",
        f"- Total RSFit-aligned steps inspected: `{total_steps}`",
        f"- Theta-valid steps under strict event screening: `{theta_event_steps}`",
        f"- Theta-usable steps under sample masking: `{theta_sample_steps}`",
        f"- Theta-usable samples under sample masking: `{theta_sample_rows}`",
        "",
        "## Equation (1) recovery",
        f"- Exact fit: `{tau_model['exact_equation']}`",
        f"- Closest one-term fit: `{tau_model['one_term_equation']}`",
        f"- Holdout derivative MSE: `{tau_model['holdout_mse']:.6e}`",
        "",
        "## Equation (2) model ladder",
        markdown_table(comparison_md, index=False) if not comparison_md.empty else "_No model comparison rows available._",
        "",
        "## Exact RSF attempt",
        f"- `{exact_model['equation']}`",
        f"- Theta coefficient active: `{exact_model['theta_term_active']}`",
        f"- Exact-model holdout derivative MSE: `{exact_model['holdout_mse']:.6e}`",
        f"- Exact-model holdout derivative RMSE: `{exact_model['holdout_rmse']:.6e}`",
        f"- Exact-model holdout derivative MAE: `{exact_model['holdout_mae']:.6e}`",
        f"- Exact-model holdout derivative R^2: `{exact_model['holdout_r2']:.6f}`",
        f"- Exact-model mean stable rollout fraction: `{exact_model['mean_stable_fraction']:.3f}`",
        f"- Exact-model mean peak timing error: `{exact_model['mean_peak_timing_error_s']:.3f}` s",
        f"- Exact-model mean onset timing error: `{exact_model['mean_onset_timing_error_s']:.3f}` s",
        "",
        "## Final selected velocity model",
        f"- Selected model: `{final_velocity_model['label']}`",
        f"- `{final_velocity_model['equation']}`",
        f"- Holdout derivative MSE: `{final_velocity_model['holdout_mse']:.6e}`",
        f"- Holdout derivative RMSE: `{final_velocity_model['holdout_rmse']:.6e}`",
        f"- Holdout derivative MAE: `{final_velocity_model['holdout_mae']:.6e}`",
        f"- Holdout derivative R^2: `{final_velocity_model['holdout_r2']:.6f}`",
        f"- Mean stable rollout fraction: `{final_velocity_model['mean_stable_fraction']:.3f}`",
        f"- Mean peak timing error: `{final_velocity_model['mean_peak_timing_error_s']:.3f}` s",
        f"- Mean onset timing error: `{final_velocity_model['mean_onset_timing_error_s']:.3f}` s",
        "",
        "## Sign checks",
        *sign_lines,
        "",
        "## De-normalized coefficients",
        f"- Tau coefficients: `{json.dumps(tau_model['coefficients_physical'])}`",
        f"- Final velocity coefficients: `{json.dumps(final_velocity_model['coefficients_physical'])}`",
        "",
        "## Parameter mapping",
        f"- `m = 1 / beta_tau = {final_velocity_model['m_hat']:.6e}`",
        f"- `mu0 = -beta_sigma / beta_tau = {final_velocity_model['mu0_hat']:.6e}`",
        f"- `a = -beta_sigmaV / beta_tau = {final_velocity_model['a_hat']:.6e}`",
        f"- `b = -beta_sigmaTheta / beta_tau = {final_velocity_model['b_hat']:.6e}`",
        "- For reduced or surrogate models, only the parameters attached to present coefficients are interpretable.",
        "",
        "## Why the theta term collapsed",
        f"- Exact-model theta coefficient: `{exact_model['coefficients_physical'].get('sigmaN_logTheta', 0.0):.6e}`",
        f"- Theta residual fraction after removing [1, tau, sigmaN, sigmaN_logV]: `{identifiability_payload.get('theta_residual_fraction_after_baseline', float('nan')):.6e}`",
        f"- Partial correlation of residualized theta term with residualized dV/dt: `{identifiability_payload.get('theta_partial_correlation_with_dVdt', float('nan')):.6e}`",
        f"- Exact-design condition number: `{identifiability_payload.get('condition_number', float('nan')):.6e}`",
        f"- SigmaN coefficient of variation in the exact design: `{identifiability_payload.get('sigma_cv', float('nan')):.6e}`",
        "",
        "## Validation",
        f"- Semi-observed tau rollout rows: `{json.dumps(tau_rollout_rows)}`",
        f"- Exact-RSF rollout rows: `{json.dumps(exact_model['rollout_rows'])}`",
        f"- Reduced-RSF rollout rows: `{json.dumps(reduced_model['rollout_rows'])}`",
        f"- Local-memory rollout rows: `{json.dumps(memory_model['rollout_rows'])}`",
        (f"- Acoustic rollout rows: `{json.dumps(acoustic_model['rollout_rows'])}`" if acoustic_model is not None else "- Acoustic rollout rows: `not available`"),
        "",
        "## Comparison against previous surrogate-heavy paths",
        f"- Previous memory-model divergence: `{previous_models.get('memory_divergence', 'NA')}` s",
        f"- Previous delay-model theta correlation: `{previous_models.get('delay_theta_corr', 'NA')}`",
        f"- Previous ablation Model B divergence: `{previous_models.get('model_b_div', 'NA')}` s",
        f"- Previous ablation Model C divergence: `{previous_models.get('model_c_div', 'NA')}` s",
        "",
        "## Acoustic feature test",
        f"- Acoustic feature tested: `{acoustic_name or 'none available'}`",
        (
            f"- Acoustic model derivative MSE changed from `{reduced_model['holdout_mse']:.6e}` to `{acoustic_model['holdout_mse']:.6e}` and peak timing error from `{reduced_model['mean_peak_timing_error_s']:.3f}` s to `{acoustic_model['mean_peak_timing_error_s']:.3f}` s."
            if acoustic_model is not None
            else "- No aligned acoustic feature was available in the loaded state frame."
        ),
        (
            f"- Acoustic residual std fraction after removing [1, tau, sigmaN, sigmaN*log(V/V0)]: `{acoustic_model['acoustic_incremental_diagnostic'].get('acoustic_residual_std_fraction', float('nan')):.6e}`; residual partial correlation with dV/dt: `{acoustic_model['acoustic_incremental_diagnostic'].get('acoustic_partial_correlation_with_dVdt', float('nan')):.6e}`."
            if acoustic_model is not None and acoustic_model.get("acoustic_incremental_diagnostic") is not None
            else "- Acoustic incremental-information diagnostic: `not available`"
        ),
        "",
        "## Conclusion",
        f"- `{closing_line}`",
    ]
    (RESULTS_DIR / "proposal_equation_recovery_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    prepared_checkpoint = load_pickle_checkpoint("prepared_segments")
    if prepared_checkpoint is None:
        outputs, inclusion_rows, insertion_note, acoustic_name = prepare_all_segments()
        prepared_checkpoint = {
            "outputs": outputs,
            "inclusion_rows": inclusion_rows,
            "insertion_note": insertion_note,
            "acoustic_name": acoustic_name,
        }
        save_pickle_checkpoint(
            "prepared_segments",
            prepared_checkpoint,
            {
                "acoustic_name": acoustic_name,
                "n_tau_train_segments": len(outputs["tau_train"]),
                "n_tau_holdout_segments": len(outputs["tau_holdout"]),
                "n_theta_sample_train_segments": len(outputs["theta_sample_train"]),
                "n_theta_sample_holdout_segments": len(outputs["theta_sample_holdout"]),
            },
        )
    else:
        print("[resume] loaded prepared_segments checkpoint", flush=True)
    outputs = prepared_checkpoint["outputs"]
    inclusion_rows = prepared_checkpoint["inclusion_rows"]
    insertion_note = prepared_checkpoint["insertion_note"]
    acoustic_name = prepared_checkpoint["acoustic_name"]
    write_dataset_summary(outputs, inclusion_rows, acoustic_name)
    if not outputs["tau_train"] or not outputs["tau_holdout"]:
        raise RuntimeError("No usable Utah FORGE step windows were prepared for tau recovery.")
    if not outputs["theta_sample_train"] or not outputs["theta_sample_holdout"]:
        print("Warning: theta sample-masked subsets are empty; falling back to event-valid theta segments for the exact RSF ladder rung.", flush=True)

    tau_checkpoint = load_pickle_checkpoint("tau_recovery")
    if tau_checkpoint is None:
        tau_model = fit_tau_recovery(outputs["tau_train"], outputs["tau_holdout"])
        tau_rollout_rows = semi_observed_tau_rollout(tau_model, outputs["tau_holdout"])
        tau_checkpoint = {"tau_model": tau_model, "tau_rollout_rows": tau_rollout_rows}
        save_pickle_checkpoint(
            "tau_recovery",
            tau_checkpoint,
            {
                "tau_equation": tau_model["exact_equation"],
                "tau_holdout_mse": tau_model["holdout_mse"],
                "tau_rollout_rows": tau_rollout_rows,
            },
        )
    else:
        print("[resume] loaded tau_recovery checkpoint", flush=True)
    tau_model = tau_checkpoint["tau_model"]
    tau_rollout_rows = tau_checkpoint["tau_rollout_rows"]

    all_train_orth, all_holdout_orth, deltaS_orth_map = add_deltaS_orth(outputs["all_train"], outputs["all_holdout"])
    acoustic_available = acoustic_name is not None and any(np.isfinite(pd.concat(outputs["all_train"], ignore_index=True)["acoustic_feature"].to_numpy(dtype=float)))

    ident_checkpoint = load_pickle_checkpoint("identifiability")
    exact_train_for_ident = outputs["theta_sample_train"] if outputs["theta_sample_train"] else outputs["theta_event_train"]
    if ident_checkpoint is None:
        identifiability_payload = build_identifiability_payload(exact_train_for_ident, inclusion_rows)
        ident_checkpoint = {"identifiability_payload": identifiability_payload}
        save_pickle_checkpoint(
            "identifiability",
            ident_checkpoint,
            {
                "total_usable_samples": identifiability_payload.get("total_usable_samples"),
                "hard_diagnosis": identifiability_payload.get("hard_diagnosis", {}),
            },
        )
    else:
        print("[resume] loaded identifiability checkpoint", flush=True)
        identifiability_payload = ident_checkpoint["identifiability_payload"]
    write_identifiability_report(identifiability_payload)
    write_json_artifact(RESULTS_DIR / "proposal_equation_identifiability.json", identifiability_payload)

    ladder_checkpoint = load_pickle_checkpoint("model_ladder")
    if ladder_checkpoint is None:
        ladder_checkpoint = {"model_results": {}}
    else:
        print("[resume] loaded model_ladder checkpoint", flush=True)
    model_results: dict[str, dict] = ladder_checkpoint.get("model_results", {})
    for model_def in model_feature_definitions(acoustic_available):
        dataset_key = model_def["dataset_key"]
        if dataset_key == "theta_sample":
            train_segments = outputs["theta_sample_train"] if outputs["theta_sample_train"] else outputs["theta_event_train"]
            holdout_segments = outputs["theta_sample_holdout"] if outputs["theta_sample_holdout"] else outputs["theta_event_holdout"]
        elif dataset_key == "all":
            train_segments = outputs["all_train"]
            holdout_segments = outputs["all_holdout"]
        elif dataset_key == "all_orth":
            train_segments = all_train_orth
            holdout_segments = all_holdout_orth
            model_def = {**model_def, "deltaS_orth_map": deltaS_orth_map}
        else:
            raise KeyError(f"Unsupported dataset key: {dataset_key}")
        if not train_segments or not holdout_segments:
            print(f"Skipping model {model_def['label']}: missing train or holdout segments.", flush=True)
            continue
        if model_def["label"] in model_results:
            print(f"[resume] skipping {model_def['label']} because it already has a checkpointed result", flush=True)
            continue
        model_results[model_def["label"]] = fit_velocity_model(train_segments, holdout_segments, model_def)
        save_pickle_checkpoint(
            "model_ladder",
            {"model_results": model_results},
            {
                "completed_models": list(model_results.keys()),
                "acoustic_available": acoustic_available,
            },
        )
        write_model_comparison_csv(model_results, acoustic_name)

    if "A_exact_rsf" not in model_results or "B_reduced_rsf" not in model_results or "C_local_memory" not in model_results:
        raise RuntimeError("The proposal recovery model ladder could not be completed.")

    final_velocity_model, final_conclusion = choose_final_velocity_model(model_results)
    comparison_df = write_model_comparison_csv(model_results, acoustic_name)
    write_rollout_validation_json(tau_model, tau_rollout_rows, model_results)

    tau_train_df = pd.concat(outputs["tau_train"], ignore_index=True)
    if final_velocity_model["label"] == "A_exact_rsf":
        exact_plot_segments = outputs["theta_sample_train"] if outputs["theta_sample_train"] else outputs["theta_event_train"]
        velocity_train_df = pd.concat(exact_plot_segments, ignore_index=True)
    elif final_velocity_model["label"] == "C_local_memory":
        velocity_train_df = pd.concat(all_train_orth, ignore_index=True)
    else:
        velocity_train_df = pd.concat(outputs["all_train"], ignore_index=True)
    build_diagnostic_figure(tau_model, final_velocity_model, tau_train_df, velocity_train_df, identifiability_payload, comparison_df)

    write_outputs(
        insertion_note=insertion_note,
        inclusion_rows=inclusion_rows,
        tau_model=tau_model,
        model_results=model_results,
        final_velocity_model=final_velocity_model,
        final_conclusion=final_conclusion,
        tau_rollout_rows=tau_rollout_rows,
        comparison_df=comparison_df,
        identifiability_payload=identifiability_payload,
        acoustic_name=acoustic_name,
    )
    save_pickle_checkpoint(
        "final_package",
        {
            "tau_model": tau_model,
            "tau_rollout_rows": tau_rollout_rows,
            "model_results": model_results,
            "final_velocity_model": final_velocity_model,
            "final_conclusion": final_conclusion,
            "identifiability_payload": identifiability_payload,
        },
        {
            "final_velocity_model": final_velocity_model["label"],
            "final_conclusion": final_conclusion,
        },
    )

    print("Proposal-equation recovery complete.", flush=True)
    print(
        json.dumps(
            {
                "tau_equation": tau_model["exact_equation"],
                "tau_one_term": tau_model["one_term_equation"],
                "exact_velocity_equation": model_results["A_exact_rsf"]["best"]["equation"],
                "final_velocity_model": final_velocity_model["label"],
                "final_velocity_equation": final_velocity_model["equation"],
                "final_conclusion": final_conclusion,
                "acoustic_feature": acoustic_name,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"proposal-equation recovery failed: {exc}", file=sys.stderr, flush=True)
        raise
