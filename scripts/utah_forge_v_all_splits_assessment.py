from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utah_forge_proposal_equation_recovery as proposal
from scripts import utah_forge_showcase_fit_visuals as showcase
from src.derivatives import derivative_savgol
from src.exact_rsf import (
    fit_exact_rsf_inverse_model,
    load_checkpoint,
    load_workflow_context,
    pack_initial_vector,
    prepare_exact_segments,
    simulate_exact_rsf_segment,
)
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
PREPARED_TAU_CHECKPOINT = RESULTS_DIR / "proposal_equation_checkpoints" / "prepared_segments.pkl"
EXACT_SPLIT_CHECKPOINT_DIR = ensure_directory(RESULTS_DIR / "v_exact_all_splits_checkpoints")

OUTPUT_MD = RESULTS_DIR / "v_all_splits_assessment.md"
OUTPUT_JSON = RESULTS_DIR / "v_all_splits_assessment.json"
OUTPUT_TABLE = RESULTS_DIR / "v_all_splits_table.csv"
STEP_TABLE = RESULTS_DIR / "v_step_difficulty_table.csv"
PAIR_TABLE = RESULTS_DIR / "v_pair_difficulty_table.csv"
EXACT_OUTPUT_TABLE = RESULTS_DIR / "v_exact_all_splits_table.csv"
EXACT_STEP_TABLE = RESULTS_DIR / "v_exact_step_difficulty_table.csv"
EXACT_PAIR_TABLE = RESULTS_DIR / "v_exact_pair_difficulty_table.csv"

FIG_V_SINGLE_REDUCED = RESULTS_DIR / "v_single_step_holdout_ranking_reduced.png"
FIG_V_PAIR_REDUCED = RESULTS_DIR / "v_leave_two_out_heatmap_reduced.png"
FIG_V_DIST_REDUCED = RESULTS_DIR / "v_split_distribution_summary_reduced.png"
FIG_V_EXAMPLES_REDUCED = RESULTS_DIR / "v_easy_vs_hard_examples_reduced.png"
FIG_V_CONTEXT_REDUCED = RESULTS_DIR / "v_step_context_comparison_reduced.png"

FIG_V_SINGLE_EXACT = RESULTS_DIR / "v_single_step_holdout_ranking_exact.png"
FIG_V_PAIR_EXACT = RESULTS_DIR / "v_leave_two_out_heatmap_exact.png"
FIG_V_DIST_EXACT = RESULTS_DIR / "v_split_distribution_summary_exact.png"
FIG_V_EXAMPLES_EXACT = RESULTS_DIR / "v_easy_vs_hard_examples_exact.png"

FIG_REDUCED_VS_EXACT = RESULTS_DIR / "reduced_vs_exact_v_summary.png"

ALL_STEPS = [
    "p5838_step2",
    "p5838_step3",
    "p5838_step4",
    "p5838_step5",
    "p5838_step7",
    "p5838_step8",
    "p5838_step9",
    "p5838_step10",
]
CURRENT_PAIR = ("p5838_step2", "p5838_step7")
EPS = 1e-12


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


def load_prepared_tau_map() -> dict[str, pd.DataFrame]:
    payload = pd.read_pickle(PREPARED_TAU_CHECKPOINT)
    prepared_map: dict[str, pd.DataFrame] = {}
    for key in ("all_train", "all_holdout"):
        for df in payload["outputs"][key]:
            prepared_map[str(df["step_name"].iloc[0])] = df.copy()
    return prepared_map


def reduced_model_def() -> dict:
    return {
        "label": "B_reduced_rsf",
        "feature_names": ["1", "tau", "sigmaN", "sigmaN_logV"],
        "mandatory_terms": {"tau", "sigmaN", "sigmaN_logV"},
        "bounds": {"tau": (0.0, np.inf), "sigmaN": (-np.inf, 0.0), "sigmaN_logV": (-np.inf, 0.0)},
        "ordered_terms": ["tau", "sigmaN", "sigmaN_logV"],
        "dataset_key": "all",
        "timing_mode": "observed_sigma",
    }


def fit_reduced_fixed_threshold(train_segments: list[pd.DataFrame]) -> dict:
    model_def = reduced_model_def()
    train_df = pd.concat(train_segments, ignore_index=True)
    design, scaling, raw_features = proposal.build_design_and_scaling(train_df, model_def["feature_names"])
    target = train_df["dV_dt"].to_numpy(dtype=float)
    lower = []
    upper = []
    for feature_name in model_def["feature_names"]:
        bounds = model_def["bounds"].get(feature_name, (-np.inf, np.inf))
        lower.append(bounds[0])
        upper.append(bounds[1])
    coefficients_z = proposal.constrained_stlsq(
        design,
        target,
        model_def["feature_names"],
        threshold=0.0,
        lower_bounds=np.array(lower, dtype=float),
        upper_bounds=np.array(upper, dtype=float),
        mandatory_terms=model_def["mandatory_terms"],
    )
    coefficients = proposal.denormalize_coefficients(coefficients_z, model_def["feature_names"], scaling)
    train_prediction = proposal.predict_from_feature_table(raw_features, coefficients, model_def["feature_names"])
    train_mse = float(np.mean((train_prediction - target) ** 2))
    active_terms = [name for name in model_def["feature_names"] if abs(coefficients.get(name, 0.0)) > 1e-10]
    return {
        **model_def,
        "threshold": 0.0,
        "coefficients_z": coefficients_z.tolist(),
        "coefficients_physical": coefficients,
        "active_terms": active_terms,
        "train_mse": train_mse,
        "equation": proposal.format_equation("dV/dt", coefficients, model_def["ordered_terms"]),
        "deltaS_orth_map": {},
    }


def make_starts(base_initial: np.ndarray, n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [base_initial.copy()]
    for _ in range(n_starts - 1):
        trial = base_initial.copy()
        trial[0] *= float(rng.uniform(0.5, 1.5))
        trial[1] *= float(rng.uniform(0.5, 2.0))
        trial[2] *= float(rng.uniform(0.85, 1.15))
        trial[3] *= float(rng.uniform(0.5, 1.5))
        trial[4] *= float(rng.uniform(0.5, 1.5))
        trial[5] *= float(rng.uniform(0.5, 1.5))
        if len(trial) > 6:
            trial[6:] += rng.normal(0.0, 0.2, size=len(trial) - 6)
        starts.append(trial)
    return starts


def classify_visual_quality(rmse: float, stable_fraction: float, q1: float, q2: float) -> str:
    if stable_fraction < 0.5:
        return "poor"
    if rmse <= q1 and stable_fraction >= 0.9:
        return "good"
    if rmse <= q2 and stable_fraction >= 0.75:
        return "moderate"
    return "poor"


def get_peak_onset(values: np.ndarray, time_axis: np.ndarray) -> tuple[float, float]:
    peak = proposal.peak_time(values, time_axis)
    onset = proposal.onset_time(values, time_axis)
    return onset, peak


def reduced_prediction_arrays(model_row: dict, seg: pd.DataFrame) -> dict:
    obs_sigma = seg["sigmaN"].to_numpy(dtype=float)
    obs_drive_gap = (seg["V_drive"] - seg["V"]).to_numpy(dtype=float)
    series = showcase.rollout_velocity_series(model_row, seg)
    time = series["time"]
    rel_time = series["rel_time"]
    obs_v = series["observed_v"]
    obs_tau = series["observed_tau"]
    predicted = series["predicted_v"]
    error = predicted - obs_v
    derivative_pred = series["predicted_dv"]
    derivative_true = series["observed_dv"]
    deriv_err = derivative_pred - derivative_true
    sigma_obs = float(np.std(obs_v))
    threshold = 3.0 * max(sigma_obs, 1e-6)
    divergence_index = len(obs_v)
    for index, (pred, obs) in enumerate(zip(predicted, obs_v)):
        if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
            divergence_index = index
            break
    stable_fraction = float(divergence_index / len(obs_v)) if len(obs_v) else 0.0
    return {
        "time": time,
        "rel_time": rel_time,
        "observed_v": obs_v,
        "predicted_v": predicted,
        "observed_tau": obs_tau,
        "observed_sigma": obs_sigma,
        "observed_gap": obs_drive_gap,
        "abs_v_error": np.abs(error),
        "velocity_rollout_rmse": float(np.sqrt(np.mean(error**2))),
        "velocity_rollout_mae": float(np.mean(np.abs(error))),
        "velocity_max_abs_error": float(np.max(np.abs(error))),
        "derivative_rmse": float(np.sqrt(np.mean(deriv_err**2))),
        "stable_fraction": stable_fraction,
        "onset_timing_error_s": abs(proposal.onset_time(predicted, time) - proposal.onset_time(obs_v, time)),
        "peak_timing_error_s": abs(proposal.peak_time(predicted, time) - proposal.peak_time(obs_v, time)),
    }


def load_exact_context():
    inventory_df, segments, steps, rsfit_globals = load_workflow_context()
    filtered_segments = {step: segments[step] for step in ALL_STEPS if step in segments}
    return inventory_df, filtered_segments, steps, rsfit_globals


def load_saved_exact_best() -> tuple[dict, dict]:
    summary = json.loads((RESULTS_DIR / "exact_rsf_multistart_summary.json").read_text(encoding="utf-8"))
    best_index = int(summary["best_run"]["start_index"])
    prepared = load_checkpoint(RESULTS_DIR / "exact_rsf_inverse_fit_checkpoints", "prepared_exact_segments")
    payload = load_checkpoint(RESULTS_DIR / "exact_rsf_multistart_checkpoints", f"exact_fit_multistart_{best_index}")
    if prepared is None or payload is None:
        raise RuntimeError("Missing saved exact RSF checkpoints for descriptive all-steps comparison.")
    return prepared, payload


def exact_prediction_arrays(segment, params: dict, delta_log_theta0: float, acoustic_z: float) -> dict:
    sim = simulate_exact_rsf_segment(segment, params, delta_log_theta0=delta_log_theta0, acoustic_z=acoustic_z)
    obs_v = segment.V
    pred_v = sim["V"]
    err = pred_v - obs_v
    pred_dv = derivative_savgol(pred_v, t=segment.time, window=15, polyorder=3)
    deriv_err = pred_dv - segment.dV_dt
    sigma_obs = float(np.std(obs_v))
    threshold = 3.0 * max(sigma_obs, 1e-6)
    divergence_index = len(obs_v)
    for index, (pred, obs) in enumerate(zip(pred_v, obs_v)):
        if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
            divergence_index = index
            break
    return {
        "time": segment.time,
        "rel_time": segment.time - float(segment.time[0]),
        "observed_v": obs_v,
        "predicted_v": pred_v,
        "observed_tau": segment.tau,
        "observed_sigma": segment.sigmaN,
        "observed_gap": segment.V_drive - segment.V,
        "abs_v_error": np.abs(err),
        "velocity_rollout_rmse": float(np.sqrt(np.nanmean(err**2))) if np.isfinite(err).any() else float("inf"),
        "velocity_rollout_mae": float(np.nanmean(np.abs(err))) if np.isfinite(err).any() else float("inf"),
        "velocity_max_abs_error": float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else float("inf"),
        "derivative_rmse": float(np.sqrt(np.nanmean(deriv_err**2))) if np.isfinite(deriv_err).any() else float("inf"),
        "stable_fraction": float(divergence_index / len(obs_v)) if len(obs_v) else 0.0,
    }


def evaluate_reduced_split(split_name: str, family: str, train_steps: list[str], holdout_steps: list[str], prepared_map: dict[str, pd.DataFrame]) -> dict:
    train_segments = [prepared_map[step].copy() for step in train_steps]
    best = fit_reduced_fixed_threshold(train_segments)
    rows = []
    for step in holdout_steps:
        seg = prepared_map[step]
        arr = reduced_prediction_arrays(best, seg)
        rows.append(
            {
                "step_name": step,
                "velocity_rollout_rmse": arr["velocity_rollout_rmse"],
                "velocity_rollout_mae": arr["velocity_rollout_mae"],
                "velocity_max_abs_error": arr["velocity_max_abs_error"],
                "derivative_rmse": arr["derivative_rmse"],
                "onset_timing_error_s": arr["onset_timing_error_s"],
                "peak_timing_error_s": arr["peak_timing_error_s"],
                "stable_fraction": arr["stable_fraction"],
            }
        )
    return {
        "branch": "reduced_rsf",
        "family": family,
        "split_name": split_name,
        "train_steps": train_steps,
        "holdout_steps": holdout_steps,
        "model": best,
        "holdout_rows": rows,
        "mean_velocity_rollout_rmse": float(np.mean([row["velocity_rollout_rmse"] for row in rows])),
        "mean_velocity_rollout_mae": float(np.mean([row["velocity_rollout_mae"] for row in rows])),
        "mean_velocity_max_abs_error": float(np.mean([row["velocity_max_abs_error"] for row in rows])),
        "mean_derivative_rmse": float(np.mean([row["derivative_rmse"] for row in rows])),
        "mean_onset_timing_error_s": float(np.mean([row["onset_timing_error_s"] for row in rows])),
        "mean_peak_timing_error_s": float(np.mean([row["peak_timing_error_s"] for row in rows])),
        "mean_stable_fraction": float(np.mean([row["stable_fraction"] for row in rows])),
    }


def evaluate_branch(branch: str, family_name: str, holdout_sets: list[tuple[str, ...]], prepared_map, exact_context=None) -> list[dict]:
    rows = []
    for holdout in holdout_sets:
        train_steps = [step for step in ALL_STEPS if step not in holdout]
        split_name = f"{family_name}_{'__'.join(holdout)}"
        if branch == "reduced_rsf":
            rows.append(evaluate_reduced_split(split_name, family_name, train_steps, list(holdout), prepared_map))
    return rows


def build_exact_descriptive_rows(prepared_exact: dict, payload: dict) -> tuple[list[dict], list[dict]]:
    params = payload["parameters"]
    acoustic_z_map = payload["acoustic_zscores"]
    theta_offsets = payload.get("theta_offsets_train", {})
    all_segments = list(prepared_exact["train_segments"]) + list(prepared_exact["holdout_segments"])
    per_step_rows = []
    for segment in all_segments:
        arr = exact_prediction_arrays(segment, params, float(theta_offsets.get(segment.step_name, 0.0)), float(acoustic_z_map.get(segment.step_name, 0.0)))
        onset = abs(proposal.onset_time(arr["predicted_v"], arr["time"]) - proposal.onset_time(arr["observed_v"], arr["time"]))
        peak = abs(proposal.peak_time(arr["predicted_v"], arr["time"]) - proposal.peak_time(arr["observed_v"], arr["time"]))
        per_step_rows.append(
            {
                "branch": "exact_rsf_descriptive",
                "family": "descriptive_all_steps",
                "split_name": f"descriptive_{segment.step_name}",
                "train_steps": ALL_STEPS,
                "holdout_steps": [segment.step_name],
                "payload": payload,
                "holdout_rows": [
                    {
                        "step_name": segment.step_name,
                        "velocity_rollout_rmse": arr["velocity_rollout_rmse"],
                        "velocity_rollout_mae": arr["velocity_rollout_mae"],
                        "velocity_max_abs_error": arr["velocity_max_abs_error"],
                        "derivative_rmse": arr["derivative_rmse"],
                        "onset_timing_error_s": onset,
                        "peak_timing_error_s": peak,
                        "stable_fraction": arr["stable_fraction"],
                        "combined_rollout_error": 0.0,
                    }
                ],
                "mean_velocity_rollout_rmse": arr["velocity_rollout_rmse"],
                "mean_velocity_rollout_mae": arr["velocity_rollout_mae"],
                "mean_velocity_max_abs_error": arr["velocity_max_abs_error"],
                "mean_derivative_rmse": arr["derivative_rmse"],
                "mean_onset_timing_error_s": onset,
                "mean_peak_timing_error_s": peak,
                "mean_stable_fraction": arr["stable_fraction"],
                "mean_combined_rollout_error": 0.0,
            }
        )
    pair_rows = []
    step_metric_map = {row["holdout_steps"][0]: row for row in per_step_rows}
    for pair in combinations(ALL_STEPS, 2):
        rows = [step_metric_map[pair[0]]["holdout_rows"][0], step_metric_map[pair[1]]["holdout_rows"][0]]
        pair_rows.append(
            {
                "branch": "exact_rsf_descriptive",
                "family": "descriptive_pairs_from_fixed_fit",
                "split_name": f"descriptive_pair_{pair[0]}__{pair[1]}",
                "train_steps": ALL_STEPS,
                "holdout_steps": list(pair),
                "payload": payload,
                "holdout_rows": rows,
                "mean_velocity_rollout_rmse": float(np.mean([row["velocity_rollout_rmse"] for row in rows])),
                "mean_velocity_rollout_mae": float(np.mean([row["velocity_rollout_mae"] for row in rows])),
                "mean_velocity_max_abs_error": float(np.mean([row["velocity_max_abs_error"] for row in rows])),
                "mean_derivative_rmse": float(np.mean([row["derivative_rmse"] for row in rows])),
                "mean_onset_timing_error_s": float(np.mean([row["onset_timing_error_s"] for row in rows])),
                "mean_peak_timing_error_s": float(np.mean([row["peak_timing_error_s"] for row in rows])),
                "mean_stable_fraction": float(np.mean([row["stable_fraction"] for row in rows])),
                "mean_combined_rollout_error": 0.0,
            }
        )
    return per_step_rows, pair_rows


def split_rows_to_frame(split_rows: list[dict], branch: str) -> pd.DataFrame:
    records = []
    for split in split_rows:
        for row in split["holdout_rows"]:
            rec = {
                "branch": branch,
                "family": split["family"],
                "split_name": split["split_name"],
                "train_steps": "|".join(split["train_steps"]),
                "holdout_steps": "|".join(split["holdout_steps"]),
                "step_name": row["step_name"],
                "velocity_rollout_rmse": row["velocity_rollout_rmse"],
                "velocity_rollout_mae": row["velocity_rollout_mae"],
                "velocity_max_abs_error": row["velocity_max_abs_error"],
                "derivative_rmse": row["derivative_rmse"],
                "onset_timing_error_s": row["onset_timing_error_s"],
                "peak_timing_error_s": row["peak_timing_error_s"],
                "stable_fraction": row["stable_fraction"],
                "split_mean_velocity_rollout_rmse": split["mean_velocity_rollout_rmse"],
                "split_mean_stable_fraction": split["mean_stable_fraction"],
            }
            if branch == "reduced_rsf":
                rec["equation"] = split["model"]["equation"]
                rec["combined_rollout_error"] = ""
            else:
                rec["equation"] = "exact_rsf_inverse"
                rec["combined_rollout_error"] = row["combined_rollout_error"]
            records.append(rec)
    return pd.DataFrame(records)


def summarize_step_difficulty(split_rows: list[dict], branch: str) -> pd.DataFrame:
    records = []
    for step in ALL_STEPS:
        vals = []
        for split in split_rows:
            for row in split["holdout_rows"]:
                if row["step_name"] == step:
                    vals.append(row)
        rmse = np.array([row["velocity_rollout_rmse"] for row in vals], dtype=float)
        timing = np.array([0.5 * (row["onset_timing_error_s"] + row["peak_timing_error_s"]) for row in vals], dtype=float)
        stable = np.array([row["stable_fraction"] for row in vals], dtype=float)
        records.append(
            {
                "branch": branch,
                "step_name": step,
                "n_holdout_appearances": len(vals),
                "mean_holdout_velocity_rmse": float(np.mean(rmse)),
                "median_holdout_velocity_rmse": float(np.median(rmse)),
                "std_holdout_velocity_rmse": float(np.std(rmse)),
                "mean_timing_error_s": float(np.mean(timing)),
                "mean_stable_fraction": float(np.mean(stable)),
            }
        )
    frame = pd.DataFrame(records).sort_values("mean_holdout_velocity_rmse").reset_index(drop=True)
    q1 = float(frame["mean_holdout_velocity_rmse"].quantile(1 / 3))
    q2 = float(frame["mean_holdout_velocity_rmse"].quantile(2 / 3))
    labels = []
    for value in frame["mean_holdout_velocity_rmse"]:
        if value <= q1:
            labels.append("easy")
        elif value <= q2:
            labels.append("medium")
        else:
            labels.append("hard")
    frame["difficulty_label"] = labels
    frame["difficulty_rank"] = np.arange(1, len(frame) + 1)
    return frame


def summarize_pair_difficulty(split_rows: list[dict], branch: str) -> pd.DataFrame:
    records = []
    for split in split_rows:
        if len(split["holdout_steps"]) != 2:
            continue
        records.append(
            {
                "branch": branch,
                "pair_name": " + ".join(split["holdout_steps"]),
                "step_a": split["holdout_steps"][0],
                "step_b": split["holdout_steps"][1],
                "mean_pair_velocity_rollout_rmse": split["mean_velocity_rollout_rmse"],
                "pair_velocity_rollout_mae": split["mean_velocity_rollout_mae"],
                "mean_pair_timing_error_s": 0.5 * (split["mean_onset_timing_error_s"] + split["mean_peak_timing_error_s"]),
                "mean_pair_stable_fraction": split["mean_stable_fraction"],
                "is_current_pair": tuple(split["holdout_steps"]) == CURRENT_PAIR,
            }
        )
    frame = pd.DataFrame(records).sort_values("mean_pair_velocity_rollout_rmse").reset_index(drop=True)
    frame["pair_rank"] = np.arange(1, len(frame) + 1)
    return frame


def add_visual_quality(all_table: pd.DataFrame) -> pd.DataFrame:
    out = all_table.copy()
    q1 = float(out["velocity_rollout_rmse"].quantile(1 / 3))
    q2 = float(out["velocity_rollout_rmse"].quantile(2 / 3))
    out["visual_quality"] = [
        classify_visual_quality(rmse, stable, q1, q2)
        for rmse, stable in zip(out["velocity_rollout_rmse"], out["stable_fraction"])
    ]
    return out


def plot_single_step_ranking(step_df: pd.DataFrame, out_path: Path, title: str) -> None:
    frame = step_df.sort_values("mean_holdout_velocity_rmse")
    colors = ["tab:green" if x == "easy" else "tab:orange" if x == "medium" else "tab:red" for x in frame["difficulty_label"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame["step_name"], frame["mean_holdout_velocity_rmse"], yerr=frame["std_holdout_velocity_rmse"], color=colors, capsize=3, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("mean holdout velocity rollout RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_heatmap(pair_df: pd.DataFrame, out_path: Path, title: str) -> None:
    heat = pd.DataFrame(np.nan, index=ALL_STEPS, columns=ALL_STEPS, dtype=float)
    for _, row in pair_df.iterrows():
        heat.loc[row["step_a"], row["step_b"]] = row["mean_pair_velocity_rollout_rmse"]
        heat.loc[row["step_b"], row["step_a"]] = row["mean_pair_velocity_rollout_rmse"]
    values = heat.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 0.0)
    threshold = np.nanmedian(values)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(ALL_STEPS)))
    ax.set_yticks(np.arange(len(ALL_STEPS)))
    ax.set_xticklabels(ALL_STEPS, rotation=35, ha="right")
    ax.set_yticklabels(ALL_STEPS)
    for i in range(len(ALL_STEPS)):
        for j in range(len(ALL_STEPS)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if values[i, j] > threshold else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pair mean velocity rollout RMSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_summary(split_rows: list[dict], out_path: Path, title: str) -> None:
    frame = pd.DataFrame(
        [{"family": row["family"], "rmse": row["mean_velocity_rollout_rmse"]} for row in split_rows]
    )
    families = list(dict.fromkeys(frame["family"].tolist()))
    colors = {"single_step": "#457b9d", "leave_two_out": "#e76f51"}
    bins = np.linspace(float(frame["rmse"].min()), float(frame["rmse"].max()), 16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for family in families:
        vals = frame.loc[frame["family"] == family, "rmse"].to_numpy(dtype=float)
        axes[0].hist(vals, bins=bins, alpha=0.5, label=family, color=colors.get(family))
    axes[0].legend(fontsize=8)
    axes[0].set_title("Split-level RMSE distribution")
    axes[0].grid(True, alpha=0.25)
    grouped = [frame.loc[frame["family"] == family, "rmse"].to_numpy(dtype=float) for family in families]
    label_key = "tick_labels" if matplotlib.__version__ >= "3.9" else "labels"
    axes[1].boxplot(grouped, **{label_key: families}, patch_artist=True, boxprops={"facecolor": "#d8e2dc", "alpha": 0.9}, medianprops={"color": "#7f5539"})
    for idx, vals in enumerate(grouped, start=1):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        axes[1].scatter(np.full(len(vals), idx, dtype=float) + jitter, vals, color="#7f5539", s=14, alpha=0.7)
    axes[1].set_title("Split-level RMSE by family")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def easiest_hardest_examples(split_rows: list[dict], branch: str) -> list[dict]:
    sorted_rows = sorted(split_rows, key=lambda row: row["mean_velocity_rollout_rmse"])
    chosen = sorted_rows[:2] + sorted_rows[-2:]
    return chosen


def plot_examples(split_rows: list[dict], prepared_map: dict[str, pd.DataFrame], out_path: Path, title: str, branch: str, exact_segment_map: dict[str, object] | None = None) -> None:
    chosen = easiest_hardest_examples([row for row in split_rows if row["family"] == "leave_two_out"], branch)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 3.8 * len(chosen)), sharex=False)
    if len(chosen) == 1:
        axes = np.array([axes])
    for ridx, split in enumerate(chosen):
        for cidx, step in enumerate(split["holdout_steps"][:2]):
            ax = axes[ridx, cidx]
            if branch == "reduced_rsf":
                arr = reduced_prediction_arrays(split["model"], prepared_map[step])
            else:
                seg = exact_segment_map[step]
                payload = split["payload"]
                arr = exact_prediction_arrays(seg, payload["parameters"], float(payload.get("theta_offsets_train", {}).get(step, 0.0)), float(payload["acoustic_zscores"].get(step, 0.0)))
            ax.plot(arr["rel_time"], arr["observed_v"], linewidth=1.25, label="observed V")
            ax.plot(arr["rel_time"], arr["predicted_v"], linewidth=1.05, label="predicted V")
            rmse = float(np.sqrt(np.nanmean((arr["predicted_v"] - arr["observed_v"]) ** 2)))
            ax.set_title(f"{'easy' if ridx < 2 else 'hard'}: {step}\nRMSE={rmse:.3f}")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("V")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_context_comparison(step_df: pd.DataFrame, prepared_map: dict[str, pd.DataFrame], out_path: Path) -> None:
    easiest = step_df.iloc[0]["step_name"]
    median = step_df.iloc[len(step_df) // 2]["step_name"]
    hardest = step_df.iloc[-1]["step_name"]
    steps = [easiest, median, hardest]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=False)
    for ridx, step in enumerate(steps):
        seg = prepared_map[step]
        rel_time = seg["time"].to_numpy(dtype=float) - float(seg["time"].iloc[0])
        axes[ridx, 0].plot(rel_time, seg["V"], linewidth=1.2)
        axes[ridx, 0].set_title(f"{step}: V")
        axes[ridx, 1].plot(rel_time, seg["tau"], linewidth=1.2)
        axes[ridx, 1].set_title(f"{step}: tau")
        axes[ridx, 2].plot(rel_time, seg["sigmaN"], linewidth=1.2, label="sigmaN")
        axes[ridx, 2].plot(rel_time, seg["V_drive"] - seg["V"], linewidth=1.0, label="V_drive-V")
        axes[ridx, 2].set_title(f"{step}: context")
        for ax in axes[ridx]:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("time [s]")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Reduced RSF step context: easiest vs median vs hardest V steps", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reduced_vs_exact_summary(reduced_step: pd.DataFrame, exact_step: pd.DataFrame, reduced_pairs: pd.DataFrame, exact_pairs: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(ALL_STEPS))
    axes[0].plot(x, reduced_step.set_index("step_name").loc[ALL_STEPS, "mean_holdout_velocity_rmse"], marker="o", label="Reduced RSF")
    axes[0].plot(x, exact_step.set_index("step_name").loc[ALL_STEPS, "mean_holdout_velocity_rmse"], marker="s", label="Exact RSF")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ALL_STEPS, rotation=35, ha="right")
    axes[0].set_title("Per-step mean holdout velocity RMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    pair_stats = pd.DataFrame(
        [
            {"branch": "Reduced RSF", "median_pair_rmse": reduced_pairs["mean_pair_velocity_rollout_rmse"].median(), "mean_pair_rmse": reduced_pairs["mean_pair_velocity_rollout_rmse"].mean()},
            {"branch": "Exact RSF", "median_pair_rmse": exact_pairs["mean_pair_velocity_rollout_rmse"].median(), "mean_pair_rmse": exact_pairs["mean_pair_velocity_rollout_rmse"].mean()},
        ]
    )
    width = 0.35
    idx = np.arange(len(pair_stats))
    axes[1].bar(idx - width / 2, pair_stats["median_pair_rmse"], width, label="median pair RMSE")
    axes[1].bar(idx + width / 2, pair_stats["mean_pair_rmse"], width, label="mean pair RMSE")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(pair_stats["branch"])
    axes[1].set_title("Across-pair generalization summary")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Reduced vs exact V summary across many holdout splits", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prepared_map = load_prepared_tau_map()
    single_sets = [(step,) for step in ALL_STEPS]
    pair_sets = list(combinations(ALL_STEPS, 2))

    reduced_single = evaluate_branch("reduced_rsf", "single_step", single_sets, prepared_map)
    reduced_pairs = evaluate_branch("reduced_rsf", "leave_two_out", pair_sets, prepared_map)
    prepared_exact, exact_payload = load_saved_exact_best()
    exact_single, exact_pairs = build_exact_descriptive_rows(prepared_exact, exact_payload)
    exact_segment_map = {seg.step_name: seg for seg in list(prepared_exact["train_segments"]) + list(prepared_exact["holdout_segments"])}

    reduced_all = add_visual_quality(split_rows_to_frame(reduced_single + reduced_pairs, "reduced_rsf"))
    exact_all = add_visual_quality(split_rows_to_frame(exact_single + exact_pairs, "exact_rsf"))
    reduced_step = summarize_step_difficulty(reduced_single + reduced_pairs, "reduced_rsf")
    exact_step = summarize_step_difficulty(exact_single + exact_pairs, "exact_rsf")
    reduced_pair = summarize_pair_difficulty(reduced_pairs, "reduced_rsf")
    exact_pair = summarize_pair_difficulty(exact_pairs, "exact_rsf")

    OUTPUT_TABLE.write_text(reduced_all.to_csv(index=False), encoding="utf-8")
    STEP_TABLE.write_text(reduced_step.to_csv(index=False), encoding="utf-8")
    PAIR_TABLE.write_text(reduced_pair.to_csv(index=False), encoding="utf-8")
    EXACT_OUTPUT_TABLE.write_text(exact_all.to_csv(index=False), encoding="utf-8")
    EXACT_STEP_TABLE.write_text(exact_step.to_csv(index=False), encoding="utf-8")
    EXACT_PAIR_TABLE.write_text(exact_pair.to_csv(index=False), encoding="utf-8")

    plot_single_step_ranking(reduced_step, FIG_V_SINGLE_REDUCED, "Reduced RSF: single-step holdout V difficulty")
    plot_pair_heatmap(reduced_pair, FIG_V_PAIR_REDUCED, "Reduced RSF leave-two-out V rollout RMSE")
    plot_distribution_summary(reduced_single + reduced_pairs, FIG_V_DIST_REDUCED, "Reduced RSF dynamic V rollout performance across splits")
    plot_examples(reduced_single + reduced_pairs, prepared_map, FIG_V_EXAMPLES_REDUCED, "Reduced RSF easy vs hard V holdout examples", "reduced_rsf")
    plot_context_comparison(reduced_step, prepared_map, FIG_V_CONTEXT_REDUCED)

    plot_single_step_ranking(exact_step, FIG_V_SINGLE_EXACT, "Exact RSF: single-step holdout V difficulty")
    plot_pair_heatmap(exact_pair, FIG_V_PAIR_EXACT, "Exact RSF selected leave-two-out V rollout RMSE")
    plot_distribution_summary(exact_single + exact_pairs, FIG_V_DIST_EXACT, "Exact RSF dynamic V rollout performance across evaluated splits")
    plot_examples(exact_single + exact_pairs, prepared_map, FIG_V_EXAMPLES_EXACT, "Exact RSF easy vs hard V examples from fixed saved fit", "exact_rsf", exact_segment_map=exact_segment_map)
    plot_reduced_vs_exact_summary(reduced_step, exact_step, reduced_pair, exact_pair, FIG_REDUCED_VS_EXACT)

    reduced_current_rank = int(reduced_pair.loc[reduced_pair["is_current_pair"], "pair_rank"].iloc[0])
    exact_current_rank = int(exact_pair.loc[exact_pair["is_current_pair"], "pair_rank"].iloc[0])
    reduced_best = reduced_step.iloc[0]["step_name"]
    reduced_hardest = reduced_step.iloc[-1]["step_name"]
    exact_best = exact_step.iloc[0]["step_name"]
    exact_hardest = exact_step.iloc[-1]["step_name"]

    report_lines = [
        "# V All-Splits Assessment",
        "",
        "## Scope",
        "- Velocity rollout is a dynamic rollout, not a semi-observed tau test.",
        "- Reduced RSF is the primary best-usable velocity law.",
        "- Exact RSF is included as the closest exact-form comparison.",
        f"- Reduced RSF uses all 8 single-step holdouts and all {len(pair_sets)} leave-two-out pairs.",
        "- Exact RSF is shown descriptively using the saved best multistart fit evaluated across all steps; it is not a fresh split-refit sweep because that was not computationally feasible here.",
        "",
        "## Which model works better overall?",
        f"- Reduced RSF median leave-two-out RMSE: `{reduced_pair['mean_pair_velocity_rollout_rmse'].median():.3f}`",
        f"- Exact RSF descriptive pair median RMSE: `{exact_pair['mean_pair_velocity_rollout_rmse'].median():.3f}`",
        f"- Reduced RSF mean stable fraction across pairs: `{reduced_pair['mean_pair_stable_fraction'].mean():.3f}`",
        f"- Exact RSF mean stable fraction across pairs: `{exact_pair['mean_pair_stable_fraction'].mean():.3f}`",
        "",
        "## Easiest and hardest steps",
        f"- Reduced RSF easiest steps: `{', '.join(reduced_step.head(3)['step_name'].tolist())}`",
        f"- Reduced RSF hardest steps: `{', '.join(reduced_step.tail(3)['step_name'].tolist())}`",
        f"- Exact RSF easiest steps: `{', '.join(exact_step.head(3)['step_name'].tolist())}`",
        f"- Exact RSF hardest steps: `{', '.join(exact_step.tail(3)['step_name'].tolist())}`",
        "",
        "## Step2 + Step7",
        f"- Reduced RSF current-pair rank: `{reduced_current_rank} / {len(reduced_pair)}`",
        f"- Exact RSF descriptive current-pair rank: `{exact_current_rank} / {len(exact_pair)}`",
        "",
        "## Presentation guidance",
        "- For V, present all-single-step summaries and leave-two-out summaries as the main evidence.",
        "- Use one representative pair and one stress-test pair, but keep the aggregate summaries primary.",
        "- Present V differently from tau because V rollout is a full dynamic forecast, not a semi-observed latent-state rollout.",
        "",
        "## Reduced RSF Step Difficulty",
        markdown_table(reduced_step[["difficulty_rank", "step_name", "mean_holdout_velocity_rmse", "mean_timing_error_s", "mean_stable_fraction", "difficulty_label"]]),
        "",
        "## Exact RSF Step Difficulty",
        markdown_table(exact_step[["difficulty_rank", "step_name", "mean_holdout_velocity_rmse", "mean_timing_error_s", "mean_stable_fraction", "difficulty_label"]]),
    ]
    OUTPUT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    write_json(
        OUTPUT_JSON,
        {
            "reduced_summary": {
                "step_difficulty": reduced_step,
                "pair_difficulty": reduced_pair,
                "current_pair_rank": reduced_current_rank,
                "best_step": reduced_best,
                "hardest_step": reduced_hardest,
            },
            "exact_summary": {
                "step_difficulty": exact_step,
                "pair_difficulty": exact_pair,
                "current_pair_rank": exact_current_rank,
                "best_step": exact_best,
                "hardest_step": exact_hardest,
                "mode": "descriptive_saved_fixed_fit",
            },
            "generated_files": {
                "report_md": str(OUTPUT_MD),
                "report_json": str(OUTPUT_JSON),
                "reduced_tables": [str(OUTPUT_TABLE), str(STEP_TABLE), str(PAIR_TABLE)],
                "exact_tables": [str(EXACT_OUTPUT_TABLE), str(EXACT_STEP_TABLE), str(EXACT_PAIR_TABLE)],
            },
        },
    )

    print(
        json.dumps(
            json_ready(
                {
                    "reduced_best_step": reduced_best,
                    "reduced_hardest_step": reduced_hardest,
                    "exact_best_step": exact_best,
                    "exact_hardest_step": exact_hardest,
                    "reduced_current_pair_rank": reduced_current_rank,
                    "exact_current_pair_rank": exact_current_rank,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
