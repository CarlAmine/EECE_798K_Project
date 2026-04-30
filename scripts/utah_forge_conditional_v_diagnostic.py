"""Conditional / semi-observed V diagnostic experiment."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utah_forge_proposal_equation_recovery as proposal
from scripts import utah_forge_showcase_fit_visuals as showcase
from src.derivatives import derivative_savgol
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
FINALV5_DIR = RESULTS_DIR / "Finalv5"
FINALV5_FIG_DIR = FINALV5_DIR / "Figures"

OUTPUT_JSON = RESULTS_DIR / "conditional_v_diagnostic_report.json"
OUTPUT_MD = RESULTS_DIR / "conditional_v_diagnostic_report.md"
OUTPUT_TABLE = RESULTS_DIR / "conditional_v_diagnostic_table.csv"

FIG_CONDITIONAL_EXAMPLES = RESULTS_DIR / "conditional_v_rollout_examples.png"
FIG_DYNAMIC_COMPARE = RESULTS_DIR / "reduced_dynamic_vs_conditional_v.png"
FIG_ERROR_MAPS = RESULTS_DIR / "conditional_v_error_maps.png"
FIG_DERIV_SCATTER = RESULTS_DIR / "conditional_v_derivative_scatter.png"
FIG_EQUATION_TABLE = RESULTS_DIR / "conditional_v_equation_table.png"

TRAIN_STEPS = ["p5838_step3", "p5838_step4", "p5838_step5", "p5838_step8", "p5838_step9", "p5838_step10"]
HOLDOUT_STEPS = ["p5838_step2", "p5838_step7"]
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


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}e}" if abs(value) >= 1e3 or (abs(value) > 0 and abs(value) < 1e-2) else f"{value:.{digits}f}"


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |" + "\n| --- |"
    working = frame.copy().fillna("")
    columns = [str(col) for col in working.columns]
    rows = [[str(val) for val in row] for row in working.astype(object).to_numpy().tolist()]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def load_context() -> tuple[dict[str, pd.DataFrame], dict[str, dict], dict, list[str]]:
    outputs, inclusion_rows, _, _ = proposal.prepare_all_segments()
    prepared_map: dict[str, pd.DataFrame] = {}
    for key in ("all_train", "all_holdout"):
        for df in outputs[key]:
            prepared_map[str(df["step_name"].iloc[0])] = df.copy()
    inclusion_map = {str(row["step_name"]): row for row in inclusion_rows}
    proposal_payload = json.loads((RESULTS_DIR / "proposal_equation_recovery.json").read_text(encoding="utf-8"))
    step_summary = json.loads((RESULTS_DIR / "v_reduced_summary.json").read_text(encoding="utf-8"))
    representative_steps = choose_representative_steps(step_summary)
    return prepared_map, inclusion_map, proposal_payload, representative_steps


def choose_representative_steps(step_summary: dict) -> list[str]:
    step_df = pd.DataFrame(step_summary["step_difficulty"]).copy()
    easy = "p5838_step2" if "p5838_step2" in step_df["step_name"].values else str(step_df.loc[step_df["difficulty_label"] == "easy"].iloc[0]["step_name"])
    medium_candidates = step_df.loc[step_df["difficulty_label"] == "medium", "step_name"].tolist()
    medium = "p5838_step9" if "p5838_step9" in medium_candidates else str(medium_candidates[0])
    hard_candidates = step_df.loc[step_df["difficulty_label"] == "hard", "step_name"].tolist()
    hard = "p5838_step5" if "p5838_step5" in hard_candidates else str(hard_candidates[-1])
    return [easy, medium, hard]


def theta_status_for_step(inclusion_map: dict[str, dict], step_name: str) -> dict:
    row = inclusion_map.get(step_name, {})
    raw_reason = row.get("theta_reason", [])
    if isinstance(raw_reason, str):
        theta_reason = [raw_reason]
    elif isinstance(raw_reason, (list, tuple)):
        theta_reason = [str(item) for item in raw_reason]
    else:
        theta_reason = [str(raw_reason)] if raw_reason else []
    return {
        "theta_event_valid": bool(row.get("theta_event_valid", False)),
        "theta_sample_valid": bool(row.get("theta_sample_valid", False)),
        "theta_log_correlation": float(row.get("theta_log_correlation", float("nan"))),
        "theta_keep_fraction": float(row.get("theta_keep_fraction", float("nan"))),
        "theta_reason": theta_reason,
    }


def build_feature_table(segment_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in segment_df.iterrows():
        rows.append({feature_name: proposal.feature_value(feature_name, row) for feature_name in feature_names})
    feature_df = pd.DataFrame(rows)
    return feature_df.replace([np.inf, -np.inf], np.nan)


def build_variant_sample(segment_df: pd.DataFrame, feature_names: list[str], theta_required: bool) -> tuple[pd.DataFrame, np.ndarray]:
    working = segment_df.copy()
    feature_df = build_feature_table(working, feature_names)
    target = working["dV_dt"].to_numpy(dtype=float)
    mask = np.isfinite(target)
    for col in feature_df.columns:
        mask &= np.isfinite(feature_df[col].to_numpy(dtype=float))
    if theta_required:
        theta = working["theta"].to_numpy(dtype=float)
        mask &= np.isfinite(theta) & (theta > EPS)
    return working.loc[mask].reset_index(drop=True), mask


def fit_variant(
    variant: dict,
    prepared_map: dict[str, pd.DataFrame],
    inclusion_map: dict[str, dict],
) -> dict:
    theta_required = bool(variant.get("theta_required", False))
    eligible_train_steps = []
    eligible_holdout_steps = []
    train_frames = []
    for step_name in TRAIN_STEPS:
        if theta_required and not theta_status_for_step(inclusion_map, step_name)["theta_event_valid"]:
            continue
        eligible_train_steps.append(step_name)
        filtered, _ = build_variant_sample(prepared_map[step_name], variant["features"], theta_required)
        if not filtered.empty:
            train_frames.append(filtered)
    for step_name in HOLDOUT_STEPS:
        if theta_required and not theta_status_for_step(inclusion_map, step_name)["theta_event_valid"]:
            continue
        eligible_holdout_steps.append(step_name)
    if not train_frames:
        raise RuntimeError(f"No training data available for {variant['name']}")

    train_df = pd.concat(train_frames, ignore_index=True)
    feature_df = build_feature_table(train_df, variant["features"]).fillna(0.0)
    target = train_df["dV_dt"].to_numpy(dtype=float)
    scaled = feature_df.copy()
    scaling = {}
    for col in variant["features"]:
        if col == "1":
            scaled[col] = 1.0
            scaling[col] = {"mean": 0.0, "std": 1.0}
            continue
        mean = float(feature_df[col].mean())
        std = float(feature_df[col].std())
        if std <= 1e-12:
            std = 1.0
        scaling[col] = {"mean": mean, "std": std}
        scaled[col] = (feature_df[col] - mean) / std
    design = scaled[variant["features"]].to_numpy(dtype=float)
    model = Ridge(alpha=1e-6, fit_intercept=False, solver="svd")
    model.fit(design, target)
    coef_z = model.coef_

    coefficients = {}
    intercept_adjust = 0.0
    for idx, feature_name in enumerate(variant["features"]):
        coeff = float(coef_z[idx])
        if feature_name == "1":
            coefficients["1"] = coeff
            continue
        beta = coeff / scaling[feature_name]["std"]
        intercept_adjust -= coeff * scaling[feature_name]["mean"] / scaling[feature_name]["std"]
        coefficients[feature_name] = beta
    coefficients["1"] = coefficients.get("1", 0.0) + intercept_adjust

    train_pred = proposal.predict_from_feature_table(feature_df.fillna(0.0), coefficients, variant["features"])
    train_metrics = metric_block(target, train_pred)

    holdout_rows = []
    for step_name in eligible_holdout_steps:
        filtered, _ = build_variant_sample(prepared_map[step_name], variant["features"], theta_required)
        if filtered.empty:
            continue
        holdout_feature_df = build_feature_table(filtered, variant["features"]).fillna(0.0)
        holdout_target = filtered["dV_dt"].to_numpy(dtype=float)
        holdout_pred = proposal.predict_from_feature_table(holdout_feature_df, coefficients, variant["features"])
        row_metrics = metric_block(holdout_target, holdout_pred)
        holdout_rows.append(
            {
                "step_name": step_name,
                "n_samples": int(len(filtered)),
                **row_metrics,
                "theta_status": theta_status_for_step(inclusion_map, step_name),
            }
        )

    equation = proposal.format_equation("dV/dt", coefficients, [f for f in variant["features"] if f != "1"])
    return {
        **variant,
        "feature_names": variant["features"],
        "coefficients_physical": coefficients,
        "equation": equation,
        "eligible_train_steps": eligible_train_steps,
        "eligible_holdout_steps": eligible_holdout_steps,
        "train_metrics": train_metrics,
        "holdout_rows": holdout_rows,
        "mean_holdout_metrics": mean_metric_rows(holdout_rows, prefix="derivative_"),
    }


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mse = float(np.mean(residual ** 2))
    return {
        "derivative_mse": mse,
        "derivative_rmse": float(np.sqrt(mse)),
        "derivative_mae": float(np.mean(np.abs(residual))),
        "derivative_r2": safe_r2(y_true, y_pred),
    }


def mean_metric_rows(rows: list[dict], prefix: str) -> dict:
    keys = [key for key in rows[0].keys() if key.startswith(prefix)] if rows else []
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if np.isfinite(row[key])]
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def conditional_rollout(segment_df: pd.DataFrame, coefficients: dict[str, float], feature_names: list[str], use_theta: bool) -> dict:
    time_values = segment_df["time"].to_numpy(dtype=float)
    observed_v = segment_df["V"].to_numpy(dtype=float)
    observed_tau = segment_df["tau"].to_numpy(dtype=float)
    observed_sigma = segment_df["sigmaN"].to_numpy(dtype=float)
    observed_theta = segment_df["theta"].to_numpy(dtype=float)
    v0 = segment_df["V0"].to_numpy(dtype=float)
    dc = segment_df["Dc"].to_numpy(dtype=float)

    def rhs(state: np.ndarray, t_value: float) -> list[float]:
        current_v = max(float(state[0]), EPS)
        row = pd.Series(
            {
                "tau": float(np.interp(t_value, time_values, observed_tau)),
                "sigmaN": float(np.interp(t_value, time_values, observed_sigma)),
                "theta": max(float(np.interp(t_value, time_values, observed_theta)), EPS) if use_theta else 1.0,
                "V": current_v,
                "V0": max(float(np.interp(t_value, time_values, v0)), EPS),
                "Dc": max(float(np.interp(t_value, time_values, dc)), EPS),
            }
        )
        dvdt = coefficients.get("1", 0.0)
        for feature_name in feature_names:
            if feature_name == "1":
                continue
            dvdt += coefficients.get(feature_name, 0.0) * proposal.feature_value(feature_name, row)
        return [dvdt]

    try:
        predicted_v = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), [float(observed_v[0])], time_values).reshape(-1)
    except Exception:
        predicted_v = np.full(len(observed_v), np.nan, dtype=float)
    predicted_dv = derivative_savgol(predicted_v, t=time_values, window=15, polyorder=3) if np.isfinite(predicted_v).all() else np.full(len(predicted_v), np.nan)
    return rollout_metrics(
        step_name=str(segment_df["step_name"].iloc[0]),
        time_values=time_values,
        observed_v=observed_v,
        predicted_v=predicted_v,
        observed_dv=segment_df["dV_dt"].to_numpy(dtype=float),
        predicted_dv=predicted_dv,
    )


def dynamic_rollout(segment_df: pd.DataFrame, model_row: dict) -> dict:
    arrays = showcase.rollout_velocity_series(model_row, segment_df)
    return rollout_metrics(
        step_name=arrays["step_name"],
        time_values=arrays["time"],
        observed_v=arrays["observed_v"],
        predicted_v=arrays["predicted_v"],
        observed_dv=arrays["observed_dv"],
        predicted_dv=arrays["predicted_dv"],
    )


def rollout_metrics(
    step_name: str,
    time_values: np.ndarray,
    observed_v: np.ndarray,
    predicted_v: np.ndarray,
    observed_dv: np.ndarray,
    predicted_dv: np.ndarray,
) -> dict:
    finite = np.isfinite(observed_v) & np.isfinite(predicted_v)
    if not np.any(finite):
        return {
            "step_name": step_name,
            "rollout_mse": float("nan"),
            "rollout_rmse": float("nan"),
            "rollout_mae": float("nan"),
            "max_abs_error": float("nan"),
            "peak_timing_error_s": float("nan"),
            "onset_timing_error_s": float("nan"),
            "stable_fraction": 0.0,
            "predicted_v": predicted_v.tolist(),
            "observed_v": observed_v.tolist(),
            "time": (time_values - time_values[0]).tolist(),
            "predicted_dv": predicted_dv.tolist(),
            "observed_dv": observed_dv.tolist(),
        }
    obs = observed_v[finite]
    pred = predicted_v[finite]
    residual = pred - obs
    mse = float(np.mean(residual ** 2))
    sigma_obs = float(np.std(obs)) if len(obs) > 1 else 1.0
    threshold = 3.0 * max(sigma_obs, 1e-6)
    divergence_index = len(predicted_v)
    for idx, (pred_i, obs_i) in enumerate(zip(predicted_v, observed_v)):
        if (not np.isfinite(pred_i)) or abs(pred_i - obs_i) > threshold:
            divergence_index = idx
            break
    return {
        "step_name": step_name,
        "rollout_mse": mse,
        "rollout_rmse": float(np.sqrt(mse)),
        "rollout_mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "peak_timing_error_s": abs(proposal.peak_time(predicted_v, time_values) - proposal.peak_time(observed_v, time_values)),
        "onset_timing_error_s": abs(proposal.onset_time(predicted_v, time_values) - proposal.onset_time(observed_v, time_values)),
        "stable_fraction": float(divergence_index / len(predicted_v)) if len(predicted_v) else 0.0,
        "predicted_v": predicted_v.tolist(),
        "observed_v": observed_v.tolist(),
        "time": (time_values - time_values[0]).tolist(),
        "predicted_dv": predicted_dv.tolist(),
        "observed_dv": observed_dv.tolist(),
    }


def summarize_rollouts(rows: list[dict]) -> dict:
    keys = ["rollout_mse", "rollout_rmse", "rollout_mae", "max_abs_error", "peak_timing_error_s", "onset_timing_error_s", "stable_fraction"]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if np.isfinite(row[key])]
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def qualitative_graph_label(dynamic_rows: list[dict], candidate_rows: list[dict]) -> str:
    dyn = np.nanmean([row["rollout_rmse"] for row in dynamic_rows])
    cand = np.nanmean([row["rollout_rmse"] for row in candidate_rows])
    if cand <= 0.8 * dyn:
        return "yes"
    if cand <= 1.05 * dyn:
        return "partly"
    return "no"


def trust_label(name: str) -> str:
    if "dynamic" in name:
        return "yes"
    if "no_tau" in name:
        return "limited"
    return "limited"


def make_comparison_rows(
    proposal_payload: dict,
    representative_steps: list[str],
    prepared_map: dict[str, pd.DataFrame],
    variant_results: dict[str, dict],
) -> tuple[list[dict], dict[str, list[dict]]]:
    dynamic_model = proposal_payload["final_velocity_model"]
    dynamic_rows = [dynamic_rollout(prepared_map[step], dynamic_model) for step in representative_steps]
    dynamic_metrics = summarize_rollouts(dynamic_rows)
    dynamic_deriv_rows = []
    for row in dynamic_rows:
        dynamic_deriv_rows.append(metric_block(np.asarray(row["observed_dv"], dtype=float), np.asarray(row["predicted_dv"], dtype=float)))
    dynamic_deriv_summary = mean_metric_rows(dynamic_deriv_rows, prefix="derivative_")

    comparison_rows = [
        {
            "variant": "current_reduced_rsf_dynamic_rollout",
            "display_name": "Current reduced RSF dynamic rollout",
            "equation_form": dynamic_model["equation"],
            "derivative_rmse": dynamic_deriv_summary["mean_derivative_rmse"],
            "rollout_rmse": dynamic_metrics["mean_rollout_rmse"],
            "timing_error_s": 0.5 * (dynamic_metrics["mean_peak_timing_error_s"] + dynamic_metrics["mean_onset_timing_error_s"]),
            "stable_fraction": dynamic_metrics["mean_stable_fraction"],
            "graph_looks_better": "baseline",
            "still_physically_trustworthy": "yes",
            "evaluation_type": "dynamic",
            "fairness_note": "strictest; predicts V dynamically rather than consuming observed tau(t)",
        }
    ]
    rollout_map = {"current_reduced_rsf_dynamic_rollout": dynamic_rows}
    for variant_name, result in variant_results.items():
        rows = result["diagnostic_rollouts"]
        rollout_map[variant_name] = rows
        summary = result["rollout_summary"]
        comparison_rows.append(
            {
                "variant": variant_name,
                "display_name": result["label"],
                "equation_form": result["equation"],
                "derivative_rmse": result["mean_holdout_metrics"]["mean_derivative_rmse"],
                "rollout_rmse": summary["mean_rollout_rmse"],
                "timing_error_s": 0.5 * (summary["mean_peak_timing_error_s"] + summary["mean_onset_timing_error_s"]),
                "stable_fraction": summary["mean_stable_fraction"],
                "graph_looks_better": qualitative_graph_label(dynamic_rows, rows),
                "still_physically_trustworthy": trust_label(variant_name),
                "evaluation_type": "conditional",
                "fairness_note": "conditional / semi-observed; observed tau(t), sigmaN(t), and sometimes theta(t) are supplied",
            }
        )
    return comparison_rows, rollout_map


def plot_rollout_examples(representative_steps: list[str], variant_results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(len(representative_steps), 1, figsize=(12, 10), sharex=False)
    if len(representative_steps) == 1:
        axes = [axes]
    colors = {"A_conditional_reduced_style_v": "#1f77b4", "B_conditional_theta_augmented_v": "#ff7f0e", "C_conditional_no_tau_v": "#2ca02c"}
    for ax, step_name in zip(axes, representative_steps):
        first = next(row for row in variant_results["A_conditional_reduced_style_v"]["diagnostic_rollouts"] if row["step_name"] == step_name)
        time_values = np.asarray(first["time"], dtype=float)
        observed_v = np.asarray(first["observed_v"], dtype=float)
        ax.plot(time_values, observed_v, color="black", linewidth=1.7, label="Observed V")
        for variant_name, result in variant_results.items():
            row = next((item for item in result["diagnostic_rollouts"] if item["step_name"] == step_name), None)
            if row is None:
                continue
            ax.plot(time_values, np.asarray(row["predicted_v"], dtype=float), linewidth=1.2, label=result["short_label"], color=colors[variant_name])
        ax.set_title(f"{step_name}: Conditional / semi-observed V rollout")
        ax.set_ylabel("V")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.01,
            0.02,
            "Observed tau(t) and sigmaN(t) are supplied.\nThis is diagnostic, not a closed-loop dynamic rollout.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("time since step start [s]")
    fig.tight_layout()
    fig.savefig(FIG_CONDITIONAL_EXAMPLES, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_vs_conditional(representative_steps: list[str], dynamic_rows: list[dict], variant_results: dict[str, dict]) -> None:
    preferred = "A_conditional_reduced_style_v"
    if variant_results["B_conditional_theta_augmented_v"]["rollout_summary"]["mean_rollout_rmse"] < variant_results[preferred]["rollout_summary"]["mean_rollout_rmse"]:
        preferred = "B_conditional_theta_augmented_v"
    cond_rows = variant_results[preferred]["diagnostic_rollouts"]
    fig, axes = plt.subplots(len(representative_steps), 2, figsize=(13, 10), sharex=False)
    if len(representative_steps) == 1:
        axes = np.array([axes])
    for ridx, step_name in enumerate(representative_steps):
        dyn_row = next(row for row in dynamic_rows if row["step_name"] == step_name)
        cond_row = next(row for row in cond_rows if row["step_name"] == step_name)
        for cidx, row in enumerate([dyn_row, cond_row]):
            ax = axes[ridx, cidx]
            time_values = np.asarray(row["time"], dtype=float)
            ax.plot(time_values, np.asarray(row["observed_v"], dtype=float), color="black", linewidth=1.6, label="Observed V")
            ax.plot(time_values, np.asarray(row["predicted_v"], dtype=float), color="#d62728", linewidth=1.2, label="Predicted V")
            title = "Current reduced RSF dynamic rollout" if cidx == 0 else f"{variant_results[preferred]['short_label']} conditional rollout"
            ax.set_title(f"{step_name}: {title}")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("V")
            if cidx == 1:
                ax.text(
                    0.01,
                    0.02,
                    "Conditional / semi-observed V rollout",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("time since step start [s]")
    axes[-1, 1].set_xlabel("time since step start [s]")
    fig.tight_layout()
    fig.savefig(FIG_DYNAMIC_COMPARE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(representative_steps: list[str], variant_results: dict[str, dict]) -> None:
    variant_names = list(variant_results.keys())
    fig, axes = plt.subplots(len(representative_steps), len(variant_names), figsize=(14, 9), sharex=False, sharey=False)
    if len(representative_steps) == 1:
        axes = np.array([axes])
    for ridx, step_name in enumerate(representative_steps):
        for cidx, variant_name in enumerate(variant_names):
            ax = axes[ridx, cidx]
            row = next((item for item in variant_results[variant_name]["diagnostic_rollouts"] if item["step_name"] == step_name), None)
            if row is None:
                ax.axis("off")
                continue
            time_values = np.asarray(row["time"], dtype=float)
            error = np.abs(np.asarray(row["predicted_v"], dtype=float) - np.asarray(row["observed_v"], dtype=float))
            ax.fill_between(time_values, error, color="#c44e52", alpha=0.85)
            ax.set_title(f"{step_name}\n{variant_results[variant_name]['short_label']}")
            ax.grid(True, alpha=0.25)
            if cidx == 0:
                ax.set_ylabel("|V error|")
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle("Conditional / semi-observed V rollout absolute error", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_ERROR_MAPS, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_derivative_scatter(variant_results: dict[str, dict], dynamic_rows: list[dict]) -> None:
    panels = [("Current reduced RSF dynamic rollout", dynamic_rows)]
    panels.extend((result["short_label"], result["diagnostic_rollouts"]) for result in variant_results.values())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, rows) in zip(axes.flat, panels):
        obs = np.concatenate([np.asarray(row["observed_dv"], dtype=float) for row in rows])
        pred = np.concatenate([np.asarray(row["predicted_dv"], dtype=float) for row in rows])
        finite = np.isfinite(obs) & np.isfinite(pred)
        obs = obs[finite]
        pred = pred[finite]
        ax.scatter(obs, pred, s=8, alpha=0.35)
        lo = float(min(np.min(obs), np.min(pred)))
        hi = float(max(np.max(obs), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Observed dV/dt")
        ax.set_ylabel("Predicted dV/dt")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Conditional vs dynamic derivative scatter", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DERIV_SCATTER, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_equation_table(comparison_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")
    shown = comparison_df[["display_name", "evaluation_type", "equation_form"]].copy()
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(FIG_EQUATION_TABLE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def maybe_make_finalv5(comparison_df: pd.DataFrame, representative_steps: list[str]) -> bool:
    best_conditional_rmse = float(comparison_df.loc[comparison_df["evaluation_type"] == "conditional", "rollout_rmse"].min())
    if not np.isfinite(best_conditional_rmse):
        return False
    ensure_directory(FINALV5_DIR)
    ensure_directory(FINALV5_FIG_DIR)
    copied = []
    for path in [FIG_CONDITIONAL_EXAMPLES, FIG_DYNAMIC_COMPARE, FIG_ERROR_MAPS, FIG_DERIV_SCATTER, FIG_EQUATION_TABLE]:
        if path.exists():
            shutil.copy2(path, FINALV5_FIG_DIR / path.name)
            copied.append(path.name)
    conditional_summary = (
        "# Conditional V Summary\n\n"
        "This package contains the focused conditional / semi-observed velocity diagnostic.\n\n"
        "- It asks whether V looks cleaner when `tau(t)` and `sigmaN(t)` are supplied from observations.\n"
        "- It does not replace the main dynamic reduced-RSF V result.\n"
    )
    comparison_summary = (
        "# Conditional vs Dynamic Comparison\n\n"
        f"- Representative steps: `{', '.join(representative_steps)}`.\n"
        "- Conditional variants are easier tests because they consume observed auxiliary forcing.\n"
        "- The reduced RSF dynamic rollout remains the main usable V result unless the conditional evidence is strong enough to change that fairly.\n"
    )
    readme = (
        "# Finalv5\n\n"
        "Focused conditional / semi-observed V diagnostic package.\n\n"
        "Contents:\n"
        "- `conditional_v_summary.md`\n"
        "- `conditional_vs_dynamic_comparison.md`\n"
        "- `Figures/`\n\n"
        "Reporting note:\n"
        "- This package is diagnostic only. It should be presented as a conditional / semi-observed velocity test, not as a replacement for the main dynamic reduced-RSF result.\n"
    )
    (FINALV5_DIR / "conditional_v_summary.md").write_text(conditional_summary, encoding="utf-8")
    (FINALV5_DIR / "conditional_vs_dynamic_comparison.md").write_text(comparison_summary, encoding="utf-8")
    (FINALV5_DIR / "README.md").write_text(readme, encoding="utf-8")
    return bool(copied)


def build_report(
    representative_steps: list[str],
    inclusion_map: dict[str, dict],
    variant_results: dict[str, dict],
    comparison_df: pd.DataFrame,
    proposal_payload: dict,
    finalv5_created: bool,
) -> str:
    dynamic_saved = proposal_payload["final_velocity_model"]
    theta_lines = []
    for step_name in representative_steps:
        status = theta_status_for_step(inclusion_map, step_name)
        theta_lines.append(
            {
                "step_name": step_name,
                "theta_event_valid": status["theta_event_valid"],
                "theta_sample_valid": status["theta_sample_valid"],
                "theta_log_correlation": fmt(status["theta_log_correlation"]),
                "theta_keep_fraction": fmt(status["theta_keep_fraction"]),
                "theta_reason": ", ".join(status["theta_reason"]) if status["theta_reason"] else "none",
            }
        )
    theta_df = pd.DataFrame(theta_lines)
    qa = []
    best_conditional = comparison_df.loc[comparison_df["evaluation_type"] == "conditional"].sort_values("rollout_rmse").iloc[0]
    dynamic_row = comparison_df.loc[comparison_df["variant"] == "current_reduced_rsf_dynamic_rollout"].iloc[0]
    qa.append(f"1. Does isolating V make the graphs look nicer? On the representative steps, the best conditional variant is `{best_conditional['display_name']}` and the graph-quality judgment is `{best_conditional['graph_looks_better']}` relative to the dynamic baseline.")
    qa.append(f"2. Does it improve derivative fit? Best conditional derivative RMSE is `{fmt(best_conditional['derivative_rmse'])}` versus `{fmt(dynamic_row['derivative_rmse'])}` for the representative-step dynamic baseline; the saved project holdout derivative RMSE for reduced RSF is `{fmt(dynamic_saved['holdout_rmse'])}`.")
    qa.append(f"3. Does it improve conditional rollout? Best conditional rollout RMSE is `{fmt(best_conditional['rollout_rmse'])}` versus `{fmt(dynamic_row['rollout_rmse'])}` for the same representative steps.")
    tau_variant = comparison_df.loc[comparison_df["variant"] == "A_conditional_reduced_style_v"].iloc[0]
    no_tau_variant = comparison_df.loc[comparison_df["variant"] == "C_conditional_no_tau_v"].iloc[0]
    qa.append(f"4. Does tau remain active or collapse? In the tau-including reduced-style fit, tau stays in the equation. Compared with the no-tau ablation, rollout RMSE changes from `{fmt(tau_variant['rollout_rmse'])}` to `{fmt(no_tau_variant['rollout_rmse'])}`.")
    theta_variant = comparison_df.loc[comparison_df["variant"] == "B_conditional_theta_augmented_v"].iloc[0]
    qa.append(f"5. Does theta help visually or numerically? Theta-augmented conditional rollout RMSE is `{fmt(theta_variant['rollout_rmse'])}` versus `{fmt(tau_variant['rollout_rmse'])}` for the reduced-style conditional fit.")
    qa.append("6. Does this overturn the main conclusion about V? No automatic overturning is warranted just because a conditional test is cleaner; the evidentiary standard remains lower than a full dynamic rollout.")
    qa.append("7. What is the honest way to present this result? Present it as a conditional / semi-observed velocity diagnostic that asks how much of the V difficulty comes from coupled rollout difficulty, while keeping the reduced RSF dynamic model as the main usable V result unless the conditional evidence is compelling on a fair comparison.")
    lines = [
        "# Conditional V Diagnostic Report",
        "",
        "## Goal",
        "Test whether equation (2) for `V` looks cleaner when isolated and evaluated conditionally, in the same spirit as the semi-observed tau rollout, while keeping the interpretation honest.",
        "",
        "## Honesty framing",
        "- The tau rollout used elsewhere in this project is semi-observed because `V(t)` and `V_drive(t)` are supplied.",
        "- This new V diagnostic is also conditional / semi-observed because `tau(t)` and `sigmaN(t)` are supplied from observations, and `theta(t)` is supplied only when RSFit-aligned theta is considered valid enough to use diagnostically.",
        "- These conditional tests are not the same evidentiary level as a full dynamic rollout.",
        "- The current reduced RSF model remains the main usable dynamic V model unless this diagnostic clearly overturns it on a fair comparison, which requires caution because the conditional task is easier.",
        "",
        "## Setup",
        f"- Train steps: `{', '.join(TRAIN_STEPS)}`",
        f"- Holdout steps: `{', '.join(HOLDOUT_STEPS)}`",
        f"- Representative steps for visualization: `{', '.join(representative_steps)}`",
        "- Representative steps were chosen from the saved reduced-RSF step-difficulty summary to span one easy, one medium, and one hard case under the existing V analysis.",
        "",
        "## Theta availability",
        markdown_table(theta_df),
        "",
        "## Tested equations",
        "",
    ]
    for result in variant_results.values():
        lines.extend(
            [
                f"### {result['label']}",
                f"- Library: `{', '.join(result['feature_names'])}`",
                f"- Equation: `{result['equation']}`",
                f"- Eligible train steps: `{', '.join(result['eligible_train_steps'])}`",
                f"- Eligible holdout steps: `{', '.join(result['eligible_holdout_steps']) if result['eligible_holdout_steps'] else 'none'}`",
                f"- Mean holdout derivative RMSE: `{fmt(result['mean_holdout_metrics'].get('mean_derivative_rmse', float('nan')))}`",
                f"- Mean representative rollout RMSE: `{fmt(result['rollout_summary'].get('mean_rollout_rmse', float('nan')))}`",
                "",
            ]
        )
    display_df = comparison_df.copy()
    for col in ["derivative_rmse", "rollout_rmse", "timing_error_s", "stable_fraction"]:
        display_df[col] = display_df[col].map(fmt)
    lines.extend(
        [
            "## Conditional vs dynamic comparison",
            markdown_table(
                display_df[
                    [
                        "display_name",
                        "equation_form",
                        "derivative_rmse",
                        "rollout_rmse",
                        "timing_error_s",
                        "stable_fraction",
                        "graph_looks_better",
                        "still_physically_trustworthy",
                        "evaluation_type",
                    ]
                ]
            ),
            "",
            "## Direct answers",
            *[f"- {item}" for item in qa],
            "",
            "## Fairness notes",
            f"- Saved project-wide reduced-RSF holdout pair (`{', '.join(HOLDOUT_STEPS)}`) metrics remain: derivative RMSE `{fmt(dynamic_saved['holdout_rmse'])}`, rollout MSE `{fmt(dynamic_saved['mean_rollout_mse'])}`, stable fraction `{fmt(dynamic_saved['mean_stable_fraction'])}`.",
            "- The representative-step dynamic row above was recomputed on the same easy/medium/hard steps used for the new conditional figures so the plots are visually comparable.",
            "- That side-by-side comparison is still not fully fair to the dynamic model because the conditional variant consumes observed auxiliary signals.",
            "",
            "## How to present this honestly",
            "- This diagnostic shows whether poor V graphs are partly due to coupled rollout difficulty rather than only poor instantaneous equation structure.",
            "- Cleaner conditional V fits do not automatically mean full equation recovery.",
            "- A conditional / semi-observed V rollout should be described as a diagnostic using observed forcing/context inputs, not as a replacement for the main dynamic result.",
            "- The reduced RSF dynamic model remains the main usable V result unless the diagnostic clearly beats it in a scientifically fair way.",
            "",
            "## Generated files",
            "- `results/utah_forge/conditional_v_diagnostic_report.md`",
            "- `results/utah_forge/conditional_v_diagnostic_report.json`",
            "- `results/utah_forge/conditional_v_diagnostic_table.csv`",
            "- `results/utah_forge/conditional_v_rollout_examples.png`",
            "- `results/utah_forge/reduced_dynamic_vs_conditional_v.png`",
            "- `results/utah_forge/conditional_v_error_maps.png`",
            "- `results/utah_forge/conditional_v_derivative_scatter.png`",
            "- `results/utah_forge/conditional_v_equation_table.png`",
            f"- `results/utah_forge/Finalv5/` created: `{finalv5_created}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    prepared_map, inclusion_map, proposal_payload, representative_steps = load_context()
    variants = [
        {
            "name": "A_conditional_reduced_style_v",
            "label": "Conditional reduced-style V fit",
            "short_label": "Conditional reduced-style",
            "features": ["1", "tau", "sigmaN", "sigmaN_logV"],
            "theta_required": False,
        },
        {
            "name": "B_conditional_theta_augmented_v",
            "label": "Conditional theta-augmented V fit",
            "short_label": "Conditional theta-augmented",
            "features": ["1", "tau", "sigmaN", "sigmaN_logV", "sigmaN_logTheta"],
            "theta_required": True,
        },
        {
            "name": "C_conditional_no_tau_v",
            "label": "Conditional no-tau ablation",
            "short_label": "Conditional no-tau",
            "features": ["1", "sigmaN", "sigmaN_logV", "sigmaN_logTheta"],
            "theta_required": True,
        },
    ]

    variant_results = {}
    for variant in variants:
        result = fit_variant(variant, prepared_map, inclusion_map)
        rollout_rows = []
        for step_name in representative_steps:
            if variant["theta_required"] and not theta_status_for_step(inclusion_map, step_name)["theta_event_valid"]:
                continue
            rollout_rows.append(
                conditional_rollout(
                    prepared_map[step_name],
                    result["coefficients_physical"],
                    result["feature_names"],
                    use_theta=variant["theta_required"],
                )
            )
        result["diagnostic_rollouts"] = rollout_rows
        result["rollout_summary"] = summarize_rollouts(rollout_rows)
        variant_results[variant["name"]] = result

    comparison_rows, rollout_map = make_comparison_rows(proposal_payload, representative_steps, prepared_map, variant_results)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_TABLE, index=False)

    dynamic_rows = rollout_map["current_reduced_rsf_dynamic_rollout"]
    plot_rollout_examples(representative_steps, variant_results)
    plot_dynamic_vs_conditional(representative_steps, dynamic_rows, variant_results)
    plot_error_maps(representative_steps, variant_results)
    plot_derivative_scatter(variant_results, dynamic_rows)
    plot_equation_table(comparison_df)

    finalv5_created = maybe_make_finalv5(comparison_df, representative_steps)
    report_text = build_report(representative_steps, inclusion_map, variant_results, comparison_df, proposal_payload, finalv5_created)
    OUTPUT_MD.write_text(report_text, encoding="utf-8")

    json_payload = {
        "experiment": "conditional_v_diagnostic",
        "train_steps": TRAIN_STEPS,
        "holdout_steps": HOLDOUT_STEPS,
        "representative_steps": representative_steps,
        "theta_status": {step: theta_status_for_step(inclusion_map, step) for step in sorted(prepared_map)},
        "variants": json_ready(variant_results),
        "comparison_rows": json_ready(comparison_rows),
        "proposal_final_velocity_model": proposal_payload["final_velocity_model"],
        "generated_files": [
            str(OUTPUT_JSON),
            str(OUTPUT_MD),
            str(OUTPUT_TABLE),
            str(FIG_CONDITIONAL_EXAMPLES),
            str(FIG_DYNAMIC_COMPARE),
            str(FIG_ERROR_MAPS),
            str(FIG_DERIV_SCATTER),
            str(FIG_EQUATION_TABLE),
        ],
        "finalv5_created": finalv5_created,
        "timestamp": time.time(),
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(json_payload), indent=2), encoding="utf-8")
    print(json.dumps({"representative_steps": representative_steps, "finalv5_created": finalv5_created}, indent=2))


if __name__ == "__main__":
    main()
