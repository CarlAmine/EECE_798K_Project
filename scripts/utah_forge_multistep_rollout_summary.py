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
from scipy.integrate import odeint


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utah_forge_proposal_equation_recovery as proposal_recovery
from scripts import utah_forge_showcase_fit_visuals as showcase
from src.derivatives import derivative_savgol
from src.exact_rsf import simulate_exact_rsf_segment


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
DEFAULT_STEP_ORDER = [
    "p5838_step2",
    "p5838_step3",
    "p5838_step4",
    "p5838_step5",
    "p5838_step7",
    "p5838_step8",
    "p5838_step9",
    "p5838_step10",
]


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_prepared_segments() -> tuple[list[pd.DataFrame], dict[str, str], dict]:
    proposal = json.loads((RESULTS_DIR / "proposal_equation_recovery.json").read_text(encoding="utf-8"))
    segments, steps, rsfit_globals = showcase.load_segments_without_writing()
    split_map = {
        str(row["step_name"]): str(row["split"])
        for row in proposal["inclusion_rows"]
        if str(row["step_name"]).startswith("p5838_step")
    }
    prepared_segments: list[pd.DataFrame] = []
    for step_name in DEFAULT_STEP_ORDER:
        if step_name not in segments or step_name not in split_map:
            continue
        segment_df = segments[step_name]
        prepared_df, _, _ = proposal_recovery.prepare_segment_with_rsf(segment_df, steps[step_name], rsfit_globals, None)
        prepared_df = prepared_df.copy()
        prepared_df["split"] = split_map[step_name]
        prepared_segments.append(prepared_df)
    return prepared_segments, split_map, proposal


def tau_rollout_arrays(tau_model: dict, segment_df: pd.DataFrame, one_term: bool = False) -> dict:
    if one_term:
        k_hat = float(tau_model["one_term_k_hat"])
        coefficients = {"1": 0.0, "V": -k_hat, "V_drive_minus_V": k_hat}
    else:
        coefficients = tau_model["coefficients_physical"]
    time = segment_df["time"].to_numpy(dtype=float)
    rel_time = time - float(time[0])
    observed_tau = segment_df["tau"].to_numpy(dtype=float)
    observed_v = segment_df["V"].to_numpy(dtype=float)
    observed_v_drive = segment_df["V_drive"].to_numpy(dtype=float)
    observed_dtau = segment_df["dtau_dt"].to_numpy(dtype=float)

    def rhs(state: np.ndarray, t_value: float) -> list[float]:
        v_now = float(np.interp(t_value, time, observed_v))
        v_drive_now = float(np.interp(t_value, time, observed_v_drive))
        return [
            coefficients.get("1", 0.0)
            + coefficients.get("V", 0.0) * v_now
            + coefficients.get("V_drive_minus_V", 0.0) * (v_drive_now - v_now)
        ]

    predicted_tau = odeint(
        lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val),
        [float(observed_tau[0])],
        time,
    ).reshape(-1)
    predicted_dtau = (
        coefficients.get("1", 0.0)
        + coefficients.get("V", 0.0) * observed_v
        + coefficients.get("V_drive_minus_V", 0.0) * (observed_v_drive - observed_v)
    )
    error = predicted_tau - observed_tau
    return {
        "step_name": str(segment_df["step_name"].iloc[0]),
        "split": str(segment_df["split"].iloc[0]),
        "time": time,
        "rel_time": rel_time,
        "observed_tau": observed_tau,
        "predicted_tau": predicted_tau,
        "observed_dtau": observed_dtau,
        "predicted_dtau": predicted_dtau,
        "observed_v": observed_v,
        "tau_error": error,
        "abs_tau_error": np.abs(error),
        "duration_s": float(rel_time[-1]) if len(rel_time) else 0.0,
        "n_samples": int(len(time)),
        "tau_rollout_mse": float(np.mean(error**2)),
        "tau_rollout_rmse": float(np.sqrt(np.mean(error**2))),
        "tau_mean_abs_error": float(np.mean(np.abs(error))),
        "tau_max_abs_error": float(np.max(np.abs(error))),
    }


def reduced_velocity_arrays(model_row: dict, segment_df: pd.DataFrame) -> dict:
    series = showcase.rollout_velocity_series(model_row, segment_df)
    predicted_v = series["predicted_v"]
    observed_v = series["observed_v"]
    error = predicted_v - observed_v
    threshold = 3.0 * max(float(np.std(observed_v)), 1e-6)
    divergence_index = len(observed_v)
    for index, (pred, obs) in enumerate(zip(predicted_v, observed_v)):
        if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
            divergence_index = index
            break
    stable_fraction = float(divergence_index / len(observed_v)) if len(observed_v) else 0.0
    return {
        **series,
        "split": str(segment_df["split"].iloc[0]),
        "duration_s": float(series["rel_time"][-1]) if len(series["rel_time"]) else 0.0,
        "n_samples": int(len(series["time"])),
        "velocity_rollout_mse": float(np.mean(error**2)),
        "velocity_rollout_rmse": float(np.sqrt(np.mean(error**2))),
        "velocity_mean_abs_error": float(np.mean(np.abs(error))),
        "velocity_max_abs_error": float(np.max(np.abs(error))),
        "stable_fraction": stable_fraction,
        "diverged": bool(stable_fraction < 1.0),
        "peak_timing_error_s": abs(
            proposal_recovery.peak_time(predicted_v, series["time"])
            - proposal_recovery.peak_time(observed_v, series["time"])
        ),
        "onset_timing_error_s": abs(
            proposal_recovery.onset_time(predicted_v, series["time"])
            - proposal_recovery.onset_time(observed_v, series["time"])
        ),
    }


def load_exact_payloads() -> tuple[dict, dict]:
    multistart = json.loads((RESULTS_DIR / "exact_rsf_multistart_summary.json").read_text(encoding="utf-8"))
    prepared_exact = showcase.load_checkpoint(RESULTS_DIR / "exact_rsf_inverse_fit_checkpoints", "prepared_exact_segments")
    best_start = int(multistart["best_run"]["start_index"])
    exact_payload = showcase.load_checkpoint(RESULTS_DIR / "exact_rsf_multistart_checkpoints", f"exact_fit_multistart_{best_start}")
    if prepared_exact is None or exact_payload is None:
        raise RuntimeError("Missing saved exact RSF checkpoints.")
    return prepared_exact, exact_payload


def exact_rows_all_steps(prepared_exact: dict, exact_payload: dict) -> list[dict]:
    train_segments = prepared_exact["train_segments"]
    holdout_segments = prepared_exact["holdout_segments"]
    params = exact_payload["parameters"]
    acoustic_z = exact_payload["acoustic_zscores"]
    theta_offsets = exact_payload.get("theta_offsets_train", {})
    rows: list[dict] = []
    for segment in list(train_segments) + list(holdout_segments):
        delta_log_theta0 = float(theta_offsets.get(segment.step_name, 0.0))
        sim = simulate_exact_rsf_segment(
            segment,
            params,
            delta_log_theta0=delta_log_theta0,
            acoustic_z=acoustic_z.get(segment.step_name, 0.0),
        )
        predicted_dtau = derivative_savgol(sim["tau"], t=segment.time, window=15, polyorder=3)
        predicted_dv = derivative_savgol(sim["V"], t=segment.time, window=15, polyorder=3)
        tau_error = sim["tau"] - segment.tau
        v_error = sim["V"] - segment.V
        metrics = proposal_recovery.EPS  # keep namespace access explicit below
        del metrics
        rollout = showcase.rollout_velocity_series  # avoid lint noise in plain script
        del rollout
        event_metrics = {
            "tau_rmse": float(np.sqrt(np.mean(tau_error**2))),
            "velocity_rollout_rmse": float(np.sqrt(np.mean(v_error**2))),
            "tau_rollout_mse": float(np.mean(tau_error**2)),
            "velocity_rollout_mse": float(np.mean(v_error**2)),
            "tau_mean_abs_error": float(np.mean(np.abs(tau_error))),
            "velocity_mean_abs_error": float(np.mean(np.abs(v_error))),
            "tau_max_abs_error": float(np.max(np.abs(tau_error))),
            "velocity_max_abs_error": float(np.max(np.abs(v_error))),
        }
        threshold = 3.0 * max(float(np.std(segment.V)), 1e-6)
        divergence_index = len(segment.V)
        for index, (pred, obs) in enumerate(zip(sim["V"], segment.V)):
            if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
                divergence_index = index
                break
        stable_fraction = float(divergence_index / len(segment.V)) if len(segment.V) else 0.0
        rows.append(
            {
                "step_name": segment.step_name,
                "split": segment.split,
                "time": segment.time,
                "rel_time": segment.time - float(segment.time[0]),
                "observed_tau": segment.tau,
                "predicted_tau": sim["tau"],
                "observed_v": segment.V,
                "predicted_v": sim["V"],
                "observed_theta": segment.theta_proxy,
                "predicted_theta": sim["theta"],
                "observed_dtau": segment.dtau_dt,
                "predicted_dtau": predicted_dtau,
                "observed_dv": segment.dV_dt,
                "predicted_dv": predicted_dv,
                "abs_tau_error": np.abs(tau_error),
                "abs_v_error": np.abs(v_error),
                "duration_s": float(segment.time[-1] - segment.time[0]) if len(segment.time) else 0.0,
                "n_samples": int(len(segment.time)),
                "theta0_used": float(sim["theta0"]),
                "stable_fraction": stable_fraction,
                "diverged": bool(stable_fraction < 1.0),
                "peak_timing_error_s": abs(
                    proposal_recovery.peak_time(sim["V"], segment.time)
                    - proposal_recovery.peak_time(segment.V, segment.time)
                ),
                "onset_timing_error_s": abs(
                    proposal_recovery.onset_time(sim["V"], segment.time)
                    - proposal_recovery.onset_time(segment.V, segment.time)
                ),
                **event_metrics,
            }
        )
    return rows


def rows_to_frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([{key: row.get(key) for key in columns} for row in rows])
    return frame.sort_values(["split", "step_name"], kind="stable").reset_index(drop=True)


def rank_steps(frame: pd.DataFrame, metric: str, ascending: bool = True) -> dict[str, list[dict]]:
    valid = frame.loc[np.isfinite(frame[metric].to_numpy(dtype=float))].sort_values(metric, ascending=ascending, kind="stable")
    best = valid.head(3)[["step_name", "split", metric]].to_dict(orient="records")
    worst = valid.tail(3)[["step_name", "split", metric]].to_dict(orient="records")
    return {"best": best, "worst": worst}


def representative_note(best_step2_metric: float, best_step7_metric: float, all_metrics: pd.Series) -> str:
    q25 = float(all_metrics.quantile(0.25))
    q75 = float(all_metrics.quantile(0.75))
    notes = []
    for step_name, value in [("p5838_step2", best_step2_metric), ("p5838_step7", best_step7_metric)]:
        if value < q25:
            notes.append(f"{step_name} is better than the lower-quartile threshold for this metric.")
        elif value > q75:
            notes.append(f"{step_name} is worse than the upper-quartile threshold for this metric.")
        else:
            notes.append(f"{step_name} sits in the middle half of the step distribution for this metric.")
    return " ".join(notes)


def gallery_layout(n_panels: int) -> tuple[int, int]:
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def save_tau_gallery(full_rows: list[dict]) -> list[str]:
    nrows, ncols = gallery_layout(len(full_rows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, full_rows):
        ax.plot(row["rel_time"], row["observed_tau"], label="Observed tau", linewidth=1.2)
        ax.plot(row["rel_time"], row["predicted_tau"], label="Predicted tau", linewidth=1.1, linestyle="--")
        ax.set_title(f"{row['step_name']} ({row['split']})")
        ax.set_xlabel("Time since step start [s]")
        ax.set_ylabel("tau")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.03,
            "semi-observed: uses observed V(t), V_drive(t)",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    for ax in axes[len(full_rows) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = RESULTS_DIR / "multistep_tau_rollout_gallery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def save_velocity_gallery(rows: list[dict]) -> list[str]:
    nrows, ncols = gallery_layout(len(rows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        ax.plot(row["rel_time"], row["observed_v"], label="Observed V", linewidth=1.2)
        ax.plot(row["rel_time"], row["predicted_v"], label="Predicted V", linewidth=1.1, linestyle="--")
        ax.set_title(f"{row['step_name']} ({row['split']})")
        ax.set_xlabel("Time since step start [s]")
        ax.set_ylabel("V")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(rows) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = RESULTS_DIR / "multistep_velocity_rollout_gallery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def save_exact_gallery(rows: list[dict]) -> list[str]:
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 3.2 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, rows):
        ax_tau, ax_v = row_axes
        ax_tau.plot(row["rel_time"], row["observed_tau"], label="Observed tau", linewidth=1.1)
        ax_tau.plot(row["rel_time"], row["predicted_tau"], label="Predicted tau", linewidth=1.0, linestyle="--")
        ax_tau.set_title(f"{row['step_name']} ({row['split']}) exact tau")
        ax_tau.set_xlabel("Time since step start [s]")
        ax_tau.set_ylabel("tau")
        ax_tau.grid(True, alpha=0.3)
        ax_v.plot(row["rel_time"], row["observed_v"], label="Observed V", linewidth=1.1)
        ax_v.plot(row["rel_time"], row["predicted_v"], label="Predicted V", linewidth=1.0, linestyle="--")
        ax_v.set_title(f"{row['step_name']} ({row['split']}) exact V")
        ax_v.set_xlabel("Time since step start [s]")
        ax_v.set_ylabel("V")
        ax_v.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = RESULTS_DIR / "multistep_exact_rsf_gallery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def save_phaseplots(exact_rows: list[dict]) -> list[str]:
    selected_names = ["p5838_step2", "p5838_step3", "p5838_step7", "p5838_step9"]
    selected = [row for row in exact_rows if row["step_name"] in selected_names]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, selected):
        ax.plot(row["observed_v"], row["observed_tau"], label="Observed", linewidth=1.1)
        ax.plot(row["predicted_v"], row["predicted_tau"], label="Exact RSF", linewidth=1.0, linestyle="--")
        ax.set_title(f"{row['step_name']} ({row['split']})")
        ax.set_xlabel("V")
        ax.set_ylabel("tau")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = RESULTS_DIR / "multistep_phaseplots.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def write_summary(
    tau_frame: pd.DataFrame,
    velocity_frame: pd.DataFrame,
    exact_frame: pd.DataFrame,
    excluded_steps: list[dict],
    figures: list[str],
) -> dict:
    tau_rank = rank_steps(tau_frame, "tau_rollout_rmse", ascending=True)
    velocity_rank = rank_steps(velocity_frame, "velocity_rollout_rmse", ascending=True)
    exact_rank = rank_steps(exact_frame, "velocity_rollout_rmse", ascending=True)
    tau_rep = representative_note(
        float(tau_frame.loc[tau_frame["step_name"] == "p5838_step2", "tau_rollout_rmse"].iloc[0]),
        float(tau_frame.loc[tau_frame["step_name"] == "p5838_step7", "tau_rollout_rmse"].iloc[0]),
        tau_frame["tau_rollout_rmse"],
    )
    exact_rep = representative_note(
        float(exact_frame.loc[exact_frame["step_name"] == "p5838_step2", "velocity_rollout_rmse"].iloc[0]),
        float(exact_frame.loc[exact_frame["step_name"] == "p5838_step7", "velocity_rollout_rmse"].iloc[0]),
        exact_frame["velocity_rollout_rmse"],
    )
    payload = {
        "usable_steps": DEFAULT_STEP_ORDER,
        "train_steps": tau_frame.loc[tau_frame["split"] == "train", "step_name"].tolist(),
        "holdout_steps": tau_frame.loc[tau_frame["split"] == "holdout", "step_name"].tolist(),
        "excluded_steps_for_theta_exact_only": excluded_steps,
        "tau_rankings": tau_rank,
        "reduced_velocity_rankings": velocity_rank,
        "exact_rsf_rankings": exact_rank,
        "representativeness": {
            "tau_equation": tau_rep,
            "exact_rsf_velocity": exact_rep,
        },
        "figures": figures,
    }
    summary_lines = [
        "# Multistep Rollout Summary",
        "",
        "## Usable steps",
        "- All evaluated RSFit-aligned `p5838` steps: `" + ", ".join(DEFAULT_STEP_ORDER) + "`",
        "- Train steps: `" + ", ".join(payload["train_steps"]) + "`",
        "- Holdout steps: `" + ", ".join(payload["holdout_steps"]) + "`",
        "",
        "## Exclusions / caveats",
    ]
    if excluded_steps:
        for row in excluded_steps:
            summary_lines.append(f"- `{row['step_name']}` is excluded from theta-usable proposal-identifiability subsets because `{row['theta_reason']}`.")
    else:
        summary_lines.append("- No additional exclusions were needed for this evaluation package.")
    summary_lines.extend(
        [
            "",
            "## Best and worst tau steps",
            f"- Best tau rollout steps by RMSE: `{json.dumps(payload['tau_rankings']['best'])}`",
            f"- Worst tau rollout steps by RMSE: `{json.dumps(payload['tau_rankings']['worst'])}`",
            "",
            "## Best and worst reduced-velocity steps",
            f"- Best reduced-velocity rollout steps by RMSE: `{json.dumps(payload['reduced_velocity_rankings']['best'])}`",
            f"- Worst reduced-velocity rollout steps by RMSE: `{json.dumps(payload['reduced_velocity_rankings']['worst'])}`",
            "",
            "## Best and worst exact-RSF steps",
            f"- Best exact-RSF steps by velocity RMSE: `{json.dumps(payload['exact_rsf_rankings']['best'])}`",
            f"- Worst exact-RSF steps by velocity RMSE: `{json.dumps(payload['exact_rsf_rankings']['worst'])}`",
            "",
            "## Interpretation",
            "- The tau equation is evaluated in a semi-observed mode: `tau(t)` is forecast while observed `V(t)` and `V_drive(t)` are supplied as exogenous inputs.",
            "- The reduced velocity and exact-RSF branches are full dynamic velocity rollouts on each step.",
            f"- Step representativeness for tau rollout: {tau_rep}",
            f"- Step representativeness for exact-RSF velocity rollout: {exact_rep}",
            "- If the exact-RSF gallery shows low stable fractions or large rollout RMSE on many steps, that should be read as evidence that the exact form remains fragile off the original holdout pair.",
            "",
        ]
    )
    (RESULTS_DIR / "multistep_rollout_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (RESULTS_DIR / "multistep_rollout_summary.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ensure_layout()
    print("[multistep-rollout] loading saved equations and aligned steps", flush=True)
    prepared_segments, _, proposal = load_all_prepared_segments()
    tau_model = proposal["tau_model"]
    velocity_model = proposal["final_velocity_model"]
    prepared_exact, exact_payload = load_exact_payloads()

    tau_rows = [tau_rollout_arrays(tau_model, segment_df, one_term=False) for segment_df in prepared_segments]
    tau_one_term_rows = [tau_rollout_arrays(tau_model, segment_df, one_term=True) for segment_df in prepared_segments]
    velocity_rows = [reduced_velocity_arrays(velocity_model, segment_df) for segment_df in prepared_segments]
    exact_rows = exact_rows_all_steps(prepared_exact, exact_payload)

    tau_frame = rows_to_frame(
        [
            {
                "step_name": row["step_name"],
                "split": row["split"],
                "n_samples": row["n_samples"],
                "duration_s": row["duration_s"],
                "tau_rollout_mse": row["tau_rollout_mse"],
                "tau_rollout_rmse": row["tau_rollout_rmse"],
                "tau_mean_abs_error": row["tau_mean_abs_error"],
                "tau_max_abs_error": row["tau_max_abs_error"],
                "tau_one_term_rollout_mse": one_term["tau_rollout_mse"],
                "tau_one_term_rollout_rmse": one_term["tau_rollout_rmse"],
                "notes": "semi_observed_tau_rollout",
            }
            for row, one_term in zip(tau_rows, tau_one_term_rows)
        ],
        [
            "step_name",
            "split",
            "n_samples",
            "duration_s",
            "tau_rollout_mse",
            "tau_rollout_rmse",
            "tau_mean_abs_error",
            "tau_max_abs_error",
            "tau_one_term_rollout_mse",
            "tau_one_term_rollout_rmse",
            "notes",
        ],
    )
    velocity_frame = rows_to_frame(
        [
            {
                "step_name": row["step_name"],
                "split": row["split"],
                "n_samples": row["n_samples"],
                "duration_s": row["duration_s"],
                "velocity_rollout_mse": row["velocity_rollout_mse"],
                "velocity_rollout_rmse": row["velocity_rollout_rmse"],
                "velocity_mean_abs_error": row["velocity_mean_abs_error"],
                "velocity_max_abs_error": row["velocity_max_abs_error"],
                "peak_timing_error_s": row["peak_timing_error_s"],
                "onset_timing_error_s": row["onset_timing_error_s"],
                "stable_fraction": row["stable_fraction"],
                "notes": "reduced_rsf_dynamic_rollout",
            }
            for row in velocity_rows
        ],
        [
            "step_name",
            "split",
            "n_samples",
            "duration_s",
            "velocity_rollout_mse",
            "velocity_rollout_rmse",
            "velocity_mean_abs_error",
            "velocity_max_abs_error",
            "peak_timing_error_s",
            "onset_timing_error_s",
            "stable_fraction",
            "notes",
        ],
    )
    exact_frame = rows_to_frame(
        [
            {
                "step_name": row["step_name"],
                "split": row["split"],
                "n_samples": row["n_samples"],
                "duration_s": row["duration_s"],
                "tau_rollout_mse": row["tau_rollout_mse"],
                "tau_rollout_rmse": row["tau_rmse"],
                "tau_mean_abs_error": row["tau_mean_abs_error"],
                "tau_max_abs_error": row["tau_max_abs_error"],
                "velocity_rollout_mse": row["velocity_rollout_mse"],
                "velocity_rollout_rmse": row["velocity_rollout_rmse"],
                "velocity_mean_abs_error": row["velocity_mean_abs_error"],
                "velocity_max_abs_error": row["velocity_max_abs_error"],
                "peak_timing_error_s": row["peak_timing_error_s"],
                "onset_timing_error_s": row["onset_timing_error_s"],
                "stable_fraction": row["stable_fraction"],
                "notes": "exact_rsf_saved_best_multistart",
            }
            for row in exact_rows
        ],
        [
            "step_name",
            "split",
            "n_samples",
            "duration_s",
            "tau_rollout_mse",
            "tau_rollout_rmse",
            "tau_mean_abs_error",
            "tau_max_abs_error",
            "velocity_rollout_mse",
            "velocity_rollout_rmse",
            "velocity_mean_abs_error",
            "velocity_max_abs_error",
            "peak_timing_error_s",
            "onset_timing_error_s",
            "stable_fraction",
            "notes",
        ],
    )

    tau_frame.to_csv(RESULTS_DIR / "multistep_tau_table.csv", index=False)
    velocity_frame.to_csv(RESULTS_DIR / "multistep_velocity_table.csv", index=False)
    exact_frame.to_csv(RESULTS_DIR / "multistep_exact_rsf_table.csv", index=False)

    figures: list[str] = []
    figures.extend(save_tau_gallery(tau_rows))
    figures.extend(save_velocity_gallery(velocity_rows))
    figures.extend(save_exact_gallery(exact_rows))
    figures.extend(save_phaseplots(exact_rows))

    excluded_steps = [
        {"step_name": str(row["step_name"]), "theta_reason": str(row["theta_reason"])}
        for row in proposal["inclusion_rows"]
        if str(row["step_name"]).startswith("p5838_step") and not bool(row["theta_event_valid"])
    ]
    write_summary(tau_frame, velocity_frame, exact_frame, excluded_steps, figures)
    print("[multistep-rollout] wrote multistep tables, galleries, and summary", flush=True)


if __name__ == "__main__":
    main()
