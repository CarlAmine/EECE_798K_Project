from __future__ import annotations

import json
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

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_proposal_equation_recovery as proposal_recovery
from scripts import utah_forge_reviewer_ablation as reviewer_ablation


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    return value


def load_segments_without_writing() -> tuple[dict[str, pd.DataFrame], dict[str, delay_ref.RSFitStep], dict]:
    state_df, _ = base.load_p5838_state()
    steps = delay_ref.load_rsfit_steps()
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
    rsfit_globals = reviewer_ablation.load_rsfit_globals()
    return segments, steps, rsfit_globals


def load_holdout_segments() -> tuple[list[pd.DataFrame], dict[str, str], dict]:
    segments, steps, rsfit_globals = load_segments_without_writing()
    acoustic_name = None
    holdout_names = list(reviewer_ablation.HOLDOUT_STEPS)
    prepared_segments: list[pd.DataFrame] = []
    for step_name in holdout_names:
        prepared, _, _ = proposal_recovery.prepare_segment_with_rsf(segments[step_name], steps[step_name], rsfit_globals, acoustic_name)
        prepared_segments.append(prepared)
    split_info = {
        "train_steps": ", ".join(reviewer_ablation.TRAIN_STEPS),
        "holdout_steps": ", ".join(holdout_names),
    }
    context = {
        "steps": steps,
        "rsfit_globals": rsfit_globals,
        "holdout_names": holdout_names,
    }
    return prepared_segments, split_info, context


def load_tau_model() -> dict:
    path = RESULTS_DIR / "proposal_equation_recovery.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    tau_model = payload["tau_model"]
    return tau_model


def semi_observed_rollout_arrays(tau_model: dict, holdout_segments: list[pd.DataFrame]) -> list[dict]:
    coefficients = tau_model["coefficients_physical"]
    rows = []
    for segment_df in holdout_segments:
        time = segment_df["time"].to_numpy(dtype=float)
        rel_time = time - float(time[0])
        observed_tau = segment_df["tau"].to_numpy(dtype=float)
        observed_v = segment_df["V"].to_numpy(dtype=float)
        observed_v_drive = segment_df["V_drive"].to_numpy(dtype=float)

        def rhs(state: np.ndarray, t_value: float) -> list[float]:
            v_now = float(np.interp(t_value, time, observed_v))
            v_drive_now = float(np.interp(t_value, time, observed_v_drive))
            return [
                coefficients.get("1", 0.0)
                + coefficients.get("V", 0.0) * v_now
                + coefficients.get("V_drive_minus_V", 0.0) * (v_drive_now - v_now)
            ]

        tau_roll = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), [float(observed_tau[0])], time).reshape(-1)
        abs_error = np.abs(tau_roll - observed_tau)
        mse = float(np.mean((tau_roll - observed_tau) ** 2))
        rows.append(
            {
                "step_name": str(segment_df["step_name"].iloc[0]),
                "time": time,
                "rel_time": rel_time,
                "observed_tau": observed_tau,
                "predicted_tau": tau_roll,
                "observed_v": observed_v,
                "observed_v_drive": observed_v_drive,
                "abs_tau_error": abs_error,
                "tau_rollout_mse": mse,
            }
        )
    return rows


def save_tau_rollout_examples(rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.5 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, rows):
        ax_tau, ax_err = row_axes
        ax_tau.plot(row["rel_time"], row["observed_tau"], label="Observed tau", linewidth=1.4)
        ax_tau.plot(row["rel_time"], row["predicted_tau"], label="Predicted tau", linewidth=1.2, linestyle="--")
        ax_tau.set_title(f"{row['step_name']} tau rollout")
        ax_tau.set_xlabel("Time since step start [s]")
        ax_tau.set_ylabel("tau")
        ax_tau.grid(True, alpha=0.3)
        ax_tau.legend(loc="best")
        ax_tau.text(
            0.02,
            0.03,
            f"MSE={row['tau_rollout_mse']:.4f}",
            transform=ax_tau.transAxes,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        ax_err.plot(row["rel_time"], row["abs_tau_error"], color="tab:red", linewidth=1.2)
        ax_err.set_title(f"{row['step_name']} absolute tau error")
        ax_err.set_xlabel("Time since step start [s]")
        ax_err.set_ylabel("|tau_pred - tau_obs|")
        ax_err.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tau_rollout_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tau_v_examples(rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.5 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, rows):
        ax_tau, ax_v = row_axes
        ax_tau.plot(row["rel_time"], row["observed_tau"], label="Observed tau", linewidth=1.4)
        ax_tau.plot(row["rel_time"], row["predicted_tau"], label="Predicted tau", linewidth=1.2, linestyle="--")
        ax_tau.set_title(f"{row['step_name']} tau(t)")
        ax_tau.set_xlabel("Time since step start [s]")
        ax_tau.set_ylabel("tau")
        ax_tau.grid(True, alpha=0.3)
        ax_tau.legend(loc="best")

        ax_v.plot(row["rel_time"], row["observed_v"], label="Observed V", linewidth=1.3, color="tab:green")
        ax_v.plot(row["rel_time"], row["observed_v_drive"], label="Supplied V_drive", linewidth=1.1, color="tab:orange", linestyle=":")
        ax_v.set_title(f"{row['step_name']} supplied velocity signal")
        ax_v.set_xlabel("Time since step start [s]")
        ax_v.set_ylabel("V")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="best")
        ax_v.text(
            0.02,
            0.03,
            "Semi-observed rollout uses observed V(t)\nand V_drive(t) as inputs; no V forecast is produced.",
            transform=ax_v.transAxes,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tau_v_rollout_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_phaseplot_examples(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5), sharex=False, sharey=False)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        ax.plot(row["observed_v"], row["observed_tau"], label="Observed trajectory", linewidth=1.4)
        ax.plot(row["observed_v"], row["predicted_tau"], label="Predicted tau on observed V path", linewidth=1.2, linestyle="--")
        ax.set_title(f"{row['step_name']} phase plot")
        ax.set_xlabel("V")
        ax.set_ylabel("tau")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "phaseplot_rollout_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows: list[dict], tau_model: dict, split_info: dict) -> None:
    explanation = {
        "metric_source_file": "scripts/utah_forge_proposal_equation_recovery.py",
        "metric_source_function": "semi_observed_tau_rollout",
        "semi_observed_definition": (
            "The tau ODE is integrated forward while using observed V(t) and observed V_drive(t) from the holdout event "
            "as exogenous inputs. Only tau is forecast in this check; V is not predicted."
        ),
        "step_definition": {
            "step2": "The holdout RSFit-aligned Utah FORGE event p5838_step2.",
            "step7": "The holdout RSFit-aligned Utah FORGE event p5838_step7.",
        },
        "split_info": split_info,
        "units_and_scaling": {
            "tau": "Same physical tau units used in the prepared Utah FORGE state; no extra normalization is applied inside semi_observed_tau_rollout.",
            "time": "seconds",
            "mse": "mean squared error in tau units squared, computed directly as mean((tau_roll - observed_tau)^2).",
        },
        "rows": [
            {
                "step_name": row["step_name"],
                "tau_rollout_mse": row["tau_rollout_mse"],
                "n_samples": int(len(row["time"])),
                "duration_s": float(row["rel_time"][-1]),
                "tau_rmse": float(np.sqrt(row["tau_rollout_mse"])),
                "mean_abs_tau_error": float(np.mean(row["abs_tau_error"])),
                "max_abs_tau_error": float(np.max(row["abs_tau_error"])),
            }
            for row in rows
        ],
        "tau_equation_used": tau_model["exact_equation"],
        "tau_one_term_equation": tau_model["one_term_equation"],
    }
    (RESULTS_DIR / "rollout_metric_explanation.json").write_text(json.dumps(json_ready(explanation), indent=2), encoding="utf-8")

    md_lines = [
        "# Rollout Metric Explanation",
        "",
        "## Exact code definition",
        "- Metric source file: `scripts/utah_forge_proposal_equation_recovery.py`",
        "- Metric source function: `semi_observed_tau_rollout()`",
        "- Exact computation: for each holdout segment, the code integrates the tau ODE forward with `odeint`, starting from the observed initial tau, and then computes `mean((tau_roll - observed_tau)^2)`.",
        "",
        "## What semi-observed means here",
        "- `semi_observed` means tau is rolled forward while `V(t)` and `V_drive(t)` are not predicted; they are supplied from the observed holdout event as time-varying inputs.",
        "- So this is a tau-only rollout check for Equation (1), not a full joint tau-V forecast.",
        "",
        "## What step2 and step7 mean",
        "- `step2` means the holdout event `p5838_step2`.",
        "- `step7` means the holdout event `p5838_step7`.",
        "- In this codebase they are holdout RSFit-aligned Utah FORGE step events, not forecast horizons or checkpoints.",
        "",
        "## Which split they are computed on",
        f"- Train steps: `{split_info['train_steps']}`",
        f"- Holdout steps: `{split_info['holdout_steps']}`",
        "- The semi-observed tau rollout metrics are computed only on the holdout events `p5838_step2` and `p5838_step7`.",
        "",
        "## Units and scaling",
        "- `tau` is used in the same physical units as the prepared Utah FORGE state in the proposal-recovery workflow.",
        "- `time` is in seconds.",
        "- `tau_rollout_mse` is plain mean squared tau error, so its units are tau-units squared.",
        "- No extra normalization is applied inside `semi_observed_tau_rollout()`.",
        "",
        "## Plain-English meaning",
        f"- `semi_observed_tau_rollout_mse_step2 = {rows[0]['tau_rollout_mse']:.12f}` means: on holdout event `p5838_step2`, if we drive the recovered tau law using the observed velocity path from that event, the average squared tau error over the event is about `{rows[0]['tau_rollout_mse']:.4f}`.",
        f"- `semi_observed_tau_rollout_mse_step7 = {rows[1]['tau_rollout_mse']:.12f}` means: on holdout event `p5838_step7`, the same tau-only rollout check gives an average squared tau error of about `{rows[1]['tau_rollout_mse']:.4f}`.",
        f"- Since `{rows[1]['tau_rollout_mse']:.4f}` is smaller than `{rows[0]['tau_rollout_mse']:.4f}`, the recovered tau law tracks `p5838_step7` better than `p5838_step2` under this semi-observed rollout definition.",
        "",
    ]
    (RESULTS_DIR / "rollout_metric_explanation.md").write_text("\n".join(md_lines), encoding="utf-8")

    summary_lines = [
        "# Rollout Visualization Summary",
        "",
        "## Figures created",
        "- `tau_rollout_examples.png`: observed vs predicted tau and absolute tau error for the two holdout events.",
        "- `tau_v_rollout_examples.png`: observed vs predicted tau plus the observed `V(t)` and supplied `V_drive(t)` used by the semi-observed rollout.",
        "- `phaseplot_rollout_examples.png`: phase plots comparing observed `tau(V)` against predicted tau traced along the observed velocity path.",
        "",
        "## Interpretation",
        "- These plots visualize the same semi-observed rollout metric reported in the proposal-recovery report.",
        "- The predicted tau curves come from integrating Equation (1) forward with observed `V(t)` and `V_drive(t)` supplied from the holdout event.",
        "- No separate `V(t)` prediction is available in this metric because the rollout is intentionally semi-observed.",
        "- The event titles `p5838_step2` and `p5838_step7` should be read as holdout event identifiers, not forecast horizons.",
        "",
    ]
    (RESULTS_DIR / "rollout_visualization_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    print("[rollout-explainer] loading best tau equation and holdout events", flush=True)
    tau_model = load_tau_model()
    holdout_segments, split_info, _ = load_holdout_segments()
    rows = semi_observed_rollout_arrays(tau_model, holdout_segments)
    save_tau_rollout_examples(rows)
    save_tau_v_examples(rows)
    save_phaseplot_examples(rows)
    write_outputs(rows, tau_model, split_info)
    print("[rollout-explainer] wrote explanation and visualization artifacts", flush=True)


if __name__ == "__main__":
    main()
