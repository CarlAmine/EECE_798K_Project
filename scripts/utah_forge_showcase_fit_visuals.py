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

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_proposal_equation_recovery as proposal_recovery
from scripts import utah_forge_reviewer_ablation as reviewer_ablation
from src.derivatives import derivative_savgol
from src.exact_rsf import load_checkpoint, simulate_exact_rsf_segment


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
EXACT_CKPT_DIR = RESULTS_DIR / "exact_rsf_multistart_checkpoints"


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def load_holdout_segments() -> list[pd.DataFrame]:
    segments, steps, rsfit_globals = load_segments_without_writing()
    prepared = []
    for step_name in reviewer_ablation.HOLDOUT_STEPS:
        segment_df = segments[step_name]
        prepared_df, _, _ = proposal_recovery.prepare_segment_with_rsf(segment_df, steps[step_name], rsfit_globals, None)
        prepared.append(prepared_df)
    return prepared


def load_payloads() -> tuple[dict, dict, dict, dict]:
    proposal = json.loads((RESULTS_DIR / "proposal_equation_recovery.json").read_text(encoding="utf-8"))
    multistart = json.loads((RESULTS_DIR / "exact_rsf_multistart_summary.json").read_text(encoding="utf-8"))
    prepared_exact = load_checkpoint(RESULTS_DIR / "exact_rsf_inverse_fit_checkpoints", "prepared_exact_segments")
    best_start = int(multistart["best_run"]["start_index"])
    exact_payload = load_checkpoint(EXACT_CKPT_DIR, f"exact_fit_multistart_{best_start}")
    if prepared_exact is None or exact_payload is None:
        raise RuntimeError("Missing exact RSF checkpoint data.")
    return proposal, multistart, prepared_exact, exact_payload


def semi_observed_tau_arrays(tau_model: dict, holdout_segments: list[pd.DataFrame]) -> list[dict]:
    coefficients = tau_model["coefficients_physical"]
    rows = []
    for segment_df in holdout_segments:
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

        predicted_tau = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), [float(observed_tau[0])], time).reshape(-1)
        predicted_dtau = (
            coefficients.get("1", 0.0)
            + coefficients.get("V", 0.0) * observed_v
            + coefficients.get("V_drive_minus_V", 0.0) * (observed_v_drive - observed_v)
        )
        rows.append(
            {
                "step_name": str(segment_df["step_name"].iloc[0]),
                "time": time,
                "rel_time": rel_time,
                "observed_tau": observed_tau,
                "predicted_tau": predicted_tau,
                "observed_dtau": observed_dtau,
                "predicted_dtau": predicted_dtau,
                "observed_v": observed_v,
                "abs_tau_error": np.abs(predicted_tau - observed_tau),
            }
        )
    return rows


def rollout_velocity_series(model_row: dict, segment_df: pd.DataFrame) -> dict:
    coefficients = model_row["coefficients_physical"]
    feature_names = model_row["feature_names"]
    time = segment_df["time"].to_numpy(dtype=float)
    observed_v = segment_df["V"].to_numpy(dtype=float)
    observed_tau = segment_df["tau"].to_numpy(dtype=float)
    observed_sigma = segment_df["sigmaN"].to_numpy(dtype=float)
    observed_theta = segment_df["theta"].to_numpy(dtype=float) if "theta" in segment_df.columns else np.full(len(segment_df), np.nan)
    observed_acoustic = segment_df["acoustic_feature"].to_numpy(dtype=float) if "acoustic_feature" in segment_df.columns else np.full(len(segment_df), np.nan)
    observed_dv = segment_df["dV_dt"].to_numpy(dtype=float)
    v0 = segment_df["V0"].to_numpy(dtype=float)
    dc = segment_df["Dc"].to_numpy(dtype=float)
    use_memory = "deltaS_orth" in feature_names
    orth_map = model_row.get("deltaS_orth_map", {})

    def exogenous_row(t_value: float, current_v: float, current_s: float) -> dict[str, float]:
        tau = float(np.interp(t_value, time, observed_tau))
        sigma = float(np.interp(t_value, time, observed_sigma))
        theta = float(np.interp(t_value, time, observed_theta)) if np.isfinite(observed_theta).any() else 1.0
        acoustic = float(np.interp(t_value, time, observed_acoustic)) if np.isfinite(observed_acoustic).any() else 0.0
        v0_now = max(float(np.interp(t_value, time, v0)), proposal_recovery.EPS)
        dc_now = max(float(np.interp(t_value, time, dc)), proposal_recovery.EPS)
        delta_s_orth = current_s - (
            orth_map.get("intercept", 0.0)
            + orth_map.get("tau", 0.0) * tau
            + orth_map.get("sigmaN_logV", 0.0) * sigma * math.log(max(current_v, proposal_recovery.EPS) / v0_now)
        )
        return {
            "tau": tau,
            "sigmaN": sigma,
            "theta": max(theta, proposal_recovery.EPS),
            "V": max(current_v, proposal_recovery.EPS),
            "V0": v0_now,
            "Dc": dc_now,
            "deltaS_local": current_s,
            "deltaS_orth": delta_s_orth,
            "acoustic_feature": acoustic,
        }

    def rhs(state: np.ndarray, t_value: float) -> list[float]:
        current_v = max(float(state[0]), proposal_recovery.EPS)
        current_s = float(state[1]) if use_memory else 0.0
        row = exogenous_row(t_value, current_v, current_s)
        dvdt = coefficients.get("1", 0.0)
        for feature_name in feature_names:
            if feature_name == "1":
                continue
            dvdt += coefficients.get(feature_name, 0.0) * proposal_recovery.feature_value(feature_name, pd.Series(row))
        if use_memory:
            return [dvdt, current_v]
        return [dvdt]

    initial_state = [float(observed_v[0]), 0.0] if use_memory else [float(observed_v[0])]
    solution = odeint(lambda state, t_val: rhs(np.asarray(state, dtype=float), t_val), initial_state, time)
    predicted_v = solution[:, 0]
    predicted_dv = derivative_savgol(predicted_v, t=time, window=15, polyorder=3)
    return {
        "step_name": str(segment_df["step_name"].iloc[0]),
        "time": time,
        "rel_time": time - float(time[0]),
        "observed_v": observed_v,
        "predicted_v": predicted_v,
        "observed_tau": observed_tau,
        "abs_v_error": np.abs(predicted_v - observed_v),
        "observed_dv": observed_dv,
        "predicted_dv": predicted_dv,
    }


def exact_fit_series(prepared_exact: dict, exact_payload: dict) -> list[dict]:
    holdout_segments = prepared_exact["holdout_segments"]
    params = exact_payload["parameters"]
    acoustic_z = exact_payload["acoustic_zscores"]
    rows = []
    for segment in holdout_segments:
        sim = simulate_exact_rsf_segment(segment, params, delta_log_theta0=0.0, acoustic_z=acoustic_z.get(segment.step_name, 0.0))
        predicted_dtau = derivative_savgol(sim["tau"], t=segment.time, window=15, polyorder=3)
        predicted_dv = derivative_savgol(sim["V"], t=segment.time, window=15, polyorder=3)
        rows.append(
            {
                "step_name": segment.step_name,
                "time": segment.time,
                "rel_time": segment.time - float(segment.time[0]),
                "observed_tau": segment.tau,
                "predicted_tau": sim["tau"],
                "observed_v": segment.V,
                "predicted_v": sim["V"],
                "observed_theta": segment.theta_proxy,
                "predicted_theta": sim["theta"],
                "abs_tau_error": np.abs(sim["tau"] - segment.tau),
                "abs_v_error": np.abs(sim["V"] - segment.V),
                "observed_dtau": segment.dtau_dt,
                "predicted_dtau": predicted_dtau,
                "observed_dv": segment.dV_dt,
                "predicted_dv": predicted_dv,
            }
        )
    return rows


def save_showcase_tau_fit(tau_rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(tau_rows), 2, figsize=(12, 4.5 * len(tau_rows)), sharex=False)
    if len(tau_rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, tau_rows):
        ax_fit, ax_err = row_axes
        ax_fit.plot(row["rel_time"], row["observed_tau"], label="Observed tau", linewidth=1.4)
        ax_fit.plot(row["rel_time"], row["predicted_tau"], label="Predicted tau", linewidth=1.2, linestyle="--")
        ax_fit.set_title(f"{row['step_name']} semi-observed tau rollout")
        ax_fit.set_xlabel("Time since step start [s]")
        ax_fit.set_ylabel("tau")
        ax_fit.grid(True, alpha=0.3)
        ax_fit.legend(loc="best")
        ax_fit.text(
            0.02,
            0.03,
            "Observed V(t) and V_drive(t)\nare supplied exogenous inputs.",
            transform=ax_fit.transAxes,
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
    fig.savefig(RESULTS_DIR / "showcase_tau_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_showcase_velocity_fit(reduced_rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(reduced_rows), 2, figsize=(12, 4.5 * len(reduced_rows)), sharex=False)
    if len(reduced_rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, reduced_rows):
        ax_fit, ax_err = row_axes
        ax_fit.plot(row["rel_time"], row["observed_v"], label="Observed V", linewidth=1.4)
        ax_fit.plot(row["rel_time"], row["predicted_v"], label="Predicted V", linewidth=1.2, linestyle="--")
        ax_fit.set_title(f"{row['step_name']} reduced-RSF velocity rollout")
        ax_fit.set_xlabel("Time since step start [s]")
        ax_fit.set_ylabel("V")
        ax_fit.grid(True, alpha=0.3)
        ax_fit.legend(loc="best")
        ax_err.plot(row["rel_time"], row["abs_v_error"], color="tab:red", linewidth=1.2)
        ax_err.set_title(f"{row['step_name']} absolute velocity error")
        ax_err.set_xlabel("Time since step start [s]")
        ax_err.set_ylabel("|V_pred - V_obs|")
        ax_err.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "showcase_velocity_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_showcase_exact_rsf_fit(exact_rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(exact_rows), 2, figsize=(12, 4.8 * len(exact_rows)), sharex=False)
    if len(exact_rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, exact_rows):
        ax_tau, ax_v = row_axes
        ax_tau.plot(row["rel_time"], row["observed_tau"], label="Observed tau", linewidth=1.3)
        ax_tau.plot(row["rel_time"], row["predicted_tau"], label="Predicted tau", linewidth=1.2, linestyle="--")
        ax_tau.set_title(f"{row['step_name']} exact-RSF tau")
        ax_tau.set_xlabel("Time since step start [s]")
        ax_tau.set_ylabel("tau")
        ax_tau.grid(True, alpha=0.3)
        ax_tau.legend(loc="best")

        ax_v.plot(row["rel_time"], row["observed_v"], label="Observed V", linewidth=1.3)
        ax_v.plot(row["rel_time"], row["predicted_v"], label="Predicted V", linewidth=1.2, linestyle="--")
        ax_v.set_title(f"{row['step_name']} exact-RSF velocity")
        ax_v.set_xlabel("Time since step start [s]")
        ax_v.set_ylabel("V")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "showcase_exact_rsf_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_showcase_phaseplot(exact_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(exact_rows), figsize=(6 * len(exact_rows), 5), sharex=False, sharey=False)
    if len(exact_rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, exact_rows):
        ax.plot(row["observed_v"], row["observed_tau"], label="Observed", linewidth=1.3)
        ax.plot(row["predicted_v"], row["predicted_tau"], label="Predicted exact RSF", linewidth=1.2, linestyle="--")
        ax.set_title(f"{row['step_name']} tau-V phase plot")
        ax.set_xlabel("V")
        ax.set_ylabel("tau")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "showcase_phaseplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_derivative_scatter(tau_rows: list[dict], reduced_rows: list[dict], exact_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    tau_obs = np.concatenate([row["observed_dtau"] for row in tau_rows])
    tau_pred = np.concatenate([row["predicted_dtau"] for row in tau_rows])
    red_obs = np.concatenate([row["observed_dv"] for row in reduced_rows])
    red_pred = np.concatenate([row["predicted_dv"] for row in reduced_rows])
    exact_obs = np.concatenate([row["observed_dv"] for row in exact_rows])
    exact_pred = np.concatenate([row["predicted_dv"] for row in exact_rows])
    panels = [
        (axes[0], tau_obs, tau_pred, "Equation (1) dtau/dt"),
        (axes[1], red_obs, red_pred, "Reduced fallback dV/dt"),
        (axes[2], exact_obs, exact_pred, "Exact RSF dV/dt"),
    ]
    for ax, obs, pred, title in panels:
        ax.scatter(obs, pred, s=6, alpha=0.3)
        lo = float(min(np.min(obs), np.min(pred)))
        hi = float(max(np.max(obs), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "showcase_derivative_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_explanation() -> None:
    lines = [
        "# Showcase Fit Explanation",
        "",
        "## What rollout means",
        "- A rollout means integrating the discovered differential equation forward in time and comparing the predicted trajectory to the observed holdout event.",
        "- For Equation (1), the rollout shown here is semi-observed: `tau(t)` is predicted, while observed `V(t)` and `V_drive(t)` are supplied as inputs.",
        "- For the reduced fallback Equation (2) and the exact RSF-looking fit, `V(t)` is rolled forward dynamically and compared against the observed velocity trajectory.",
        "",
        "## What derivative fit means",
        "- Derivative fit compares the observed numerical derivative, such as `dtau/dt` or `dV/dt`, against the derivative predicted by the equation at the same samples.",
        "- A good derivative fit means the equation matches instantaneous slopes well, even before full rollout is considered.",
        "- Rollout is stricter because small derivative errors can accumulate over time into larger trajectory errors.",
        "",
        "## Which plots correspond to which equations",
        "- `showcase_tau_fit.png`: best Equation (1), the compact spring-loading tau law.",
        "- `showcase_velocity_fit.png`: best final usable Equation (2), the reduced RSF fallback.",
        "- `showcase_exact_rsf_fit.png`: closest exact RSF-looking fit on holdout events.",
        "- `showcase_phaseplot.png`: tau-V phase portrait for the exact RSF-looking fit.",
        "- `showcase_derivative_scatter.png`: derivative-fit scatter for Equation (1), the reduced fallback, and the exact RSF-looking fit.",
        "",
        "## What a good fit looks like",
        "- In rollout plots, a good fit keeps predicted and observed curves close over most of the event and avoids early divergence.",
        "- In absolute-error plots, a good fit keeps the error low and prevents it from growing rapidly over time.",
        "- In derivative scatter plots, a good fit places points close to the diagonal line.",
        "",
        "## Where the exact RSF fit succeeds and fails",
        "- The exact RSF-looking fit succeeds by preserving the full RSF form and achieving reasonably good timing metrics, especially peak timing.",
        "- It fails as a final trusted model because its holdout trajectories are still unstable, parameter estimates are not robust across starts, and the identifiability diagnostics remain poor.",
        "- So it is strongest as a scientific showcase of the closest exact form, not as the final usable governing law.",
        "",
    ]
    (RESULTS_DIR / "showcase_fit_explanation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    print("[showcase-fit] loading saved best equations and checkpoints", flush=True)
    holdout_segments = load_holdout_segments()
    proposal, multistart, prepared_exact, exact_payload = load_payloads()
    tau_rows = semi_observed_tau_arrays(proposal["tau_model"], holdout_segments)
    reduced_rows = [rollout_velocity_series(proposal["final_velocity_model"], segment_df) for segment_df in holdout_segments]
    exact_rows = exact_fit_series(prepared_exact, exact_payload)
    save_showcase_tau_fit(tau_rows)
    save_showcase_velocity_fit(reduced_rows)
    save_showcase_exact_rsf_fit(exact_rows)
    save_showcase_phaseplot(exact_rows)
    save_derivative_scatter(tau_rows, reduced_rows, exact_rows)
    write_explanation()
    print("[showcase-fit] wrote showcase figures and explanation", flush=True)


if __name__ == "__main__":
    main()
