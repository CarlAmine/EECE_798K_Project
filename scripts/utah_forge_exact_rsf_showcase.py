from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exact_rsf import load_checkpoint, simulate_exact_rsf_segment


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
BASE_CKPT_DIR = RESULTS_DIR / "exact_rsf_inverse_fit_checkpoints"
MULTI_CKPT_DIR = RESULTS_DIR / "exact_rsf_multistart_checkpoints"


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


def load_context() -> dict:
    multi_summary = json.loads((RESULTS_DIR / "exact_rsf_multistart_summary.json").read_text(encoding="utf-8"))
    proposal = json.loads((RESULTS_DIR / "proposal_equation_recovery.json").read_text(encoding="utf-8"))
    exact_inverse = json.loads((RESULTS_DIR / "exact_rsf_inverse_fit.json").read_text(encoding="utf-8"))
    best_start_index = int(multi_summary["best_run"]["start_index"])
    best_payload = load_checkpoint(MULTI_CKPT_DIR, f"exact_fit_multistart_{best_start_index}")
    baseline_payload = load_checkpoint(BASE_CKPT_DIR, "exact_fit_base")
    prepared = load_checkpoint(BASE_CKPT_DIR, "prepared_exact_segments")
    if best_payload is None or baseline_payload is None or prepared is None:
        raise RuntimeError("Missing exact RSF checkpoints needed for showcase.")
    return {
        "multi_summary": multi_summary,
        "proposal": proposal,
        "exact_inverse": exact_inverse,
        "best_payload": best_payload,
        "baseline_payload": baseline_payload,
        "prepared": prepared,
        "best_start_index": best_start_index,
    }


def simulate_best_fit(prepared: dict, best_payload: dict) -> dict:
    train_segments = prepared["train_segments"]
    holdout_segments = prepared["holdout_segments"]
    params = best_payload["parameters"]
    acoustic_z = best_payload["acoustic_zscores"]
    theta_offsets = best_payload["theta_offsets_train"]

    train_sims = {}
    for segment in train_segments:
        train_sims[segment.step_name] = simulate_exact_rsf_segment(
            segment,
            params,
            delta_log_theta0=theta_offsets.get(segment.step_name, 0.0),
            acoustic_z=acoustic_z.get(segment.step_name, 0.0),
        )
    holdout_sims = {}
    for segment in holdout_segments:
        holdout_sims[segment.step_name] = simulate_exact_rsf_segment(
            segment,
            params,
            delta_log_theta0=0.0,
            acoustic_z=acoustic_z.get(segment.step_name, 0.0),
        )
    return {"train": train_sims, "holdout": holdout_sims}


def save_tau_rollout(holdout_segments, holdout_sims) -> None:
    fig, axes = plt.subplots(len(holdout_segments), 2, figsize=(12, 4.5 * len(holdout_segments)), sharex=False)
    if len(holdout_segments) == 1:
        axes = np.array([axes])
    for row_axes, segment in zip(axes, holdout_segments):
        sim = holdout_sims[segment.step_name]
        rel_time = segment.time - segment.time[0]
        tau_error = np.abs(sim["tau"] - segment.tau)
        row_axes[0].plot(rel_time, segment.tau, label="Observed tau", linewidth=1.3)
        row_axes[0].plot(rel_time, sim["tau"], label="Predicted tau", linewidth=1.2, linestyle="--")
        row_axes[0].set_title(f"{segment.step_name} tau rollout")
        row_axes[0].set_xlabel("Time since step start [s]")
        row_axes[0].set_ylabel("tau")
        row_axes[0].grid(True, alpha=0.3)
        row_axes[0].legend(loc="best")

        row_axes[1].plot(rel_time, tau_error, color="tab:red", linewidth=1.2)
        row_axes[1].set_title(f"{segment.step_name} |tau error|")
        row_axes[1].set_xlabel("Time since step start [s]")
        row_axes[1].set_ylabel("|tau_pred - tau_obs|")
        row_axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exact_rsf_showcase_tau_rollout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_velocity_rollout(holdout_segments, holdout_sims) -> None:
    fig, axes = plt.subplots(len(holdout_segments), 2, figsize=(12, 4.5 * len(holdout_segments)), sharex=False)
    if len(holdout_segments) == 1:
        axes = np.array([axes])
    for row_axes, segment in zip(axes, holdout_segments):
        sim = holdout_sims[segment.step_name]
        rel_time = segment.time - segment.time[0]
        v_error = np.abs(sim["V"] - segment.V)
        row_axes[0].plot(rel_time, segment.V, label="Observed V", linewidth=1.3)
        row_axes[0].plot(rel_time, sim["V"], label="Predicted V", linewidth=1.2, linestyle="--")
        row_axes[0].set_title(f"{segment.step_name} velocity rollout")
        row_axes[0].set_xlabel("Time since step start [s]")
        row_axes[0].set_ylabel("V")
        row_axes[0].grid(True, alpha=0.3)
        row_axes[0].legend(loc="best")

        row_axes[1].plot(rel_time, v_error, color="tab:red", linewidth=1.2)
        row_axes[1].set_title(f"{segment.step_name} |V error|")
        row_axes[1].set_xlabel("Time since step start [s]")
        row_axes[1].set_ylabel("|V_pred - V_obs|")
        row_axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exact_rsf_showcase_velocity_rollout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_phaseplot(holdout_segments, holdout_sims) -> None:
    fig, axes = plt.subplots(1, len(holdout_segments), figsize=(6 * len(holdout_segments), 5), sharex=False, sharey=False)
    if len(holdout_segments) == 1:
        axes = [axes]
    for ax, segment in zip(axes, holdout_segments):
        sim = holdout_sims[segment.step_name]
        ax.plot(segment.V, segment.tau, label="Observed", linewidth=1.3)
        ax.plot(sim["V"], sim["tau"], label="Predicted", linewidth=1.2, linestyle="--")
        ax.set_title(f"{segment.step_name} tau-V phase portrait")
        ax.set_xlabel("V")
        ax.set_ylabel("tau")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exact_rsf_showcase_phaseplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_theta_examples(train_segments, holdout_segments, train_sims, holdout_sims) -> None:
    examples = []
    if train_segments:
        examples.append(("train", train_segments[0], train_sims[train_segments[0].step_name]))
    if holdout_segments:
        examples.append(("holdout", holdout_segments[0], holdout_sims[holdout_segments[0].step_name]))
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 4 * len(examples)), sharex=False)
    if len(examples) == 1:
        axes = [axes]
    for ax, (split_name, segment, sim) in zip(axes, examples):
        rel_time = segment.time - segment.time[0]
        ax.plot(rel_time, segment.theta_proxy, label="RSFit theta proxy", linewidth=1.3)
        ax.plot(rel_time, sim["theta"], label="Simulated theta", linewidth=1.2, linestyle="--")
        ax.set_title(f"{segment.step_name} theta example ({split_name})")
        ax.set_xlabel("Time since step start [s]")
        ax.set_ylabel("theta")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exact_rsf_showcase_theta_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parameter_rows(best_payload: dict, multi_summary: dict) -> list[dict]:
    params = best_payload["parameters"]
    stability = multi_summary["parameter_stability"]
    rows = []
    for name in ["k", "m", "mu0", "a", "b", "Dc"]:
        rows.append(
            {
                "parameter": name,
                "value": float(params[name]),
                "mean_across_starts": float(stability[name]["mean"]),
                "std_across_starts": float(stability[name]["std"]),
                "cv_across_starts": float(stability[name]["cv"]),
            }
        )
    return rows


def build_comparison_table(best_payload: dict, proposal: dict, multi_summary: dict) -> pd.DataFrame:
    best_row = multi_summary["best_run"]
    reduced = proposal["velocity_models"]["B_reduced_rsf"]["best"]
    rows = [
        {
            "model": "exact_rsf_closest_exact_fit",
            "workflow_source": "exact_rsf_multistart_check -> fit_exact_rsf_inverse_model",
            "equation": best_payload["velocity_equation"],
            "optimization_success": bool(best_payload["optimization"]["success"]),
            "nfev": int(best_payload["optimization"]["nfev"]),
            "cost": float(best_payload["optimization"]["cost"]),
            "derivative_rmse": np.nan,
            "rollout_error": float(best_row["mean_holdout_error"]),
            "stable_fraction": float(best_row["mean_holdout_stable_fraction"]),
            "peak_timing_error_s": float(best_row["mean_holdout_peak_timing_error_s"]),
            "onset_timing_error_s": float(best_row["mean_holdout_onset_timing_error_s"]),
            "identifiability_status": "non-identifiable",
            "judgment": "closest exact fit but non-identifiable",
        },
        {
            "model": "reduced_rsf_final_fallback",
            "workflow_source": "proposal_equation_recovery",
            "equation": reduced["equation"],
            "optimization_success": True,
            "nfev": np.nan,
            "cost": np.nan,
            "derivative_rmse": float(reduced["holdout_rmse"]),
            "rollout_error": float(reduced["mean_rollout_mse"]),
            "stable_fraction": float(reduced["mean_stable_fraction"]),
            "peak_timing_error_s": float(reduced["mean_peak_timing_error_s"]),
            "onset_timing_error_s": float(reduced["mean_onset_timing_error_s"]),
            "identifiability_status": "usable reduced fallback",
            "judgment": "best final usable equation",
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(ctx: dict, sims: dict) -> None:
    multi_summary = ctx["multi_summary"]
    proposal = ctx["proposal"]
    best_payload = ctx["best_payload"]
    baseline_payload = ctx["baseline_payload"]
    prepared = ctx["prepared"]
    best_start_index = ctx["best_start_index"]
    best_row = multi_summary["best_run"]
    parameter_rows = build_parameter_rows(best_payload, multi_summary)
    comparison_df = build_comparison_table(best_payload, proposal, multi_summary)
    comparison_df.to_csv(RESULTS_DIR / "exact_rsf_showcase_table.csv", index=False)

    equations_text = "\n".join(
        [
            "Exact RSF-looking equation set (best multistart exact-form fit)",
            best_payload["tau_equation"],
            best_payload["velocity_equation"],
            best_payload["theta_equation"],
            "",
            "Reduced RSF fallback used as final best usable velocity law",
            proposal["velocity_models"]["B_reduced_rsf"]["best"]["equation"],
        ]
    )
    (RESULTS_DIR / "exact_rsf_showcase_equations.txt").write_text(equations_text, encoding="utf-8")

    report_json = {
        "origin": {
            "equation_source_workflow": "scripts/utah_forge_exact_rsf_multistart_check.py",
            "equation_source_function": "fit_exact_rsf_inverse_model in src/exact_rsf.py",
            "equation_source_run": {
                "best_start_index": best_start_index,
                "checkpoint_stage_name": f"exact_fit_multistart_{best_start_index}",
                "from_multistart_refinement": True,
                "from_base_exact_fit": False,
            },
            "files_containing_equation": [
                "results/utah_forge/exact_rsf_multistart_summary.json",
                "results/utah_forge/exact_rsf_multistart_check.md",
                f"results/utah_forge/exact_rsf_multistart_checkpoints/exact_fit_multistart_{best_start_index}.pkl",
            ],
        },
        "data_subset": {
            "train_steps": prepared["train_names"],
            "holdout_steps": prepared["holdout_names"],
            "n_train_events": len(prepared["train_segments"]),
            "n_holdout_events": len(prepared["holdout_segments"]),
            "acoustic_name_detected": prepared["acoustic_name"],
            "theta_treatment": "latent state evolved dynamically during fitting; RSFit theta proxy used for structure confirmation and diagnostics, not directly supplied as the fitted state trajectory",
        },
        "optimization": {
            "optimized_parameters": ["k", "m", "mu0", "a", "b", "Dc"] + [f"delta_log_theta0:{seg.step_name}" for seg in prepared["train_segments"]],
            "loss_type": "mixed trajectory-based loss with tau(t) residuals, V(t) residuals, a small derivative-consistency penalty on dV/dt, and a regularization penalty on event theta offsets",
            "constraints": [
                "k > 0",
                "m > 0",
                "a > 0",
                "b > 0",
                "Dc > 0",
                "theta(t) > 0 by positive initialization and clipped integration",
                "V(t) > 0 by positive clipped integration",
            ],
        },
        "best_payload": best_payload,
        "baseline_payload": baseline_payload,
        "best_run_summary": best_row,
        "parameter_table": parameter_rows,
        "parameter_stability": multi_summary["parameter_stability"],
        "comparison_to_reduced_fallback": json_ready(comparison_df),
        "final_warning": {
            "sigma_too_constant": bool(best_row["sigma_too_constant"]),
            "parameter_confounding_flag": bool(best_row["parameter_confounding_flag"]),
            "jtj_condition_number": float(best_row["jtj_condition_number"]),
            "jtj_rank": int(best_row["jtj_rank"]),
            "final_statement": multi_summary["final_statement"],
        },
    }
    (RESULTS_DIR / "exact_rsf_showcase_report.json").write_text(json.dumps(json_ready(report_json), indent=2), encoding="utf-8")

    report_lines = [
        "# Exact RSF Showcase Report",
        "",
        "## 1. Exact equation set",
        f"- `{best_payload['tau_equation']}`",
        f"- `{best_payload['velocity_equation']}`",
        f"- `{best_payload['theta_equation']}`",
        "",
        "## 2. Where it came from",
        "- Workflow source: `scripts/utah_forge_exact_rsf_multistart_check.py`",
        "- Core fitter: `fit_exact_rsf_inverse_model()` in `src/exact_rsf.py`",
        f"- Winning run: `start_index = {best_start_index}`",
        f"- Winning checkpoint: `results/utah_forge/exact_rsf_multistart_checkpoints/exact_fit_multistart_{best_start_index}.pkl`",
        "- It came from the bounded multistart refinement, not from the original base exact fit.",
        "",
        "## 3. How it was obtained",
        f"- Data subset: train steps `{', '.join(prepared['train_names'])}`; holdout steps `{', '.join(prepared['holdout_names'])}`.",
        "- Events were selected by the exact-RSF workflow from the RSFit-aligned Utah FORGE step windows, then downsampled/smoothed into `ExactRSFSegment` objects.",
        "- Theta was treated as a latent dynamical state during fitting, with one event-specific `delta_log_theta0` offset per training event.",
        "- The optimized global parameters were `k, m, mu0, a, b, Dc`.",
        "- The loss was mixed: trajectory matching on `tau(t)` and `V(t)`, plus a small derivative-consistency penalty on `dV/dt`, plus a penalty on theta offsets.",
        "- Constraints enforced were `k > 0`, `m > 0`, `a > 0`, `b > 0`, `Dc > 0`, with positive `theta(t)` and positive `V(t)` maintained during simulation.",
        "- Multistart generated four nearby initializations and chose the best exact-form fit by lowest final cost.",
        "",
        "## 4. Parameter table",
        "| parameter | value | mean_across_starts | std_across_starts | cv_across_starts |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in parameter_rows:
        report_lines.append(
            f"| {row['parameter']} | {row['value']:.6e} | {row['mean_across_starts']:.6e} | {row['std_across_starts']:.6e} | {row['cv_across_starts']:.6e} |"
        )
    report_lines.extend(
        [
            "",
            "Standard RSF form:",
            "- `m dV/dt = tau - sigmaN [ mu0 + a log(V/V0) + b log(theta*V0/Dc) ]`",
            f"- `m = {best_payload['parameters']['m']:.6e}`",
            f"- `mu0 = {best_payload['parameters']['mu0']:.6e}`",
            f"- `a = {best_payload['parameters']['a']:.6e}`",
            f"- `b = {best_payload['parameters']['b']:.6e}`",
            f"- `Dc = {best_payload['parameters']['Dc']:.6e}`",
            "",
            "Event-specific theta0 values from the winning run:",
        ]
    )
    for step_name, theta0 in best_payload["per_event_theta0"].items():
        report_lines.append(f"- `{step_name}`: `{theta0:.6e}`")
    report_lines.extend(
        [
            "",
            "## 5. Metrics and rollout",
            f"- Optimization success: `{best_payload['optimization']['success']}`",
            f"- Status / message: `{best_payload['optimization']['status']}` / `{best_payload['optimization']['message']}`",
            f"- `nfev = {best_payload['optimization']['nfev']}`",
            f"- Final cost: `{best_payload['optimization']['cost']:.6e}`",
            f"- Mean holdout rollout error: `{best_row['mean_holdout_error']:.6e}`",
            f"- Mean holdout stable fraction: `{best_row['mean_holdout_stable_fraction']:.6f}`",
            f"- Mean holdout onset timing error: `{best_row['mean_holdout_onset_timing_error_s']:.3f}` s",
            f"- Mean holdout peak timing error: `{best_row['mean_holdout_peak_timing_error_s']:.3f}` s",
            "- Holdout event metrics from the winning run:",
        ]
    )
    for row in best_payload["holdout_rows"]:
        report_lines.append(
            f"- `{row['step_name']}`: tau RMSE `{row['tau_rmse']:.6e}`, V RMSE `{row['V_rmse']:.6e}`, combined rollout error `{row['combined_rollout_error']:.6e}`, stable fraction `{row['stable_fraction']:.6f}`, onset timing error `{row['onset_timing_error_s']:.3f}` s, peak timing error `{row['peak_timing_error_s']:.3f}` s"
        )
    report_lines.extend(
        [
            "",
            "## 6. Why it is attractive",
            "- It is the closest exact-form RSF-looking system found in the repo.",
            "- The tau equation stays in the expected spring-loading form.",
            "- The velocity law keeps explicit `tau`, `log(V/V0)`, and `log(theta*V0/Dc)` structure.",
            "- After multistart, the fit converged cleanly and the theta term remained meaningfully active.",
            "",
            "## 7. Why it is not the final trusted model",
            "- It was not selected as the final trusted model because identifiability remained poor even after multistart.",
            f"- Near-constant sigmaN remained true: `{best_row['sigma_too_constant']}`.",
            f"- Parameter confounding remained true: `{best_row['parameter_confounding_flag']}`.",
            f"- JTJ condition number stayed very large: `{best_row['jtj_condition_number']:.6e}`.",
            f"- Parameter stability across starts was weak, especially for `m`, `mu0`, `a`, `b`, and `Dc`.",
            "- In plain English: the exact form can be fit, but the parameters do not lock down uniquely enough to treat it as a reliable recovered governing law.",
            "",
            "## 8. Comparison to the reduced fallback",
            comparison_df.to_csv(index=False),
            "",
            "Key comparison points:",
            f"- Exact RSF-looking fit mean rollout error: `{best_row['mean_holdout_error']:.6e}` versus reduced fallback rollout MSE `{proposal['velocity_models']['B_reduced_rsf']['best']['mean_rollout_mse']:.6e}`.",
            f"- Exact RSF-looking fit stable fraction: `{best_row['mean_holdout_stable_fraction']:.6f}` versus reduced fallback `{proposal['velocity_models']['B_reduced_rsf']['best']['mean_stable_fraction']:.6f}`.",
            f"- Exact RSF-looking fit peak timing error: `{best_row['mean_holdout_peak_timing_error_s']:.3f}` s versus reduced fallback `{proposal['velocity_models']['B_reduced_rsf']['best']['mean_peak_timing_error_s']:.3f}` s.",
            "- The reduced fallback stayed the best final usable equation because it was more defensible under the identifiability diagnostics, even though the exact-form fit looked more physically complete on paper.",
            "",
            "## 9. Short presentation-ready explanation",
            "The equation set came from the bounded multistart exact-RSF refinement, specifically winning start index 2 in `exact_rsf_multistart_check.py`. It is attractive because it is the closest exact-form RSF-looking system we found: spring-loading tau, explicit log-rate friction, and an active theta term. It was obtained by directly simulating the coupled tau-V-theta system and fitting global RSF parameters plus per-event theta offsets under positivity constraints. However, it was not selected as the final trusted model because sigmaN stayed nearly constant, the parameters remained confounded, and the estimates were unstable across starts even after cleaner convergence. So this is the best exact-looking fit to showcase, but not the best final usable equation. The reduced RSF fallback remains the final reportable velocity law because it is more scientifically defensible under the identifiability diagnostics.",
        ]
    )
    (RESULTS_DIR / "exact_rsf_showcase_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    print("[exact-rsf-showcase] loading exact RSF multistart artifacts", flush=True)
    ctx = load_context()
    sims = simulate_best_fit(ctx["prepared"], ctx["best_payload"])
    save_tau_rollout(ctx["prepared"]["holdout_segments"], sims["holdout"])
    save_velocity_rollout(ctx["prepared"]["holdout_segments"], sims["holdout"])
    save_phaseplot(ctx["prepared"]["holdout_segments"], sims["holdout"])
    save_theta_examples(ctx["prepared"]["train_segments"], ctx["prepared"]["holdout_segments"], sims["train"], sims["holdout"])
    write_outputs(ctx, sims)
    print("[exact-rsf-showcase] wrote showcase report, table, equations, and figures", flush=True)


if __name__ == "__main__":
    main()
