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

from src.exact_rsf import (
    json_ready,
    load_checkpoint,
    load_workflow_context,
    prepare_exact_segments,
    rollout_metrics,
    save_checkpoint,
    sparse_structure_confirmation,
    split_segments,
    fit_exact_rsf_inverse_model,
)
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
CHECKPOINT_DIR = ensure_directory(RESULTS_DIR / "exact_rsf_inverse_fit_checkpoints")


def summarize_fit(payload: dict) -> dict:
    holdout_df = pd.DataFrame(payload["holdout_rows"])
    train_df = pd.DataFrame(payload["train_rows"])
    return {
        "model": "exact_rsf_acoustic" if payload["use_acoustic"] else "exact_rsf_latent",
        "equation_tau": payload["tau_equation"],
        "equation_v": payload["velocity_equation"],
        "equation_theta": payload["theta_equation"],
        "holdout_combined_rollout_error": float(holdout_df["combined_rollout_error"].mean()),
        "holdout_tau_rmse": float(holdout_df["tau_rmse"].mean()),
        "holdout_v_rmse": float(holdout_df["V_rmse"].mean()),
        "holdout_stable_fraction": float(holdout_df["stable_fraction"].mean()),
        "holdout_peak_timing_error_s": float(holdout_df["peak_timing_error_s"].mean()),
        "holdout_onset_timing_error_s": float(holdout_df["onset_timing_error_s"].mean()),
        "train_combined_rollout_error": float(train_df["combined_rollout_error"].mean()),
        "success": bool(payload["optimization"]["success"]),
        "parameter_confounding_flag": bool(payload["identifiability"]["parameter_confounding_flag"]),
        "sigma_too_constant": bool(payload["identifiability"]["sigma_too_constant_for_mu_a_b_separation"]),
        "weak_theta_observability": bool(payload["identifiability"]["weak_theta_observability"]),
        "acoustic_used": bool(payload["use_acoustic"]),
    }


def load_reduced_fallback_summary() -> dict:
    path = RESULTS_DIR / "proposal_equation_recovery.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    reduced = payload["velocity_models"]["B_reduced_rsf"]["best"]
    return {
        "model": "reduced_rsf_fallback",
        "equation_tau": payload["tau_model"]["exact_equation"],
        "equation_v": reduced["equation"],
        "equation_theta": "not modeled explicitly",
        "holdout_combined_rollout_error": float(reduced["mean_rollout_mse"]),
        "holdout_tau_rmse": float("nan"),
        "holdout_v_rmse": float("nan"),
        "holdout_stable_fraction": float(reduced["mean_stable_fraction"]),
        "holdout_peak_timing_error_s": float(reduced["mean_peak_timing_error_s"]),
        "holdout_onset_timing_error_s": float(reduced["mean_onset_timing_error_s"]),
        "train_combined_rollout_error": float(reduced["train_mse"]),
        "success": True,
        "parameter_confounding_flag": True,
        "sigma_too_constant": True,
        "weak_theta_observability": True,
        "acoustic_used": False,
    }


def choose_final_conclusion(best_exact: dict, acoustic_summary: dict | None) -> tuple[str, str]:
    exact_recovered = (
        best_exact["optimization"]["success"]
        and not best_exact["identifiability"]["parameter_confounding_flag"]
        and not best_exact["identifiability"]["weak_theta_observability"]
        and not best_exact["identifiability"]["sigma_too_constant_for_mu_a_b_separation"]
        and np.mean([row["stable_fraction"] for row in best_exact["holdout_rows"]]) >= 0.5
    )
    if exact_recovered:
        return "A", "Exact proposal equations (1)-(3) recovered credibly"
    partial = (
        best_exact["optimization"]["success"]
        and not best_exact["identifiability"]["sigma_too_constant_for_mu_a_b_separation"]
        and not best_exact["identifiability"]["parameter_confounding_flag"]
    )
    if partial:
        return "B", "Equation (1) recovered and exact equation (2) partially recovered, but still weakly identifiable"
    return "C", "Equation (1) recovered; exact equation (2) implemented and tested directly, but still not identifiable from current Utah FORGE data"


def blocker_text(ident: dict) -> str:
    reasons = []
    if ident["sigma_too_constant_for_mu_a_b_separation"]:
        reasons.append("near-constant sigmaN")
    if ident["weak_theta_observability"]:
        reasons.append("weak theta observability")
    if ident["parameter_confounding_flag"]:
        reasons.append("parameter confounding")
    if not reasons:
        reasons.append("insufficient event diversity")
    return ", ".join(reasons)


def build_diagnostic_figure(best_payload: dict, train_segments, holdout_segments) -> None:
    train_by_name = {segment.step_name: segment for segment in train_segments}
    holdout_by_name = {segment.step_name: segment for segment in holdout_segments}
    sample_train = train_segments[0]
    sample_holdout = holdout_segments[0]
    params = best_payload["parameters"]
    acoustic_z = best_payload["acoustic_zscores"]
    theta_offsets = best_payload["theta_offsets_train"]

    from src.exact_rsf import simulate_exact_rsf_segment

    train_sim = simulate_exact_rsf_segment(
        sample_train,
        params,
        delta_log_theta0=theta_offsets.get(sample_train.step_name, 0.0),
        acoustic_z=acoustic_z.get(sample_train.step_name, 0.0),
    )
    holdout_sim = simulate_exact_rsf_segment(
        sample_holdout,
        params,
        delta_log_theta0=0.0,
        acoustic_z=acoustic_z.get(sample_holdout.step_name, 0.0),
    )

    corr = np.array(best_payload["identifiability"]["parameter_correlation_matrix"], dtype=float)
    labels = best_payload["identifiability"]["parameter_names"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(sample_holdout.time - sample_holdout.time[0], sample_holdout.tau, label="observed tau", linewidth=1.0)
    axes[0, 0].plot(sample_holdout.time - sample_holdout.time[0], holdout_sim["tau"], label="simulated tau", linewidth=1.0)
    axes[0, 0].set_title(f"Holdout tau: {sample_holdout.step_name}")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(sample_holdout.time - sample_holdout.time[0], sample_holdout.V, label="observed V", linewidth=1.0)
    axes[0, 1].plot(sample_holdout.time - sample_holdout.time[0], holdout_sim["V"], label="simulated V", linewidth=1.0)
    axes[0, 1].set_title(f"Holdout V: {sample_holdout.step_name}")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(sample_train.time - sample_train.time[0], train_sim["theta"], label="fitted latent theta", linewidth=1.0)
    axes[1, 0].plot(sample_train.time - sample_train.time[0], sample_train.theta_proxy, label="RSFit theta proxy", linewidth=1.0, alpha=0.8)
    axes[1, 0].set_title(f"Theta example: {sample_train.step_name}")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="best")

    image = axes[1, 1].imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_yticks(range(len(labels)))
    axes[1, 1].set_yticklabels(labels, fontsize=8)
    axes[1, 1].set_title("Parameter correlation summary")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exact_rsf_inverse_fit_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    structure_payload: dict,
    exact_payload: dict,
    acoustic_payload: dict | None,
    train_names: list[str],
    holdout_names: list[str],
    acoustic_name: str | None,
) -> None:
    reduced_summary = load_reduced_fallback_summary()
    exact_summary = summarize_fit(exact_payload)
    acoustic_summary = summarize_fit(acoustic_payload) if acoustic_payload is not None else None
    conclusion_code, conclusion_text = choose_final_conclusion(exact_payload, acoustic_summary)
    ident = exact_payload["identifiability"]
    blocker = blocker_text(ident)

    comparison_rows = [reduced_summary, exact_summary]
    if acoustic_summary is not None:
        comparison_rows.append(acoustic_summary)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_DIR / "exact_rsf_model_comparison.csv", index=False)

    output_json = {
        "structure_confirmation": structure_payload,
        "exact_fit": exact_payload,
        "acoustic_fit": acoustic_payload,
        "reduced_fallback_summary": reduced_summary,
        "train_steps": train_names,
        "holdout_steps": holdout_names,
        "acoustic_feature_name": acoustic_name,
        "conclusion_code": conclusion_code,
        "conclusion_text": conclusion_text,
        "blocker": blocker,
    }
    (RESULTS_DIR / "exact_rsf_inverse_fit.json").write_text(json.dumps(json_ready(output_json), indent=2), encoding="utf-8")

    equations_text = "\n".join(
        [
            exact_payload["tau_equation"],
            exact_payload["velocity_equation"],
            exact_payload["theta_equation"],
            acoustic_payload["velocity_equation"] if acoustic_payload is not None else "Acoustic branch not run",
        ]
    )
    (RESULTS_DIR / "exact_rsf_inverse_fit_equations.txt").write_text(equations_text + "\n", encoding="utf-8")

    ident_lines = [
        "# Exact RSF Identifiability Summary",
        "",
        f"- Conclusion: `{conclusion_text}`",
        f"- Main blocker(s): `{blocker}`",
        f"- JTJ condition number: `{ident['jtj_condition_number']:.6e}`",
        f"- JTJ rank: `{ident['jtj_rank']}`",
        f"- SigmaN coefficient of variation on training data: `{ident['sigma_cv']:.6e}`",
        f"- SigmaN too constant for separating mu0/a/b: `{ident['sigma_too_constant_for_mu_a_b_separation']}`",
        f"- Weak theta observability: `{ident['weak_theta_observability']}`",
        f"- Parameter confounding flag: `{ident['parameter_confounding_flag']}`",
    ]
    (RESULTS_DIR / "exact_rsf_identifiability_summary.md").write_text("\n".join(ident_lines) + "\n", encoding="utf-8")

    holdout_df = pd.DataFrame(exact_payload["holdout_rows"])
    acoustic_lines = (
        [
            f"- Acoustic branch used: `{True}`",
            f"- Acoustic feature: `{acoustic_name}`",
            f"- Acoustic holdout error: `{summarize_fit(acoustic_payload)['holdout_combined_rollout_error']:.6e}`",
        ]
        if acoustic_payload is not None
        else ["- Acoustic branch used: `False`"]
    )
    report_lines = [
        "# Exact RSF Inverse Fit Report",
        "",
        "## Proposal-faithful workflow",
        "- Stage A: sparse structure confirmation using physics-guided candidate libraries",
        "- Stage B: constrained inverse fitting of the coupled RSF system with latent theta",
        "",
        "## Train / holdout split",
        f"- Train steps: `{', '.join(train_names)}`",
        f"- Holdout steps: `{', '.join(holdout_names)}`",
        "",
        "## SINDy-confirmed structure",
        f"- Equation (1) spring-loading confirmed: `{structure_payload['tau_spring_loading_confirmed']}`",
        f"- Velocity tau term confirmed: `{structure_payload['velocity_tau_confirmed']}`",
        f"- Velocity log(V) term confirmed: `{structure_payload['velocity_logv_confirmed']}`",
        f"- Hidden-state evidence from theta proxy: `{structure_payload['velocity_hidden_state_evidence']}`",
        "",
        "## Exact fitted equations",
        f"- `{exact_payload['tau_equation']}`",
        f"- `{exact_payload['velocity_equation']}`",
        f"- `{exact_payload['theta_equation']}`",
        "",
        "## Parameter estimates",
        f"- `{json.dumps(exact_payload['parameters'])}`",
        f"- Event-specific theta0 values: `{json.dumps(exact_payload['per_event_theta0'])}`",
        f"- Event-specific theta offsets: `{json.dumps(exact_payload['theta_offsets_train'])}`",
        "",
        "## Constraints used",
        "- `k > 0`, `m > 0`, `a > 0`, `b > 0`, `Dc > 0`",
        "- `theta(t) > 0` enforced through positive initialization and clipped forward integration",
        "- `V(t) > 0` enforced through positive forward integration",
        "",
        "## Validation",
        f"- Mean holdout tau RMSE: `{holdout_df['tau_rmse'].mean():.6e}`",
        f"- Mean holdout V RMSE: `{holdout_df['V_rmse'].mean():.6e}`",
        f"- Mean holdout combined rollout error: `{holdout_df['combined_rollout_error'].mean():.6e}`",
        f"- Mean holdout stable fraction: `{holdout_df['stable_fraction'].mean():.3f}`",
        f"- Mean holdout onset timing error: `{holdout_df['onset_timing_error_s'].mean():.3f}` s",
        f"- Mean holdout peak timing error: `{holdout_df['peak_timing_error_s'].mean():.3f}` s",
        "",
        "## Identifiability findings",
        f"- SigmaN too constant for clean mu0/a/b separation: `{ident['sigma_too_constant_for_mu_a_b_separation']}`",
        f"- Weak theta observability: `{ident['weak_theta_observability']}`",
        f"- Parameter confounding flag: `{ident['parameter_confounding_flag']}`",
        f"- Main blocker(s): `{blocker}`",
        "",
        "## Comparison to current reduced RSF fallback",
        f"- Reduced fallback equation: `{reduced_summary['equation_v']}`",
        f"- Reduced fallback stable fraction: `{reduced_summary['holdout_stable_fraction']:.3f}`",
        f"- Exact RSF stable fraction: `{exact_summary['holdout_stable_fraction']:.3f}`",
        f"- Reduced fallback peak timing error: `{reduced_summary['holdout_peak_timing_error_s']:.3f}` s",
        f"- Exact RSF peak timing error: `{exact_summary['holdout_peak_timing_error_s']:.3f}` s",
        "",
        "## Acoustic branch",
        *acoustic_lines,
        "",
        "## Final conclusion",
        f"- `{conclusion_text}`",
        f"- Blocker(s): `{blocker}`",
    ]
    (RESULTS_DIR / "exact_rsf_inverse_fit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    prepared = load_checkpoint(CHECKPOINT_DIR, "prepared_exact_segments")
    if prepared is None:
        inventory_df, segments, steps, rsfit_globals = load_workflow_context()
        train_segments_raw, holdout_segments_raw, train_names, holdout_names = split_segments(inventory_df, segments)
        train_segments, holdout_segments, acoustic_name = prepare_exact_segments(train_segments_raw, holdout_segments_raw, steps, rsfit_globals)
        prepared = {
            "train_segments": train_segments,
            "holdout_segments": holdout_segments,
            "train_names": train_names,
            "holdout_names": holdout_names,
            "acoustic_name": acoustic_name,
        }
        save_checkpoint(
            CHECKPOINT_DIR,
            "prepared_exact_segments",
            prepared,
            {"train_steps": train_names, "holdout_steps": holdout_names, "acoustic_name": acoustic_name},
        )
    else:
        print("[resume] loaded prepared_exact_segments checkpoint", flush=True)

    train_segments = prepared["train_segments"]
    holdout_segments = prepared["holdout_segments"]
    train_names = prepared["train_names"]
    holdout_names = prepared["holdout_names"]
    acoustic_name = prepared["acoustic_name"]

    structure_payload = load_checkpoint(CHECKPOINT_DIR, "structure_confirmation")
    if structure_payload is None:
        structure_payload = sparse_structure_confirmation(train_segments, acoustic_name)
        save_checkpoint(CHECKPOINT_DIR, "structure_confirmation", structure_payload, structure_payload)

    exact_payload = fit_exact_rsf_inverse_model(
        train_segments,
        holdout_segments,
        use_acoustic=False,
        checkpoint_dir=CHECKPOINT_DIR,
        stage_name="exact_fit_base",
    )
    acoustic_payload = None
    if acoustic_name is not None and any(np.isfinite(segment.acoustic_event_value) for segment in train_segments):
        acoustic_payload = fit_exact_rsf_inverse_model(
            train_segments,
            holdout_segments,
            use_acoustic=True,
            checkpoint_dir=CHECKPOINT_DIR,
            stage_name="exact_fit_acoustic",
        )

    build_diagnostic_figure(exact_payload, train_segments, holdout_segments)
    write_outputs(structure_payload, exact_payload, acoustic_payload, train_names, holdout_names, acoustic_name)
    print("Exact RSF inverse-fit workflow complete.", flush=True)
    print(
        json.dumps(
            {
                "tau_equation": exact_payload["tau_equation"],
                "velocity_equation": exact_payload["velocity_equation"],
                "theta_equation": exact_payload["theta_equation"],
                "acoustic_branch_used": acoustic_payload is not None,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
