from __future__ import annotations

import json
import shutil
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

from scripts import utah_forge_proposal_equation_recovery as proposal_recovery
from scripts import utah_forge_showcase_fit_visuals as showcase
from scripts import utah_forge_reviewer_ablation as reviewer_ablation


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
FINAL_DIR = RESULTS_DIR / "Final"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    if pd.isna(value):
        return None
    return value


def load_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def frame_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: ""
            if pd.isna(value)
            else (f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value))
        )
    headers = [str(col) for col in display.columns]
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def load_old_truth_segments() -> dict[str, pd.DataFrame]:
    segments, steps, rsfit_globals = reviewer_ablation.load_segments()
    truth: dict[str, pd.DataFrame] = {}
    for step_name in reviewer_ablation.HOLDOUT_STEPS:
        prepared_df = reviewer_ablation.prepare_step_segment(
            segments[step_name],
            reviewer_ablation.MODEL_B_CONFIG["smoothing"],
            reviewer_ablation.MODEL_B_CONFIG["memory_window"],
            reviewer_ablation.MODEL_B_CONFIG["ema_span"],
        )
        prepared_df = prepared_df.copy()
        prepared_df.insert(0, "step_name", step_name)
        truth[step_name] = prepared_df
    return truth


def plot_old_model_rollout(rows: list[dict], truth_map: dict[str, pd.DataFrame], title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.5 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = np.array([axes])
    for row_axes, row in zip(axes, rows):
        step_name = row["step_name"]
        truth = truth_map[step_name]
        time = truth["time"].to_numpy(dtype=float) - float(truth["time"].iloc[0])
        ax_tau, ax_v = row_axes
        ax_tau.plot(time, truth["tau"].to_numpy(dtype=float), label="Observed tau", linewidth=1.2)
        ax_tau.plot(time, np.asarray(row["tau_prediction"], dtype=float), label="Predicted tau", linewidth=1.1, linestyle="--")
        ax_tau.set_title(f"{step_name} tau")
        ax_tau.set_xlabel("Time since step start [s]")
        ax_tau.set_ylabel("tau")
        ax_tau.grid(True, alpha=0.3)
        ax_v.plot(time, truth["V"].to_numpy(dtype=float), label="Observed V", linewidth=1.2)
        ax_v.plot(time, np.asarray(row["V_prediction"], dtype=float), label="Predicted V", linewidth=1.1, linestyle="--")
        ax_v.set_title(f"{step_name} V")
        ax_v.set_xlabel("Time since step start [s]")
        ax_v.set_ylabel("V")
        ax_v.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_theta_consistency(theta_payload: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    rows = pd.DataFrame(theta_payload["table_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(rows["variant"], rows["holdout_rmse"], color=["#4c78a8", "#f58518"])
    axes[0].set_title("Theta consistency holdout RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(rows["variant"], rows["Dc_hat"], color=["#4c78a8", "#f58518"])
    axes[1].set_title("Implied Dc from c1")
    axes[1].set_ylabel("Dc_hat")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bc_tau_fix(table: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = table["family"] + "_" + table["variant"]
    axes[0].bar(labels, table["tau_n_active_terms"], color="#59a14f")
    axes[0].set_title("Tau equation active terms")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, table["holdout_rollout_combined_rmse"], color="#e15759")
    axes[1].set_title("Holdout combined rollout RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_isolation(table: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(table["variant"], table["holdout_derivative_rmse"], color="#4c78a8")
    axes[0].set_title("Holdout derivative RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(table["variant"], table["holdout_rollout_combined_rmse"], color="#f58518")
    axes[1].set_title("Holdout rollout combined RMSE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_subfolder(model_dir: Path, summary: dict, metrics: pd.DataFrame) -> None:
    ensure_dir(model_dir)
    (model_dir / "model_summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    metrics.to_csv(model_dir / "model_metrics.csv", index=False)
    lines = [
        f"# {summary['display_name']}",
        "",
        "## Model Card",
        f"- Category: `{summary['category']}`",
        f"- Workflow/source: `{summary['script_source']}`",
        f"- What is special: {summary['what_is_special']}",
        f"- Theta mode: `{summary['theta_mode']}`",
        "",
        "## Equations",
    ]
    for key in ("equation_1", "equation_2", "equation_3"):
        if summary.get(key):
            lines.append(f"- `{summary[key]}`")
    lines.extend(
        [
            "",
            "## Data / Split Summary",
            f"- Dataset: `{summary['dataset_name']}`",
            f"- Usable steps: `{summary['usable_steps']}`",
            f"- Train steps: `{summary['train_steps']}`",
            f"- Holdout steps: `{summary['holdout_steps']}`",
            f"- Exclusions: `{summary['exclusions']}`",
            "",
            "## Metrics",
            frame_to_markdown(metrics),
            "",
            "## Interpretation",
            f"- What worked: {summary['what_worked']}",
            f"- What failed: {summary['what_failed']}",
            f"- What it taught us: {summary['what_it_taught_us']}",
            "",
            f"- Final judgment: `{summary['final_judgment']}`",
        ]
    )
    (model_dir / "model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(FINAL_DIR)

    proposal = load_json("proposal_equation_recovery.json")
    exact_showcase = load_json("exact_rsf_showcase_report.json")
    exact_multistart = load_json("exact_rsf_multistart_summary.json")
    theta_consistency = load_json("theta_equation_consistency.json")
    bc_tau_fix = load_json("model_BC_tau_fix_comparison.json")
    c_velocity_isolation = load_json("model_C_velocity_isolation_comparison.json")
    multistep_summary = load_json("multistep_rollout_summary.json")
    performance = load_json("project_performance_assessment.json")
    ablation = load_csv("p5838_ablation_table.csv")
    performance_master = load_csv("project_performance_master_table.csv")
    bc_tau_fix_table = load_csv("model_BC_tau_fix_table.csv")
    c_velocity_table = load_csv("model_C_velocity_isolation_table.csv")
    multistep_tau = load_csv("multistep_tau_table.csv")
    multistep_velocity = load_csv("multistep_velocity_table.csv")
    multistep_exact = load_csv("multistep_exact_rsf_table.csv")

    truth_map = load_old_truth_segments()
    plot_old_model_rollout(bc_tau_fix["original_B"]["holdout_rollout"]["rows"], truth_map, "Model B holdout rollout", FINAL_DIR / "Model_B" / "Model_B_rollout.png")
    plot_old_model_rollout(bc_tau_fix["original_C"]["holdout_rollout"]["rows"], truth_map, "Model C holdout rollout", FINAL_DIR / "Model_C" / "Model_C_rollout.png")
    plot_theta_consistency(theta_consistency, FINAL_DIR / "Theta_Consistency" / "Theta_consistency_plot.png")
    plot_bc_tau_fix(bc_tau_fix_table, FINAL_DIR / "BC_Tau_Fix" / "BC_tau_fix_summary.png")
    plot_velocity_isolation(c_velocity_table, FINAL_DIR / "C_Velocity_Isolation" / "C_velocity_isolation_summary.png")

    copy_if_exists(RESULTS_DIR / "baseline_rollout.png", FINAL_DIR / "Model_A" / "Model_A_rollout.png")
    copy_if_exists(RESULTS_DIR / "showcase_tau_fit.png", FINAL_DIR / "Proposal_Tau" / "Proposal_Tau_rollout.png")
    copy_if_exists(RESULTS_DIR / "showcase_velocity_fit.png", FINAL_DIR / "Reduced_RSF" / "Reduced_RSF_rollout.png")
    copy_if_exists(RESULTS_DIR / "showcase_exact_rsf_fit.png", FINAL_DIR / "Exact_RSF" / "Exact_RSF_rollout.png")
    copy_if_exists(RESULTS_DIR / "showcase_phaseplot.png", FINAL_DIR / "Exact_RSF" / "Exact_RSF_phaseplot.png")
    copy_if_exists(RESULTS_DIR / "showcase_derivative_scatter.png", FINAL_DIR / "Proposal_Tau" / "Proposal_derivative_scatter.png")
    copy_if_exists(RESULTS_DIR / "multistep_tau_rollout_gallery.png", FINAL_DIR / "Multistep_Assessment" / "multistep_tau_rollout_gallery.png")
    copy_if_exists(RESULTS_DIR / "multistep_velocity_rollout_gallery.png", FINAL_DIR / "Multistep_Assessment" / "multistep_velocity_rollout_gallery.png")
    copy_if_exists(RESULTS_DIR / "multistep_exact_rsf_gallery.png", FINAL_DIR / "Multistep_Assessment" / "multistep_exact_rsf_gallery.png")
    copy_if_exists(RESULTS_DIR / "multistep_phaseplots.png", FINAL_DIR / "Multistep_Assessment" / "multistep_phaseplots.png")
    copy_if_exists(RESULTS_DIR / "overall_performance_comparison.png", FINAL_DIR / "Multistep_Assessment" / "overall_performance_comparison.png")
    copy_if_exists(RESULTS_DIR / "train_vs_holdout_summary.png", FINAL_DIR / "Multistep_Assessment" / "train_vs_holdout_summary.png")
    copy_if_exists(RESULTS_DIR / "step_difficulty_heatmap.png", FINAL_DIR / "Multistep_Assessment" / "step_difficulty_heatmap.png")
    copy_if_exists(RESULTS_DIR / "exact_vs_reduced_summary.png", FINAL_DIR / "Multistep_Assessment" / "exact_vs_reduced_summary.png")

    model_c_subset = bc_tau_fix["model_c_subset"]
    exact_params = exact_multistart["best_run"]["parameters"]

    model_specs = [
        {
            "folder": "Model_A",
            "display_name": "Model A",
            "category": "observed-only baseline",
            "script_source": "scripts/utah_forge_reviewer_ablation.py",
            "equation_1": ablation.loc[ablation["model"] == "A", "tau_equation"].iloc[0],
            "equation_2": ablation.loc[ablation["model"] == "A", "V_equation"].iloc[0],
            "equation_3": "",
            "uses_theta": False,
            "theta_mode": "none",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE p5838 RSFit-aligned steps",
            "usable_steps": ", ".join(reviewer_ablation.TRAIN_STEPS + reviewer_ablation.HOLDOUT_STEPS),
            "train_steps": ", ".join(reviewer_ablation.TRAIN_STEPS),
            "holdout_steps": ", ".join(reviewer_ablation.HOLDOUT_STEPS),
            "exclusions": "none",
            "what_is_special": "Observed-only baseline using only measured state variables with no memory surrogate and no theta proxy.",
            "what_worked": "It kept spring-loading and log-rate structure and performed very well on holdout step2.",
            "what_failed": "It collapsed almost immediately on holdout step7, so it was not balanced across held-out regimes.",
            "what_it_taught_us": "Observed variables alone can capture some events, but not robustly across the Utah FORGE holdout pair.",
            "final_judgment": "old baseline / observed-only reference",
            "metrics": pd.DataFrame([ablation.loc[ablation["model"] == "A"].iloc[0].to_dict()]),
        },
        {
            "folder": "Model_B",
            "display_name": "Model B",
            "category": "memory-augmented surrogate-memory model",
            "script_source": "scripts/utah_forge_reviewer_ablation.py",
            "equation_1": bc_tau_fix["original_B"]["tau_equation"],
            "equation_2": bc_tau_fix["original_B"]["V_equation"],
            "equation_3": "",
            "uses_theta": False,
            "theta_mode": "none",
            "uses_memory_surrogates": True,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE p5838 RSFit-aligned steps",
            "usable_steps": ", ".join(reviewer_ablation.TRAIN_STEPS + reviewer_ablation.HOLDOUT_STEPS),
            "train_steps": ", ".join(reviewer_ablation.TRAIN_STEPS),
            "holdout_steps": ", ".join(reviewer_ablation.HOLDOUT_STEPS),
            "exclusions": "none",
            "what_is_special": "Memory-augmented surrogate model using tau_avg, tau_ema, and cumulative slip S.",
            "what_worked": "It was the most balanced old holdout model by worst-case divergence and kept both tau and logV structure active.",
            "what_failed": "Its equations stayed dense and surrogate-heavy rather than compact or proposal-faithful.",
            "what_it_taught_us": "Memory surrogates are practical for rollout balance, even when they are not the cleanest physics story.",
            "final_judgment": "old best-balanced surrogate model",
            "metrics": pd.DataFrame([bc_tau_fix["original_B"]["holdout_derivative"]["tau"] | {"holdout_rollout_combined_rmse": bc_tau_fix["original_B"]["holdout_rollout"]["mean_combined_rmse"], "mean_divergence_s": bc_tau_fix["original_B"]["holdout_rollout"]["mean_divergence_s"]}]),
        },
        {
            "folder": "Model_C",
            "display_name": "Model C",
            "category": "theta-informed RSFit-theta model",
            "script_source": "scripts/utah_forge_reviewer_ablation.py",
            "equation_1": bc_tau_fix["original_C"]["tau_equation"],
            "equation_2": bc_tau_fix["original_C"]["V_equation"],
            "equation_3": "",
            "uses_theta": True,
            "theta_mode": "supplied",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE p5838 RSFit-aligned steps",
            "usable_steps": ", ".join(model_c_subset["train_steps"] + model_c_subset["holdout_steps"]),
            "train_steps": ", ".join(model_c_subset["train_steps"]),
            "holdout_steps": ", ".join(model_c_subset["holdout_steps"]),
            "exclusions": "steps with invalid or clipped reconstructed theta are skipped",
            "what_is_special": "Theta-informed model using externally reconstructed RSFit theta in the candidate library.",
            "what_worked": "It made state dependence explicit and achieved the lowest AIC/BIC among the old A/B/C family.",
            "what_failed": "It did not outperform Model B on holdout rollout and its theta coefficient did not line up cleanly with RSF scale expectations.",
            "what_it_taught_us": "Supplying theta can make the algebra look more RSF-like, but it does not guarantee better generalization or trustworthy coefficients.",
            "final_judgment": "old theta-informed model; instructive but not final",
            "metrics": pd.DataFrame([bc_tau_fix["original_C"]["holdout_derivative"]["tau"] | {"holdout_rollout_combined_rmse": bc_tau_fix["original_C"]["holdout_rollout"]["mean_combined_rmse"], "mean_divergence_s": bc_tau_fix["original_C"]["holdout_rollout"]["mean_divergence_s"]}]),
        },
        {
            "folder": "Proposal_Tau",
            "display_name": "Proposal Tau Equation",
            "category": "isolated compact physical tau law",
            "script_source": "scripts/utah_forge_proposal_equation_recovery.py",
            "equation_1": proposal["tau_model"]["exact_equation"],
            "equation_2": proposal["tau_model"]["one_term_equation"],
            "equation_3": "",
            "uses_theta": False,
            "theta_mode": "none",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE p5838 RSFit-aligned steps",
            "usable_steps": ", ".join(multistep_summary["usable_steps"]),
            "train_steps": ", ".join(multistep_summary["train_steps"]),
            "holdout_steps": ", ".join(multistep_summary["holdout_steps"]),
            "exclusions": "semi-observed rollout uses observed V(t) and V_drive(t)",
            "what_is_special": "Equation (1) was isolated and fit with a tiny physical library [1, V, V_drive_minus_V].",
            "what_worked": "It recovered a compact spring-loading law and stayed robust under bounded split checks.",
            "what_failed": "Its multistep holdout tau rollout was much worse than its train-step behavior, so the original holdout pair was not representative.",
            "what_it_taught_us": "Tau isolation is a transferable identification improvement and the strongest methodological success in the project.",
            "final_judgment": "strongest recovered equation",
            "metrics": pd.DataFrame([performance_master.loc[performance_master["result_label"] == "Best tau equation"].iloc[0].to_dict()]),
        },
        {
            "folder": "Reduced_RSF",
            "display_name": "Reduced RSF Fallback",
            "category": "best final usable velocity law",
            "script_source": "scripts/utah_forge_proposal_equation_recovery.py",
            "equation_1": "",
            "equation_2": proposal["final_velocity_model"]["equation"],
            "equation_3": "",
            "uses_theta": False,
            "theta_mode": "none",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": True,
            "dataset_name": "Utah FORGE p5838 RSFit-aligned steps",
            "usable_steps": ", ".join(multistep_summary["usable_steps"]),
            "train_steps": ", ".join(multistep_summary["train_steps"]),
            "holdout_steps": ", ".join(multistep_summary["holdout_steps"]),
            "exclusions": "theta removed after identifiability failure",
            "what_is_special": "Reduced RSF fallback keeps the negative sigmaN*log(V/V0) structure while dropping untrustworthy theta dependence.",
            "what_worked": "It became the best final usable velocity model and performed best on the original holdout pair.",
            "what_failed": "Its multistep train-to-holdout pattern was uneven, so it should be presented as usable rather than universally strong.",
            "what_it_taught_us": "The most defensible velocity law is reduced and physics-informed, not the full exact theta-bearing form.",
            "final_judgment": "best final usable model",
            "metrics": pd.DataFrame([performance_master.loc[performance_master["result_label"] == "Reduced RSF fallback velocity"].iloc[0].to_dict()]),
        },
        {
            "folder": "Exact_RSF",
            "display_name": "Exact RSF Multistart Fit",
            "category": "closest exact RSF-looking fit",
            "script_source": "scripts/utah_forge_exact_rsf_multistart_check.py",
            "equation_1": f"dtau/dt = {exact_params['k']:.6e}*(V_drive - V)",
            "equation_2": f"dV/dt = (1/{exact_params['m']:.6e})*[tau - sigmaN*({exact_params['mu0']:.6e} + {exact_params['a']:.6e}*log(V/V0) + {exact_params['b']:.6e}*log(theta*V0/{exact_params['Dc']:.6e}))]",
            "equation_3": f"dtheta/dt = 1 - V*theta/{exact_params['Dc']:.6e}",
            "uses_theta": True,
            "theta_mode": "latent",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": True,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE p5838 RSFit-aligned steps",
            "usable_steps": ", ".join(multistep_summary["usable_steps"]),
            "train_steps": ", ".join(exact_showcase["data_subset"]["train_steps"]),
            "holdout_steps": ", ".join(exact_showcase["data_subset"]["holdout_steps"]),
            "exclusions": "none for rollout; identifiability remains poor",
            "what_is_special": "Exact latent-state inverse fit of the full RSF form with constrained multistart optimization.",
            "what_worked": "It preserved the exact proposal form and improved convergence and timing relative to the earlier exact attempt.",
            "what_failed": "It remained non-identifiable, parameter-unstable across starts, and catastrophically weak on multistep holdout rollout.",
            "what_it_taught_us": "Cleaner convergence is not the same as scientific trustworthiness when sigmaN is nearly constant and parameters are confounded.",
            "final_judgment": "closest exact fit but non-identifiable",
            "metrics": pd.DataFrame([performance_master.loc[performance_master["result_label"] == "Closest exact RSF-looking fit"].iloc[0].to_dict()]),
        },
        {
            "folder": "Theta_Consistency",
            "display_name": "Theta Consistency Check",
            "category": "conditional consistency check",
            "script_source": "scripts/utah_forge_theta_equation_consistency.py",
            "equation_1": "",
            "equation_2": "",
            "equation_3": theta_consistency["tiny_library"]["equation"],
            "uses_theta": True,
            "theta_mode": "consistency-check",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE theta-valid subset",
            "usable_steps": ", ".join(theta_consistency["usable_subset"]["train_steps"] + theta_consistency["usable_subset"]["holdout_steps"]),
            "train_steps": ", ".join(theta_consistency["usable_subset"]["train_steps"]),
            "holdout_steps": ", ".join(theta_consistency["usable_subset"]["holdout_steps"]),
            "exclusions": "conditional on supplied theta only; not raw hidden-state discovery",
            "what_is_special": "Conditional test of whether externally reconstructed theta is self-consistent with the third RSF state equation.",
            "what_worked": "The fitted Vtheta coefficient had the expected negative sign.",
            "what_failed": "The intercept was nowhere near 1 and holdout R2 stayed essentially zero, so the consistency check remained weak.",
            "what_it_taught_us": "Reconstructed theta carries suggestive structure, but not a clean standalone validation of equation (3).",
            "final_judgment": "weak conditional consistency check",
            "metrics": pd.DataFrame(theta_consistency["table_rows"]),
        },
        {
            "folder": "BC_Tau_Fix",
            "display_name": "Model B/C Tau-Fix Comparison",
            "category": "ablation / refinement",
            "script_source": "scripts/utah_forge_model_bc_tau_fix_comparison.py",
            "equation_1": bc_tau_fix["tau_fixed_B"]["tau_equation"],
            "equation_2": bc_tau_fix["tau_fixed_C"]["tau_equation"],
            "equation_3": "",
            "uses_theta": True,
            "theta_mode": "supplied",
            "uses_memory_surrogates": True,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": False,
            "dataset_name": "Utah FORGE controlled comparison on original B/C families",
            "usable_steps": ", ".join(multistep_summary["usable_steps"]),
            "train_steps": "Model B: original split; Model C: theta-valid subset",
            "holdout_steps": ", ".join(multistep_summary["holdout_steps"]),
            "exclusions": "Model C uses theta-valid subset",
            "what_is_special": "Controlled test showing that isolating tau with a tiny physical library cleans equation (1) in both Model B and Model C.",
            "what_worked": "It removed nonphysical tau-side clutter and improved holdout rollout behavior for both model families.",
            "what_failed": "It did not solve the velocity-side theta limitation in Model C.",
            "what_it_taught_us": "Tau isolation is a transferable methodological cleanup, not a one-off trick.",
            "final_judgment": "supporting ablation result",
            "metrics": bc_tau_fix_table,
        },
        {
            "folder": "C_Velocity_Isolation",
            "display_name": "Model C Velocity Isolation",
            "category": "ablation / refinement",
            "script_source": "scripts/utah_forge_model_c_velocity_isolation.py",
            "equation_1": "",
            "equation_2": c_velocity_isolation["isolated_timevarying"]["equation"],
            "equation_3": "",
            "uses_theta": True,
            "theta_mode": "supplied",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": False,
            "uses_reduced_rsf_form": True,
            "dataset_name": "Utah FORGE conditional recovery experiment",
            "usable_steps": ", ".join(c_velocity_isolation["usable_subset"]["train_steps"] + c_velocity_isolation["usable_subset"]["holdout_steps"]),
            "train_steps": ", ".join(c_velocity_isolation["usable_subset"]["train_steps"]),
            "holdout_steps": ", ".join(c_velocity_isolation["usable_subset"]["holdout_steps"]),
            "exclusions": "theta treated as supplied dataset feature; constant-theta version is ablation only",
            "what_is_special": "Conditional recovery experiment asking whether supplied RSFit theta makes equation (2) cleaner when velocity is isolated.",
            "what_worked": "It produced cleaner algebraic velocity equations and confirmed that theta is genuinely time-varying.",
            "what_failed": "Tau collapsed out of the isolated fit and rollout got worse, so the cleaner equation was not actually a better recovered physical law.",
            "what_it_taught_us": "Isolating equation (2) does not help the way isolating equation (1) helped.",
            "final_judgment": "supporting ablation result",
            "metrics": c_velocity_table,
        },
        {
            "folder": "Multistep_Assessment",
            "display_name": "Multistep Rollout Assessment",
            "category": "evaluation package",
            "script_source": "scripts/utah_forge_multistep_rollout_summary.py and scripts/utah_forge_project_performance_assessment.py",
            "equation_1": proposal["tau_model"]["exact_equation"],
            "equation_2": proposal["final_velocity_model"]["equation"],
            "equation_3": f"dtheta/dt = 4.084008e-05 - 5.703972e-03*Vtheta",
            "uses_theta": True,
            "theta_mode": "mixed: none / supplied / latent / consistency-check",
            "uses_memory_surrogates": False,
            "uses_exact_rsf_form": True,
            "uses_reduced_rsf_form": True,
            "dataset_name": "All usable RSFit-aligned p5838 steps",
            "usable_steps": ", ".join(multistep_summary["usable_steps"]),
            "train_steps": ", ".join(multistep_summary["train_steps"]),
            "holdout_steps": ", ".join(multistep_summary["holdout_steps"]),
            "exclusions": "step4/5/10 are theta-low-quality for some theta-dependent analyses",
            "what_is_special": "Project-wide evaluation expansion across all eight usable p5838 steps, not just the original two holdouts.",
            "what_worked": "It exposed which results are actually robust and which ones only looked strong on the original holdout pair.",
            "what_failed": "It revealed that the original holdouts were not representative in the same way for tau, reduced velocity, and exact RSF.",
            "what_it_taught_us": "Generalization conclusions change noticeably when evaluation expands beyond step2 and step7.",
            "final_judgment": "project-level assessment",
            "metrics": pd.DataFrame(
                [
                    {"table": "multistep_tau_table.csv", "n_rows": len(multistep_tau)},
                    {"table": "multistep_velocity_table.csv", "n_rows": len(multistep_velocity)},
                    {"table": "multistep_exact_rsf_table.csv", "n_rows": len(multistep_exact)},
                    {"table": "project_performance_master_table.csv", "n_rows": len(performance_master)},
                ]
            ),
        },
    ]

    master_rows = []
    index_rows = []
    for spec in model_specs:
        folder = FINAL_DIR / spec["folder"]
        summary = {k: v for k, v in spec.items() if k != "metrics"}
        metrics = spec["metrics"]
        write_subfolder(folder, summary, metrics)
        master_rows.append(
            {
                "display_name": spec["display_name"],
                "category": spec["category"],
                "script_source": spec["script_source"],
                "equation_1": spec["equation_1"],
                "equation_2": spec["equation_2"],
                "equation_3": spec["equation_3"],
                "uses_theta": spec["uses_theta"],
                "theta_mode": spec["theta_mode"],
                "uses_memory_surrogates": spec["uses_memory_surrogates"],
                "uses_exact_rsf_form": spec["uses_exact_rsf_form"],
                "uses_reduced_rsf_form": spec["uses_reduced_rsf_form"],
                "derivative_rmse": performance_master.loc[performance_master["result_label"] == spec["display_name"], "derivative_rmse"].iloc[0] if spec["display_name"] in set(performance_master["result_label"]) else np.nan,
                "rollout_rmse": performance_master.loc[performance_master["result_label"] == spec["display_name"], "rollout_rmse"].iloc[0] if spec["display_name"] in set(performance_master["result_label"]) else np.nan,
                "stable_fraction": performance_master.loc[performance_master["result_label"] == spec["display_name"], "stable_fraction"].iloc[0] if spec["display_name"] in set(performance_master["result_label"]) else np.nan,
                "peak_timing_error_s": performance_master.loc[performance_master["result_label"] == spec["display_name"], "peak_timing_error_s"].iloc[0] if spec["display_name"] in set(performance_master["result_label"]) else np.nan,
                "onset_timing_error_s": performance_master.loc[performance_master["result_label"] == spec["display_name"], "onset_timing_error_s"].iloc[0] if spec["display_name"] in set(performance_master["result_label"]) else np.nan,
                "train_holdout_note": f"train={spec['train_steps']}; holdout={spec['holdout_steps']}",
                "identifiability_note": summary["final_judgment"],
                "what_is_special": spec["what_is_special"],
                "final_judgment": spec["final_judgment"],
            }
        )
        index_rows.append(
            {
                "display_name": spec["display_name"],
                "folder": spec["folder"],
                "category": spec["category"],
                "script_source": spec["script_source"],
                "what_is_special": spec["what_is_special"],
                "final_judgment": spec["final_judgment"],
            }
        )

    master_table = pd.DataFrame(master_rows)
    model_index = pd.DataFrame(index_rows)
    master_table.to_csv(FINAL_DIR / "final_master_table.csv", index=False)
    model_index.to_csv(FINAL_DIR / "final_model_index.csv", index=False)

    overall_rank = performance["ranking"]
    special_lines = [f"- {row['display_name']}: {row['what_is_special']}" for row in index_rows]
    report_lines = [
        "# Utah FORGE Final Master Report",
        "",
        "## 1. Overview",
        "- This folder assembles the old Utah FORGE p5838 A/B/C models together with the later proposal-recovery, exact-RSF, theta-check, tau-fix, velocity-isolation, and multistep assessment results.",
        "- Dataset used: RSFit-aligned Utah FORGE `p5838` steps with train/holdout split centered on `p5838_step2` and `p5838_step7` for the main proposal workflows.",
        "- Overall modeling philosophy: use sparse discovery for structure, keep the physics legible, and separate strongest recovered equations from weaker exact-form or conditional checks.",
        "",
        "## 2. Model Lineup",
        frame_to_markdown(model_index),
        "",
        "### What is special about each one",
    ]
    report_lines.extend(special_lines)
    report_lines.extend(
        [
            "",
            "## 3. Old Models: A / B / C",
            "- Model A: observed-only baseline with no memory surrogate and no theta proxy.",
            "- Model B: memory-augmented surrogate model using `tau_avg`, `tau_ema`, and `S`.",
            "- Model C: theta-informed model using supplied RSFit theta in the library.",
            f"- Old-family summary table:\n{frame_to_markdown(ablation)}",
            "",
            "## 4. Newer Proposal-Specific Results",
            f"- Best tau equation: `{proposal['tau_model']['exact_equation']}`",
            f"- One-term tau approximation: `{proposal['tau_model']['one_term_equation']}`",
            f"- Best reduced velocity fallback: `{proposal['final_velocity_model']['equation']}`",
            "- Relative to A/B/C, the newer workflow isolated equation (1), enforced a tiny physical tau library, and made the velocity branch explicitly confront RSF identifiability.",
            "",
            "## 5. Exact RSF Branch",
            f"- Exact tau equation: `dtau/dt = {exact_params['k']:.6e}*(V_drive - V)`",
            f"- Exact velocity equation: `dV/dt = (1/{exact_params['m']:.6e})*[tau - sigmaN*({exact_params['mu0']:.6e} + {exact_params['a']:.6e}*log(V/V0) + {exact_params['b']:.6e}*log(theta*V0/{exact_params['Dc']:.6e}))]`",
            f"- Exact theta equation: `dtheta/dt = 1 - V*theta/{exact_params['Dc']:.6e}`",
            f"- Why it is attractive: closest exact-form RSF-looking fit with multistart convergence and decent timing on the original holdout pair.",
            f"- Why it is not final: {performance['biggest_weakness']}",
            "",
            "## 6. Theta Consistency Branch",
            f"- Tiny-library conditional equation: `{theta_consistency['tiny_library']['equation']}`",
            f"- Conclusion: `{theta_consistency['conclusions']['consistency_strength']}` consistency only.",
            "",
            "## 7. Cross-Model Comparison",
            f"- Strongest recovered equation: `{performance['strongest_success']}`",
            "- Best final usable velocity model: reduced RSF fallback.",
            "- Most exact-form model: exact latent-state RSF multistart fit.",
            "- Most defensible old model: Model B as the best-balanced surrogate in the original holdout study.",
            "- Generalization warning: holdout steps `p5838_step2` and `p5838_step7` were not representative in the same way for all branches.",
            "",
            "## 8. Final Conclusions",
        ]
    )
    report_lines.extend([f"- {line}" for line in overall_rank])
    report_lines.extend(
        [
            f"- Main presentation claim: {performance['abstract_style_conclusion']}",
            "",
            "## Key Project-Level Figures",
            "- `overall_performance_comparison.png`",
            "- `train_vs_holdout_summary.png`",
            "- `step_difficulty_heatmap.png`",
            "- `exact_vs_reduced_summary.png`",
        ]
    )
    (FINAL_DIR / "final_master_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    final_json = {
        "overview": {
            "dataset": "Utah FORGE p5838 RSFit-aligned steps",
            "modeling_philosophy": "Sparse discovery plus physically targeted recovery, with explicit separation between compact recovered laws, reduced fallbacks, exact-form near-hits, and conditional checks.",
        },
        "ranking": performance["ranking"],
        "special_model_notes": index_rows,
        "master_rows": master_rows,
        "project_conclusion": performance["abstract_style_conclusion"],
    }
    (FINAL_DIR / "final_master_report.json").write_text(json.dumps(json_ready(final_json), indent=2), encoding="utf-8")
    (FINAL_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Utah FORGE Final Folder",
                "",
                "This folder is a clean aggregation of the full Utah FORGE p5838 model history.",
                "",
                "Main files:",
                "- `final_master_report.md`: human-readable master report.",
                "- `final_master_report.json`: machine-readable summary.",
                "- `final_master_table.csv`: one-row-per-model/result master table.",
                "- `final_model_index.csv`: quick index of included folders.",
                "",
                "Subfolders:",
                "- `Model_A`, `Model_B`, `Model_C`",
                "- `Proposal_Tau`, `Reduced_RSF`, `Exact_RSF`, `Theta_Consistency`",
                "- `BC_Tau_Fix`, `C_Velocity_Isolation`, `Multistep_Assessment`",
            ]
        ),
        encoding="utf-8",
    )
    print("[final-assembly] wrote Utah FORGE Final report package", flush=True)


if __name__ == "__main__":
    main()
