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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"


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


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


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
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def split_stats(frame: pd.DataFrame, metric: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for split_name, split_frame in frame.groupby("split"):
        values = split_frame[metric].astype(float)
        best_row = split_frame.loc[values.idxmin()]
        worst_row = split_frame.loc[values.idxmax()]
        out[str(split_name)] = {
            "n_steps": int(len(split_frame)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "best_step": str(best_row["step_name"]),
            "best_value": float(best_row[metric]),
            "worst_step": str(worst_row["step_name"]),
            "worst_value": float(worst_row[metric]),
        }
    return out


def difficulty_bucket(series: pd.Series, higher_is_harder: bool = True) -> pd.Series:
    values = series.astype(float)
    ranks = values.rank(method="average", ascending=higher_is_harder, pct=True)
    buckets = []
    for rank in ranks:
        if rank <= 1 / 3:
            buckets.append("easy")
        elif rank <= 2 / 3:
            buckets.append("medium")
        else:
            buckets.append("hard")
    return pd.Series(buckets, index=series.index)


def build_master_table(
    proposal: dict,
    exact_multistart: dict,
    theta_consistency: dict,
    multistep_tau: pd.DataFrame,
    multistep_velocity: pd.DataFrame,
    multistep_exact: pd.DataFrame,
    bc_tau_fix: dict,
) -> pd.DataFrame:
    tau_holdout = split_stats(multistep_tau, "tau_rollout_rmse").get("holdout", {})
    tau_train = split_stats(multistep_tau, "tau_rollout_rmse").get("train", {})
    reduced_holdout = split_stats(multistep_velocity, "velocity_rollout_rmse").get("holdout", {})
    reduced_train = split_stats(multistep_velocity, "velocity_rollout_rmse").get("train", {})
    exact_holdout = split_stats(multistep_exact, "velocity_rollout_rmse").get("holdout", {})
    exact_train = split_stats(multistep_exact, "velocity_rollout_rmse").get("train", {})

    final_velocity = proposal["final_velocity_model"]
    exact_fit = exact_multistart["best_run"]
    tiny_theta = theta_consistency["tiny_library"]

    rows = [
        {
            "result_label": "Best tau equation",
            "workflow_source": "scripts/utah_forge_proposal_equation_recovery.py",
            "equation_text": proposal["tau_model"]["exact_equation"],
            "training_split": proposal["insertion_note"]["train_steps"],
            "holdout_split": proposal["insertion_note"]["holdout_steps"],
            "derivative_mse": safe_float(proposal["tau_model"]["holdout_mse"]),
            "derivative_rmse": math.sqrt(float(proposal["tau_model"]["holdout_mse"])),
            "derivative_mae": None,
            "derivative_r2": None,
            "rollout_rmse": safe_float(tau_holdout.get("mean")),
            "rollout_rmse_train": safe_float(tau_train.get("mean")),
            "stable_fraction": 1.0,
            "onset_timing_error_s": None,
            "peak_timing_error_s": None,
            "identifiability_status": "strongly identifiable compact spring-loading law",
            "interpretability_status": "strongest recovered equation",
            "scientific_judgment": "strongest recovered equation",
        },
        {
            "result_label": "Reduced RSF fallback velocity",
            "workflow_source": "scripts/utah_forge_proposal_equation_recovery.py",
            "equation_text": final_velocity["equation"],
            "training_split": proposal["insertion_note"]["train_steps"],
            "holdout_split": proposal["insertion_note"]["holdout_steps"],
            "derivative_mse": safe_float(final_velocity["holdout_mse"]),
            "derivative_rmse": safe_float(final_velocity["holdout_rmse"]),
            "derivative_mae": safe_float(final_velocity["holdout_mae"]),
            "derivative_r2": safe_float(final_velocity["holdout_r2"]),
            "rollout_rmse": safe_float(reduced_holdout.get("mean")),
            "rollout_rmse_train": safe_float(reduced_train.get("mean")),
            "stable_fraction": safe_float(final_velocity["mean_stable_fraction"]),
            "onset_timing_error_s": safe_float(final_velocity["mean_onset_timing_error_s"]),
            "peak_timing_error_s": safe_float(final_velocity["mean_peak_timing_error_s"]),
            "identifiability_status": "usable reduced law; theta removed",
            "interpretability_status": "best final usable velocity law",
            "scientific_judgment": "best final usable model",
        },
        {
            "result_label": "Closest exact RSF-looking fit",
            "workflow_source": "scripts/utah_forge_exact_rsf_multistart_check.py",
            "equation_text": (
                f"dtau/dt = {exact_fit['parameters']['k']:.6e}*(V_drive - V); "
                f"dV/dt = (1/{exact_fit['parameters']['m']:.6e})*[tau - sigmaN*("
                f"{exact_fit['parameters']['mu0']:.6e} + {exact_fit['parameters']['a']:.6e}*log(V/V0) + "
                f"{exact_fit['parameters']['b']:.6e}*log(theta*V0/{exact_fit['parameters']['Dc']:.6e}))]; "
                f"dtheta/dt = 1 - V*theta/{exact_fit['parameters']['Dc']:.6e}"
            ),
            "training_split": proposal["insertion_note"]["train_steps"],
            "holdout_split": proposal["insertion_note"]["holdout_steps"],
            "derivative_mse": None,
            "derivative_rmse": None,
            "derivative_mae": None,
            "derivative_r2": None,
            "rollout_rmse": safe_float(exact_holdout.get("mean")),
            "rollout_rmse_train": safe_float(exact_train.get("mean")),
            "stable_fraction": safe_float(exact_fit["mean_holdout_stable_fraction"]),
            "onset_timing_error_s": safe_float(exact_fit["mean_holdout_onset_timing_error_s"]),
            "peak_timing_error_s": safe_float(exact_fit["mean_holdout_peak_timing_error_s"]),
            "identifiability_status": "non-identifiable exact-form fit",
            "interpretability_status": "closest exact fit but non-identifiable",
            "scientific_judgment": "closest exact fit but non-identifiable",
        },
        {
            "result_label": "Theta consistency check",
            "workflow_source": "scripts/utah_forge_theta_equation_consistency.py",
            "equation_text": tiny_theta["equation"],
            "training_split": ", ".join(theta_consistency["usable_subset"]["train_steps"]),
            "holdout_split": ", ".join(theta_consistency["usable_subset"]["holdout_steps"]),
            "derivative_mse": safe_float(tiny_theta["holdout_metrics"]["mse"]),
            "derivative_rmse": safe_float(tiny_theta["holdout_metrics"]["rmse"]),
            "derivative_mae": safe_float(tiny_theta["holdout_metrics"]["mae"]),
            "derivative_r2": safe_float(tiny_theta["holdout_metrics"]["r2"]),
            "rollout_rmse": None,
            "rollout_rmse_train": None,
            "stable_fraction": None,
            "onset_timing_error_s": None,
            "peak_timing_error_s": None,
            "identifiability_status": "weak conditional consistency only",
            "interpretability_status": "weak consistency check on supplied theta",
            "scientific_judgment": "weak consistency check",
        },
        {
            "result_label": "Tau-fixed Model B",
            "workflow_source": "scripts/utah_forge_model_bc_tau_fix_comparison.py",
            "equation_text": bc_tau_fix["tau_fixed_B"]["tau_equation"],
            "training_split": "original B/C split",
            "holdout_split": "original B/C holdout",
            "derivative_mse": safe_float(bc_tau_fix["tau_fixed_B"]["holdout_derivative"]["tau"]["mse"]),
            "derivative_rmse": safe_float(bc_tau_fix["tau_fixed_B"]["holdout_derivative"]["tau"]["rmse"]),
            "derivative_mae": safe_float(bc_tau_fix["tau_fixed_B"]["holdout_derivative"]["tau"]["mae"]),
            "derivative_r2": safe_float(bc_tau_fix["tau_fixed_B"]["holdout_derivative"]["tau"]["r2"]),
            "rollout_rmse": safe_float(bc_tau_fix["tau_fixed_B"]["holdout_rollout"]["mean_combined_rmse"]),
            "rollout_rmse_train": safe_float(bc_tau_fix["tau_fixed_B"]["train_rollout"]["mean_combined_rmse"]),
            "stable_fraction": safe_float(bc_tau_fix["tau_fixed_B"]["holdout_rollout"]["stable_fraction"]),
            "onset_timing_error_s": None,
            "peak_timing_error_s": None,
            "identifiability_status": "supporting ablation",
            "interpretability_status": "cleaner tau after isolation",
            "scientific_judgment": "supporting ablation result",
        },
        {
            "result_label": "Tau-fixed Model C",
            "workflow_source": "scripts/utah_forge_model_bc_tau_fix_comparison.py",
            "equation_text": bc_tau_fix["tau_fixed_C"]["tau_equation"],
            "training_split": "theta-screened Model C subset",
            "holdout_split": "theta-screened Model C holdout",
            "derivative_mse": safe_float(bc_tau_fix["tau_fixed_C"]["holdout_derivative"]["tau"]["mse"]),
            "derivative_rmse": safe_float(bc_tau_fix["tau_fixed_C"]["holdout_derivative"]["tau"]["rmse"]),
            "derivative_mae": safe_float(bc_tau_fix["tau_fixed_C"]["holdout_derivative"]["tau"]["mae"]),
            "derivative_r2": safe_float(bc_tau_fix["tau_fixed_C"]["holdout_derivative"]["tau"]["r2"]),
            "rollout_rmse": safe_float(bc_tau_fix["tau_fixed_C"]["holdout_rollout"]["mean_combined_rmse"]),
            "rollout_rmse_train": safe_float(bc_tau_fix["tau_fixed_C"]["train_rollout"]["mean_combined_rmse"]),
            "stable_fraction": safe_float(bc_tau_fix["tau_fixed_C"]["holdout_rollout"]["stable_fraction"]),
            "onset_timing_error_s": None,
            "peak_timing_error_s": None,
            "identifiability_status": "supporting ablation",
            "interpretability_status": "tau cleaned but velocity still theta-limited",
            "scientific_judgment": "supporting ablation result",
        },
    ]
    return pd.DataFrame(rows)


def build_step_difficulty_table(
    multistep_tau: pd.DataFrame,
    multistep_velocity: pd.DataFrame,
    multistep_exact: pd.DataFrame,
    proposal: dict,
) -> pd.DataFrame:
    inclusion = {
        str(row["step_name"]): row
        for row in proposal["inclusion_rows"]
        if str(row["step_name"]).startswith("p5838_step")
    }
    base = multistep_tau.merge(
        multistep_velocity[["step_name", "velocity_rollout_rmse", "stable_fraction", "peak_timing_error_s", "onset_timing_error_s"]],
        on="step_name",
        how="left",
        suffixes=("", "_reduced"),
    ).merge(
        multistep_exact[
            [
                "step_name",
                "tau_rollout_rmse",
                "velocity_rollout_rmse",
                "stable_fraction",
                "peak_timing_error_s",
                "onset_timing_error_s",
            ]
        ].rename(
            columns={
                "tau_rollout_rmse": "exact_tau_rollout_rmse",
                "velocity_rollout_rmse": "exact_velocity_rollout_rmse",
                "stable_fraction": "exact_stable_fraction",
                "peak_timing_error_s": "exact_peak_timing_error_s",
                "onset_timing_error_s": "exact_onset_timing_error_s",
            }
        ),
        on="step_name",
        how="left",
    )
    base["theta_quality"] = base["step_name"].map(
        lambda name: (
            "high-quality" if inclusion[name]["theta_event_valid"] else ("invalid" if "invalid" in str(inclusion[name]["theta_reason"]) else "low-quality")
        )
    )
    base["theta_reason"] = base["step_name"].map(lambda name: str(inclusion[name]["theta_reason"]))
    base["tau_difficulty"] = difficulty_bucket(base["tau_rollout_rmse"])
    base["reduced_difficulty"] = difficulty_bucket(base["velocity_rollout_rmse"])
    base["exact_difficulty"] = difficulty_bucket(base["exact_velocity_rollout_rmse"])

    notes = []
    for _, row in base.iterrows():
        row_notes = []
        if row["split"] == "holdout":
            row_notes.append("holdout step")
        if row["theta_quality"] != "high-quality":
            row_notes.append(f"theta {row['theta_quality']}")
        if float(row["exact_stable_fraction"]) < 0.1:
            row_notes.append("exact RSF unstable early")
        if float(row["velocity_rollout_rmse"]) > float(base["velocity_rollout_rmse"].median()):
            row_notes.append("reduced velocity harder than median")
        if float(row["tau_rollout_rmse"]) > float(base["tau_rollout_rmse"].median()):
            row_notes.append("tau harder than median")
        notes.append("; ".join(row_notes) if row_notes else "no major warning")
    base["difficulty_notes"] = notes
    return base[
        [
            "step_name",
            "split",
            "n_samples",
            "duration_s",
            "tau_rollout_rmse",
            "velocity_rollout_rmse",
            "exact_velocity_rollout_rmse",
            "theta_quality",
            "theta_reason",
            "tau_difficulty",
            "reduced_difficulty",
            "exact_difficulty",
            "difficulty_notes",
        ]
    ].sort_values(["split", "step_name"], kind="stable")


def overall_performance_figure(master: pd.DataFrame) -> None:
    focus = master.iloc[:4].copy()
    labels = focus["result_label"].tolist()
    derivative = [np.nan if pd.isna(v) else float(v) for v in focus["derivative_rmse"]]
    rollout = [np.nan if pd.isna(v) else float(v) for v in focus["rollout_rmse"]]
    stability = [np.nan if pd.isna(v) else float(v) for v in focus["stable_fraction"]]
    peak = [np.nan if pd.isna(v) else float(v) for v in focus["peak_timing_error_s"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        (axes[0, 0], derivative, "Holdout Derivative RMSE", True),
        (axes[0, 1], rollout, "Holdout Rollout RMSE", True),
        (axes[1, 0], stability, "Stable Fraction", False),
        (axes[1, 1], peak, "Peak Timing Error [s]", True),
    ]
    x = np.arange(len(labels))
    for ax, values, title, log_scale in panels:
        ax.bar(x, values, color=["#3b6fb6", "#5c9e31", "#cc6b2c", "#7a5195"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if log_scale:
            finite = [v for v in values if np.isfinite(v) and v > 0]
            if finite:
                ax.set_yscale("log")
        if title == "Stable Fraction":
            ax.set_ylim(0, 1.05)
    fig.suptitle("Utah FORGE Main Result Families", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS_DIR / "overall_performance_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_vs_holdout_figure(multistep_tau: pd.DataFrame, multistep_velocity: pd.DataFrame, multistep_exact: pd.DataFrame) -> None:
    tau_stats = split_stats(multistep_tau, "tau_rollout_rmse")
    reduced_stats = split_stats(multistep_velocity, "velocity_rollout_rmse")
    exact_stats = split_stats(multistep_exact, "velocity_rollout_rmse")
    labels = ["Tau", "Reduced V", "Exact RSF V"]
    train_vals = [tau_stats["train"]["mean"], reduced_stats["train"]["mean"], exact_stats["train"]["mean"]]
    holdout_vals = [tau_stats["holdout"]["mean"], reduced_stats["holdout"]["mean"], exact_stats["holdout"]["mean"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, train_vals, width=width, label="Train mean RMSE", color="#4c78a8")
    ax.bar(x + width / 2, holdout_vals, width=width, label="Holdout mean RMSE", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_title("Train vs Holdout Multistep Performance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "train_vs_holdout_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def step_difficulty_heatmap(step_table: pd.DataFrame) -> None:
    matrix = step_table[["tau_rollout_rmse", "velocity_rollout_rmse", "exact_velocity_rollout_rmse"]].astype(float).copy()
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(normalized.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Tau", "Reduced V", "Exact RSF V"])
    ax.set_yticks(range(len(step_table)))
    ax.set_yticklabels([f"{row.step_name} ({row.split[0].upper()})" for row in step_table.itertuples()])
    ax.set_title("Step Difficulty Heatmap (normalized RMSE)")
    fig.colorbar(im, ax=ax, label="relative difficulty")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "step_difficulty_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def exact_vs_reduced_figure(multistep_velocity: pd.DataFrame, multistep_exact: pd.DataFrame, proposal: dict, exact_multistart: dict) -> None:
    reduced_stats = split_stats(multistep_velocity, "velocity_rollout_rmse")
    exact_stats = split_stats(multistep_exact, "velocity_rollout_rmse")
    error_categories = [
        "train RMSE",
        "holdout RMSE",
        "peak timing",
        "onset timing",
    ]
    reduced_vals = [
        reduced_stats["train"]["mean"],
        reduced_stats["holdout"]["mean"],
        float(proposal["final_velocity_model"]["mean_stable_fraction"]),
        float(proposal["final_velocity_model"]["mean_peak_timing_error_s"]),
        float(proposal["final_velocity_model"]["mean_onset_timing_error_s"]),
    ]
    exact_vals = [
        exact_stats["train"]["mean"],
        exact_stats["holdout"]["mean"],
        float(exact_multistart["best_run"]["mean_holdout_stable_fraction"]),
        float(exact_multistart["best_run"]["mean_holdout_peak_timing_error_s"]),
        float(exact_multistart["best_run"]["mean_holdout_onset_timing_error_s"]),
    ]
    x = np.arange(len(error_categories))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(x - width / 2, reduced_vals[:2] + reduced_vals[3:], width=width, label="Reduced fallback", color="#59a14f")
    axes[0].bar(x + width / 2, exact_vals[:2] + exact_vals[3:], width=width, label="Exact RSF", color="#e15759")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(error_categories, rotation=20, ha="right")
    axes[0].set_title("Reduced vs Exact: error and timing")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar([0, 1], [reduced_vals[2], exact_vals[2]], color=["#59a14f", "#e15759"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Reduced fallback", "Exact RSF"])
    axes[1].set_ylim(0, max(reduced_vals[2], exact_vals[2]) * 1.25 + 1e-6)
    axes[1].set_title("Holdout stable fraction")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exact_vs_reduced_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_assessment_payload(
    proposal: dict,
    exact_multistart: dict,
    theta_consistency: dict,
    multistep_summary: dict,
    multistep_tau: pd.DataFrame,
    multistep_velocity: pd.DataFrame,
    multistep_exact: pd.DataFrame,
    master: pd.DataFrame,
    step_table: pd.DataFrame,
    robustness: dict,
    velocity_isolation: dict,
) -> dict:
    tau_split = split_stats(multistep_tau, "tau_rollout_rmse")
    reduced_split = split_stats(multistep_velocity, "velocity_rollout_rmse")
    exact_split = split_stats(multistep_exact, "velocity_rollout_rmse")
    strongest_success = (
        "The compact spring-loading tau law is the strongest result: it is physically correct, stable under alternate splits, and remains the cleanest recovered governing equation."
    )
    biggest_weakness = (
        "The theta-bearing exact RSF branch remains scientifically weak because near-constant sigmaN and parameter confounding prevent trustworthy identification even after multistart convergence improves."
    )
    ranking = [
        "1. Best tau equation: strongest recovered equation and most defensible compact law.",
        "2. Reduced RSF fallback velocity law: best final usable velocity model.",
        "3. Closest exact RSF-looking fit: attractive structurally but non-identifiable and weak on holdout rollout.",
        "4. Theta consistency check: weak conditional consistency only, not independent recovery.",
    ]
    abstract = (
        "The Utah FORGE p5838 equation-discovery project now supports a compact, physically sensible recovery of the spring-loading stress law and a reduced RSF-style fallback for the velocity law, but not a scientifically trustworthy recovery of the full theta-bearing RSF system. "
        "Across saved derivative, rollout, multistep, and multistart diagnostics, the tau equation remains the strongest and most robust result, while the reduced velocity law is the most usable final model despite limited multistep generalization. "
        "The closest exact RSF-looking fit is valuable as a structural near-hit because it preserves the full form and achieves decent timing on the original holdout pair, but it fails the trustworthiness test because parameter estimates remain confounded under near-constant sigmaN and holdout rollout degrades sharply outside the training-like steps. "
        "Conditional theta checks likewise remain weak, indicating that externally reconstructed theta carries suggestive structure but does not validate a clean standalone RSF state recovery on the current subset."
    )
    professor_bullets = [
        "Equation (1) is the strongest success: compact, physically interpretable, and robust under bounded split changes.",
        "The reduced RSF fallback is the best final usable velocity law, not the exact theta-bearing form.",
        "The original holdout pair was not representative for tau or exact RSF; both holdout steps were unusually hard there.",
        "The original holdout pair was unusually favorable for the reduced velocity rollout, so that branch should be presented as usable rather than universally strong.",
        "Multistart improved convergence for exact RSF but did not fix identifiability or parameter trustworthiness.",
        "The project’s defensible claim is compact stress-law recovery plus reduced RSF structure, not credible exact latent-state RSF recovery.",
    ]
    oral = [
        "The strongest result is the tau equation, because it became compact, physically correct, and stayed robust under bounded split checks.",
        "The best final usable velocity model is the reduced RSF fallback, not the full exact theta-bearing form.",
        "The exact RSF-looking fit is still worth showing because it is the closest structural match to the proposal, but it is not trustworthy enough to be the final model.",
        "The main reason is identifiability, not just optimization: sigmaN is nearly constant and the exact RSF parameters stay confounded across starts.",
        "When we expanded to all usable steps, the original holdout pair turned out to be unusually hard for tau and exact RSF, but unusually favorable for the reduced velocity rollout.",
        "So the honest project claim is strong recovery of equation (1), support for a reduced RSF-like equation (2), and no credible full recovery of the theta-bearing exact RSF system.",
    ]
    return {
        "ranking": ranking,
        "strongest_success": strongest_success,
        "biggest_weakness": biggest_weakness,
        "holdout_representative": False,
        "exact_rsf_failure_mode": "mostly identifiability, not mostly optimization",
        "split_statistics": {
            "tau": tau_split,
            "reduced_velocity": reduced_split,
            "exact_rsf": exact_split,
        },
        "multistep_representativeness": multistep_summary["representativeness"],
        "exact_identifiability": {
            "sigma_too_constant": bool(exact_multistart["best_identifiability"]["sigma_too_constant"]),
            "parameter_confounding_flag": bool(exact_multistart["best_identifiability"]["parameter_confounding_flag"]),
            "best_jtj_condition_number": float(exact_multistart["best_identifiability"]["jtj_condition_number"]),
            "best_jtj_rank": int(exact_multistart["best_identifiability"]["jtj_rank"]),
            "parameter_stability_cv": {k: float(v["cv"]) for k, v in exact_multistart["parameter_stability"].items()},
            "theta_term_meaningfully_active": bool(exact_multistart["checks"]["theta_term_becomes_meaningfully_active"]),
            "parameter_estimates_stabilize": bool(exact_multistart["checks"]["parameter_estimates_stabilize_across_starts"]),
        },
        "theta_consistency": theta_consistency["conclusions"],
        "robustness": robustness["aggregate"],
        "velocity_isolation_support": velocity_isolation["conclusions"],
        "master_rows": master.to_dict(orient="records"),
        "step_difficulty_rows": step_table.to_dict(orient="records"),
        "abstract_style_conclusion": abstract,
        "professor_summary": professor_bullets,
        "oral_explanation": oral,
    }


def write_markdown(
    payload: dict,
    master: pd.DataFrame,
    step_table: pd.DataFrame,
    multistep_summary: dict,
) -> None:
    lines = [
        "# Utah FORGE Project Performance Assessment",
        "",
        "## Final Ranking",
    ]
    lines.extend([f"- {line}" for line in payload["ranking"]])
    lines.extend(
        [
            "",
            "## Strongest Success",
            f"- {payload['strongest_success']}",
            "",
            "## Biggest Weakness",
            f"- {payload['biggest_weakness']}",
            "",
            "## Master Comparison",
            frame_to_markdown(master),
            "",
            "## Train vs Holdout Generalization",
            f"- Tau train mean RMSE: `{payload['split_statistics']['tau']['train']['mean']:.4f}`; holdout mean RMSE: `{payload['split_statistics']['tau']['holdout']['mean']:.4f}`",
            f"- Reduced velocity train mean RMSE: `{payload['split_statistics']['reduced_velocity']['train']['mean']:.4f}`; holdout mean RMSE: `{payload['split_statistics']['reduced_velocity']['holdout']['mean']:.4f}`",
            f"- Exact RSF train mean RMSE: `{payload['split_statistics']['exact_rsf']['train']['mean']:.4f}`; holdout mean RMSE: `{payload['split_statistics']['exact_rsf']['holdout']['mean']:.4f}`",
            f"- Holdout representativeness: tau `{multistep_summary['representativeness']['tau_equation']}`",
            f"- Holdout representativeness: exact RSF `{multistep_summary['representativeness']['exact_rsf_velocity']}`",
            "",
            "## Step Difficulty",
            frame_to_markdown(step_table),
            "",
            "## Error Decomposition",
            "- The tau law is comparatively strong because it is evaluated in a semi-observed way and only needs to map the observed loading and slip-rate path into a stress evolution; that is much easier than predicting the full velocity trajectory.",
            "- The reduced fallback keeps the most stable local RSF-like ingredient, the negative `sigmaN*log(V/V0)` term, so it performs best as a usable velocity law even though its rollout quality varies a lot across steps.",
            "- The exact RSF fit is structurally attractive because it keeps the full coupled form and can achieve good timing on the original holdout pair, but its rollout error blows up on the hardest holdout steps and its stability remains poor.",
            "- Theta consistency remains weak because the supplied theta signal gives the right sign on the `V*theta` term but not the expected intercept or implied `Dc`, so it is not a strong independent validation of equation (3).",
            "",
            "## Identifiability",
            f"- `sigmaN` too constant: `{payload['exact_identifiability']['sigma_too_constant']}`",
            f"- Best JTJ condition number after multistart: `{payload['exact_identifiability']['best_jtj_condition_number']:.6e}`",
            f"- Best JTJ rank after multistart: `{payload['exact_identifiability']['best_jtj_rank']}`",
            f"- Parameter estimates stabilized across starts: `{payload['exact_identifiability']['parameter_estimates_stabilize']}`",
            f"- Theta became numerically active: `{payload['exact_identifiability']['theta_term_meaningfully_active']}`",
            "- Scientific interpretation: multistart solved convergence much better than the baseline exact fit, but it did not solve trustworthiness because the parameter set remains confounded and unstable.",
            "",
            "## Final Scientific Judgment",
            f"- Strongest result: {payload['strongest_success']}",
            "- Best final usable model: the reduced RSF fallback velocity law.",
            "- Best exact-form result: the multistart exact RSF-looking fit, but it remains non-identifiable.",
            "- Unresolved issue: trustworthy theta-bearing exact RSF recovery on the current Utah FORGE subset.",
            "- Confident claim: Equation (1) is recovered compactly and equation (2) supports a reduced RSF-like form with persistent log-rate structure.",
            "- Cautious claim: the full exact theta-bearing RSF system is implemented and tested directly, but still not scientifically identifiable from the present data.",
            "",
            "## Abstract-Style Conclusion",
            payload["abstract_style_conclusion"],
            "",
            "## Professor-Facing Summary",
        ]
    )
    lines.extend([f"- {line}" for line in payload["professor_summary"]])
    lines.extend(["", "## Oral Explanation"])
    lines.extend([f"- {line}" for line in payload["oral_explanation"]])
    (RESULTS_DIR / "project_performance_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    proposal = load_json("proposal_equation_recovery.json")
    exact_multistart = load_json("exact_rsf_multistart_summary.json")
    theta_consistency = load_json("theta_equation_consistency.json")
    multistep_summary = load_json("multistep_rollout_summary.json")
    bc_tau_fix = load_json("model_BC_tau_fix_comparison.json")
    velocity_isolation = load_json("model_C_velocity_isolation_comparison.json")
    robustness = load_json("proposal_equation_robustness_summary.json")
    multistep_tau = load_csv("multistep_tau_table.csv")
    multistep_velocity = load_csv("multistep_velocity_table.csv")
    multistep_exact = load_csv("multistep_exact_rsf_table.csv")

    master = build_master_table(
        proposal,
        exact_multistart,
        theta_consistency,
        multistep_tau,
        multistep_velocity,
        multistep_exact,
        bc_tau_fix,
    )
    step_table = build_step_difficulty_table(multistep_tau, multistep_velocity, multistep_exact, proposal)
    payload = build_assessment_payload(
        proposal,
        exact_multistart,
        theta_consistency,
        multistep_summary,
        multistep_tau,
        multistep_velocity,
        multistep_exact,
        master,
        step_table,
        robustness,
        velocity_isolation,
    )

    master.to_csv(RESULTS_DIR / "project_performance_master_table.csv", index=False)
    step_table.to_csv(RESULTS_DIR / "project_step_difficulty_table.csv", index=False)
    overall_performance_figure(master)
    train_vs_holdout_figure(multistep_tau, multistep_velocity, multistep_exact)
    step_difficulty_heatmap(step_table)
    exact_vs_reduced_figure(multistep_velocity, multistep_exact, proposal, exact_multistart)
    write_markdown(payload, master, step_table, multistep_summary)
    (RESULTS_DIR / "project_performance_assessment.json").write_text(
        json.dumps(json_ready(payload), indent=2),
        encoding="utf-8",
    )
    print("[project-performance] wrote master assessment package", flush=True)


if __name__ == "__main__":
    main()
