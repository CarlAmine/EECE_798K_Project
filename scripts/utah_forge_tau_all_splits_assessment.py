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

from scripts import utah_forge_regime_balanced_tau_evaluation as balanced_eval
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
STEP_DIAG_JSON = RESULTS_DIR / "step_variability_diagnostics.json"
SHIFT_JSON = RESULTS_DIR / "holdout_shift_assessment.json"
SPLIT_JSON = RESULTS_DIR / "tau_split_strategy_comparison.json"

OUTPUT_MD = RESULTS_DIR / "tau_all_splits_assessment.md"
OUTPUT_JSON = RESULTS_DIR / "tau_all_splits_assessment.json"
OUTPUT_TABLE = RESULTS_DIR / "tau_all_splits_table.csv"
STEP_TABLE = RESULTS_DIR / "tau_step_difficulty_table.csv"
PAIR_TABLE = RESULTS_DIR / "tau_pair_difficulty_table.csv"

FIG_SINGLE_RANK = RESULTS_DIR / "tau_single_step_holdout_ranking.png"
FIG_PAIR_HEATMAP = RESULTS_DIR / "tau_leave_two_out_heatmap.png"
FIG_STEP2_STEP5 = RESULTS_DIR / "step2_vs_step5_comparison.png"
FIG_COEFFS = RESULTS_DIR / "tau_coefficients_across_splits.png"
FIG_DISTRIBUTION = RESULTS_DIR / "tau_split_distribution_summary.png"
FIG_EASY_HARD = RESULTS_DIR / "tau_easy_vs_hard_examples.png"

FINALV3_DIR = RESULTS_DIR / "Finalv3"
FINALV3_FIG_DIR = FINALV3_DIR / "Figures"

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
BALANCED_PAIR = ("p5838_step2", "p5838_step5")
SELECTED_LEAVE_THREE_OUT = [
    ("p5838_step2", "p5838_step7", "p5838_step9"),
    ("p5838_step2", "p5838_step5", "p5838_step8"),
    ("p5838_step3", "p5838_step8", "p5838_step10"),
    ("p5838_step4", "p5838_step5", "p5838_step7"),
]
EPS = 1e-12


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def duration_seconds(segment_df: pd.DataFrame) -> float:
    return float(segment_df["time"].iloc[-1] - segment_df["time"].iloc[0])


def evaluate_holdout_family(
    family_name: str,
    holdout_sets: list[tuple[str, ...]],
    prepared_map: dict[str, pd.DataFrame],
) -> list[dict]:
    rows: list[dict] = []
    for holdout_steps in holdout_sets:
        train_steps = [step for step in ALL_STEPS if step not in holdout_steps]
        summary = balanced_eval.evaluate_split(
            name=f"{family_name}_{'__'.join(holdout_steps)}",
            label=f"{family_name}: {', '.join(holdout_steps)}",
            train_steps=train_steps,
            holdout_steps=list(holdout_steps),
            prepared_map=prepared_map,
        )
        summary["family"] = family_name
        summary["holdout_size"] = len(holdout_steps)
        rows.append(summary)
    return rows


def summarize_step_difficulty(split_rows: list[dict], variability_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    variability_map = variability_df.set_index("step_name").to_dict(orient="index")
    for step in ALL_STEPS:
        per_step = []
        for split in split_rows:
            for row in split["holdout_rows"]:
                if row["step_name"] == step:
                    per_step.append(
                        {
                            "family": split["family"],
                            "tau_rollout_rmse": row["tau_rollout_rmse"],
                            "tau_rollout_r2": row["tau_rollout_r2"],
                            "derivative_rmse": row["derivative_rmse"],
                            "mean_abs_tau_error": row["mean_abs_tau_error"],
                            "max_abs_tau_error": row["max_abs_tau_error"],
                        }
                    )
        rmse_vals = np.array([row["tau_rollout_rmse"] for row in per_step], dtype=float)
        deriv_vals = np.array([row["derivative_rmse"] for row in per_step], dtype=float)
        record = {
            "step_name": step,
            "n_holdout_appearances": int(len(per_step)),
            "mean_holdout_tau_rmse": float(np.mean(rmse_vals)),
            "median_holdout_tau_rmse": float(np.median(rmse_vals)),
            "std_holdout_tau_rmse": float(np.std(rmse_vals)),
            "min_holdout_tau_rmse": float(np.min(rmse_vals)),
            "max_holdout_tau_rmse": float(np.max(rmse_vals)),
            "mean_derivative_rmse": float(np.mean(deriv_vals)),
            "median_derivative_rmse": float(np.median(deriv_vals)),
            "single_step_holdout_rmse": float(next(split["mean_tau_rollout_rmse"] for split in split_rows if split["family"] == "single_step" and split["holdout_steps"] == [step])),
            "tau_variability_rank": int(variability_map[step]["tau_variability_rank"]),
            "tau_total_variation_per_s": float(variability_map[step]["tau_total_variation_per_s"]),
            "dtau_std": float(variability_map[step]["dtau_std"]),
            "V_drive_minus_V_std": float(variability_map[step]["V_drive_minus_V_std"]),
            "duration_s": float(variability_map[step]["duration_s"]),
        }
        records.append(record)
    frame = pd.DataFrame(records).sort_values("mean_holdout_tau_rmse", ascending=True).reset_index(drop=True)
    q1 = float(frame["mean_holdout_tau_rmse"].quantile(1 / 3))
    q2 = float(frame["mean_holdout_tau_rmse"].quantile(2 / 3))
    labels = []
    for value in frame["mean_holdout_tau_rmse"]:
        if value <= q1:
            labels.append("easy")
        elif value <= q2:
            labels.append("medium")
        else:
            labels.append("hard")
    frame["difficulty_label"] = labels
    frame["difficulty_rank"] = np.arange(1, len(frame) + 1)
    return frame


def summarize_pair_difficulty(pair_rows: list[dict], centroid_map: dict[str, float] | None = None) -> pd.DataFrame:
    records: list[dict] = []
    for split in pair_rows:
        steps = tuple(split["holdout_steps"])
        per_step_rollout = {row["step_name"]: row["tau_rollout_rmse"] for row in split["holdout_rows"]}
        per_step_derivative = {row["step_name"]: row["derivative_rmse"] for row in split["holdout_rows"]}
        records.append(
            {
                "pair_name": " + ".join(steps),
                "step_a": steps[0],
                "step_b": steps[1],
                "mean_pair_tau_rollout_rmse": float(split["mean_tau_rollout_rmse"]),
                "mean_pair_derivative_rmse": float(split["mean_derivative_rmse"]),
                "pair_tau_rollout_mae": float(split["mean_abs_tau_error"]),
                "pair_tau_rollout_r2": float(split["mean_tau_rollout_r2"]),
                "step_a_rollout_rmse": float(per_step_rollout[steps[0]]),
                "step_b_rollout_rmse": float(per_step_rollout[steps[1]]),
                "step_a_derivative_rmse": float(per_step_derivative[steps[0]]),
                "step_b_derivative_rmse": float(per_step_derivative[steps[1]]),
                "pair_shift_distance_from_global_center": float(split.get("pair_shift_distance_from_global_center", float("nan"))),
                "is_current_pair": bool(tuple(steps) == tuple(CURRENT_PAIR)),
                "is_balanced_pair": bool(tuple(steps) == tuple(BALANCED_PAIR)),
                "equation": split["equation"],
            }
        )
    frame = pd.DataFrame(records).sort_values("mean_pair_tau_rollout_rmse", ascending=True).reset_index(drop=True)
    frame["pair_rank"] = np.arange(1, len(frame) + 1)
    if centroid_map:
        frame["distance_minus_median_abs"] = np.abs(frame["mean_pair_tau_rollout_rmse"] - centroid_map["median_pair_rmse"])
    return frame


def split_rows_table(split_rows: list[dict]) -> pd.DataFrame:
    records = []
    for split in split_rows:
        coeffs = split["coefficients_physical"]
        for row in split["holdout_rows"]:
            records.append(
                {
                    "family": split["family"],
                    "split_name": split["split_name"],
                    "holdout_size": split["holdout_size"],
                    "train_steps": "|".join(split["train_steps"]),
                    "holdout_steps": "|".join(split["holdout_steps"]),
                    "equation": split["equation"],
                    "one_term_equation": split["one_term_equation"],
                    "coef_1": float(coeffs.get("1", 0.0)),
                    "coef_V": float(coeffs.get("V", 0.0)),
                    "coef_V_drive_minus_V": float(coeffs.get("V_drive_minus_V", 0.0)),
                    "step_name": row["step_name"],
                    "duration_s": row["duration_s"],
                    "n_samples": row["n_samples"],
                    "derivative_rmse": row["derivative_rmse"],
                    "derivative_mae": row["derivative_mae"],
                    "derivative_r2": row["derivative_r2"],
                    "tau_rollout_rmse": row["tau_rollout_rmse"],
                    "mean_abs_tau_error": row["mean_abs_tau_error"],
                    "max_abs_tau_error": row["max_abs_tau_error"],
                    "tau_rollout_r2": row["tau_rollout_r2"],
                    "split_mean_tau_rollout_rmse": split["mean_tau_rollout_rmse"],
                    "split_mean_derivative_rmse": split["mean_derivative_rmse"],
                }
            )
    return pd.DataFrame(records)


def normalized_time(segment_df: pd.DataFrame) -> np.ndarray:
    time = segment_df["time"].to_numpy(dtype=float)
    duration = max(time[-1] - time[0], EPS)
    return (time - time[0]) / duration


def holdout_prediction(split_row: dict, step_name: str, prepared_map: dict[str, pd.DataFrame]) -> dict:
    segment_df = prepared_map[step_name]
    derivative = balanced_eval.tau_derivative_metrics(split_row["coefficients_physical"], segment_df)
    rollout = balanced_eval.tau_rollout_metrics(split_row["coefficients_physical"], segment_df)
    return {
        "segment": segment_df,
        "tau_pred": rollout["tau_prediction"],
        "tau_abs_error": rollout["abs_error"],
        "dtau_pred": derivative["prediction"],
    }


def plot_single_step_ranking(step_df: pd.DataFrame) -> None:
    frame = step_df.sort_values("mean_holdout_tau_rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:green" if label == "easy" else "tab:orange" if label == "medium" else "tab:red" for label in frame["difficulty_label"]]
    ax.bar(frame["step_name"], frame["mean_holdout_tau_rmse"], yerr=frame["std_holdout_tau_rmse"], color=colors, alpha=0.85, capsize=3)
    ax.set_title("Compact tau law: average semi-observed holdout RMSE by step")
    ax.set_ylabel("mean holdout tau rollout RMSE")
    ax.set_xlabel("step")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_SINGLE_RANK, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_heatmap(pair_df: pd.DataFrame) -> None:
    heat = pd.DataFrame(np.nan, index=ALL_STEPS, columns=ALL_STEPS, dtype=float)
    for _, row in pair_df.iterrows():
        heat.loc[row["step_a"], row["step_b"]] = row["mean_pair_tau_rollout_rmse"]
        heat.loc[row["step_b"], row["step_a"]] = row["mean_pair_tau_rollout_rmse"]
    heat_values = heat.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(heat_values, 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(heat_values, cmap="magma_r", aspect="auto")
    ax.set_xticks(np.arange(len(ALL_STEPS)))
    ax.set_yticks(np.arange(len(ALL_STEPS)))
    ax.set_xticklabels(ALL_STEPS, rotation=35, ha="right")
    ax.set_yticklabels(ALL_STEPS)
    for i in range(len(ALL_STEPS)):
        for j in range(len(ALL_STEPS)):
            value = float(heat.iloc[i, j])
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white" if value > np.nanmedian(heat.to_numpy(dtype=float)) else "black")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("pair mean tau rollout RMSE")
    ax.set_title("Leave-two-out semi-observed tau rollout RMSE")
    fig.tight_layout()
    fig.savefig(FIG_PAIR_HEATMAP, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_step2_vs_step5(
    single_rows: list[dict],
    prepared_map: dict[str, pd.DataFrame],
) -> None:
    step_rows = {
        row["holdout_steps"][0]: row
        for row in single_rows
    }
    pred2 = holdout_prediction(step_rows["p5838_step2"], "p5838_step2", prepared_map)
    pred5 = holdout_prediction(step_rows["p5838_step5"], "p5838_step5", prepared_map)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for idx, (step, pred) in enumerate((("p5838_step2", pred2), ("p5838_step5", pred5))):
        seg = pred["segment"]
        rel_t = seg["time"].to_numpy(dtype=float) - float(seg["time"].iloc[0])
        axes[idx, 0].plot(rel_t, seg["tau"], label="observed tau", linewidth=1.4)
        axes[idx, 0].plot(rel_t, pred["tau_pred"], label="predicted tau", linewidth=1.2)
        axes[idx, 0].set_title(f"{step}: single-step holdout tau rollout")
        axes[idx, 0].set_ylabel("tau")
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend(fontsize=8)

        axes[idx, 1].plot(rel_t, pred["tau_abs_error"], color="tab:red", linewidth=1.2)
        axes[idx, 1].set_title(f"{step}: absolute tau error over time")
        axes[idx, 1].set_ylabel("|tau error|")
        axes[idx, 1].grid(True, alpha=0.3)

        axes[idx, 2].plot(rel_t, seg["V"], label="V", linewidth=1.2)
        axes[idx, 2].plot(rel_t, seg["V_drive"] - seg["V"], label="V_drive - V", linewidth=1.2)
        axes[idx, 2].set_title(f"{step}: forcing context")
        axes[idx, 2].set_ylabel("context")
        axes[idx, 2].grid(True, alpha=0.3)
        axes[idx, 2].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle("Step2 vs Step5 under single-step holdout fits", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_STEP2_STEP5, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients_across_splits(split_rows: list[dict]) -> None:
    records = []
    for split in split_rows:
        coeffs = split["coefficients_physical"]
        for term in ("1", "V", "V_drive_minus_V"):
            records.append(
                {
                    "family": split["family"],
                    "term": term,
                    "coefficient": float(coeffs.get(term, 0.0)),
                }
            )
    frame = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    for ax, term in zip(axes, ("1", "V", "V_drive_minus_V")):
        subset = frame.loc[frame["term"] == term]
        families = list(dict.fromkeys(subset["family"].tolist()))
        grouped = [subset.loc[subset["family"] == family, "coefficient"].to_numpy(dtype=float) for family in families]
        ax.boxplot(grouped, labels=families, patch_artist=True, boxprops={"facecolor": "#bcd2ee", "alpha": 0.9}, medianprops={"color": "#1d3557"})
        for idx, values in enumerate(grouped, start=1):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), idx, dtype=float) + jitter, values, color="#274c77", s=12, alpha=0.6)
        ax.set_title(f"Coefficient across splits: {term}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("")
    fig.suptitle("Compact tau coefficient variation across holdout families", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_COEFFS, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(split_rows: list[dict]) -> None:
    frame = pd.DataFrame(
        [
            {
                "family": split["family"],
                "holdout_size": split["holdout_size"],
                "mean_tau_rollout_rmse": split["mean_tau_rollout_rmse"],
                "mean_derivative_rmse": split["mean_derivative_rmse"],
            }
            for split in split_rows
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    families = list(dict.fromkeys(frame["family"].tolist()))
    colors = {"single_step": "#457b9d", "leave_two_out": "#e76f51", "selected_leave_three_out": "#6a994e"}
    bins = np.linspace(float(frame["mean_tau_rollout_rmse"].min()), float(frame["mean_tau_rollout_rmse"].max()), 16)
    for family in families:
        vals = frame.loc[frame["family"] == family, "mean_tau_rollout_rmse"].to_numpy(dtype=float)
        axes[0].hist(vals, bins=bins, alpha=0.45, label=family, color=colors.get(family))
    axes[0].set_title("Distribution of split-level tau rollout RMSE")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    grouped = [frame.loc[frame["family"] == family, "mean_tau_rollout_rmse"].to_numpy(dtype=float) for family in families]
    axes[1].boxplot(grouped, labels=families, patch_artist=True, boxprops={"facecolor": "#d8e2dc", "alpha": 0.9}, medianprops={"color": "#7f5539"})
    for idx, values in enumerate(grouped, start=1):
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        axes[1].scatter(np.full(len(values), idx, dtype=float) + jitter, values, color="#7f5539", s=14, alpha=0.7)
    axes[1].set_title("Split-level tau rollout RMSE by family")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Compact tau semi-observed holdout performance across many splits", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DISTRIBUTION, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_easy_hard_examples(pair_df: pd.DataFrame, pair_rows: list[dict], prepared_map: dict[str, pd.DataFrame]) -> None:
    chosen = pair_df.sort_values("mean_pair_tau_rollout_rmse").iloc[[0, 1, -2, -1]]
    row_map = {tuple(row["holdout_steps"]): row for row in pair_rows}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for ax, (_, chosen_row) in zip(axes.flatten(), chosen.iterrows()):
        split = row_map[(chosen_row["step_a"], chosen_row["step_b"])]
        step = split["holdout_steps"][0]
        pred = holdout_prediction(split, step, prepared_map)
        seg = pred["segment"]
        rel_t = seg["time"].to_numpy(dtype=float) - float(seg["time"].iloc[0])
        ax.plot(rel_t, seg["tau"], label="observed", linewidth=1.3)
        ax.plot(rel_t, pred["tau_pred"], label="predicted", linewidth=1.1)
        ax.set_title(f"{chosen_row['pair_name']} | example {step}\nmean pair RMSE={chosen_row['mean_pair_tau_rollout_rmse']:.3f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("tau")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Easy vs hard leave-two-out examples", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_EASY_HARD, dpi=200, bbox_inches="tight")
    plt.close(fig)


def classify_pair_representativeness(pair_df: pd.DataFrame) -> dict:
    median_pair_rmse = float(pair_df["mean_pair_tau_rollout_rmse"].median())
    current_row = pair_df.loc[pair_df["is_current_pair"]].iloc[0]
    balanced_row = pair_df.loc[pair_df["is_balanced_pair"]].iloc[0]
    return {
        "median_pair_rmse": median_pair_rmse,
        "current_pair_rank": int(current_row["pair_rank"]),
        "balanced_pair_rank": int(balanced_row["pair_rank"]),
        "current_pair_rmse": float(current_row["mean_pair_tau_rollout_rmse"]),
        "balanced_pair_rmse": float(balanced_row["mean_pair_tau_rollout_rmse"]),
        "current_pair_is_harsh": bool(current_row["pair_rank"] >= len(pair_df) - 1),
        "balanced_pair_is_representative_by_rank_gap": bool(abs(float(balanced_row["mean_pair_tau_rollout_rmse"]) - median_pair_rmse) <= 0.10 * max(median_pair_rmse, EPS)),
        "balanced_pair_is_optimistic": bool(float(balanced_row["mean_pair_tau_rollout_rmse"]) < 0.8 * median_pair_rmse),
    }


def build_step2_vs_step5_explanation(
    step_df: pd.DataFrame,
    variability_df: pd.DataFrame,
    single_rows: list[dict],
    prepared_map: dict[str, pd.DataFrame],
) -> dict:
    single_map = {row["holdout_steps"][0]: row for row in single_rows}
    step_summary = step_df.set_index("step_name").to_dict(orient="index")
    variability_map = variability_df.set_index("step_name").to_dict(orient="index")
    pred2 = holdout_prediction(single_map["p5838_step2"], "p5838_step2", prepared_map)
    pred5 = holdout_prediction(single_map["p5838_step5"], "p5838_step5", prepared_map)

    def shape_features(step_name: str, pred: dict) -> dict:
        seg = pred["segment"]
        tau_true = seg["tau"].to_numpy(dtype=float)
        tau_pred = pred["tau_pred"]
        idx_true = int(np.argmax(tau_true))
        idx_pred = int(np.argmax(tau_pred))
        time = seg["time"].to_numpy(dtype=float)
        duration = max(time[-1] - time[0], EPS)
        return {
            "peak_time_fraction_error": float(abs((time[idx_pred] - time[0]) - (time[idx_true] - time[0])) / duration),
            "peak_amplitude_error": float(abs(tau_pred[idx_pred] - tau_true[idx_true])),
            "range_true": float(np.ptp(tau_true)),
            "range_pred": float(np.ptp(tau_pred)),
        }

    shape2 = shape_features("p5838_step2", pred2)
    shape5 = shape_features("p5838_step5", pred5)
    return {
        "step2": {
            "global_mean_holdout_rmse": float(step_summary["p5838_step2"]["mean_holdout_tau_rmse"]),
            "single_step_holdout_rmse": float(step_summary["p5838_step2"]["single_step_holdout_rmse"]),
            "tau_variability_rank": int(variability_map["p5838_step2"]["tau_variability_rank"]),
            "tau_total_variation_per_s": float(variability_map["p5838_step2"]["tau_total_variation_per_s"]),
            "dtau_std": float(variability_map["p5838_step2"]["dtau_std"]),
            "V_std": float(variability_map["p5838_step2"]["V_std"]),
            "V_drive_minus_V_std": float(variability_map["p5838_step2"]["V_drive_minus_V_std"]),
            "duration_s": float(variability_map["p5838_step2"]["duration_s"]),
            "tau_std_ratio_prepared_over_raw": float(variability_map["p5838_step2"]["tau_std_ratio_prepared_over_raw"]),
            "tau_total_variation_ratio_prepared_over_raw": float(variability_map["p5838_step2"]["tau_total_variation_ratio_prepared_over_raw"]),
            "dtau_std_ratio_prepared_over_raw": float(variability_map["p5838_step2"]["dtau_std_ratio_prepared_over_raw"]),
            "shape_features": shape2,
        },
        "step5": {
            "global_mean_holdout_rmse": float(step_summary["p5838_step5"]["mean_holdout_tau_rmse"]),
            "single_step_holdout_rmse": float(step_summary["p5838_step5"]["single_step_holdout_rmse"]),
            "tau_variability_rank": int(variability_map["p5838_step5"]["tau_variability_rank"]),
            "tau_total_variation_per_s": float(variability_map["p5838_step5"]["tau_total_variation_per_s"]),
            "dtau_std": float(variability_map["p5838_step5"]["dtau_std"]),
            "V_std": float(variability_map["p5838_step5"]["V_std"]),
            "V_drive_minus_V_std": float(variability_map["p5838_step5"]["V_drive_minus_V_std"]),
            "duration_s": float(variability_map["p5838_step5"]["duration_s"]),
            "tau_std_ratio_prepared_over_raw": float(variability_map["p5838_step5"]["tau_std_ratio_prepared_over_raw"]),
            "tau_total_variation_ratio_prepared_over_raw": float(variability_map["p5838_step5"]["tau_total_variation_ratio_prepared_over_raw"]),
            "dtau_std_ratio_prepared_over_raw": float(variability_map["p5838_step5"]["dtau_std_ratio_prepared_over_raw"]),
            "shape_features": shape5,
        },
    }


def build_report(
    step_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    representativeness: dict,
    step2_vs_step5: dict,
    split_summary: dict,
    selected_three_df: pd.DataFrame,
) -> str:
    easiest_steps = ", ".join(step_df.head(3)["step_name"].tolist())
    hardest_steps = ", ".join(step_df.tail(3)["step_name"].tolist())
    best_pairs = ", ".join(pair_df.head(3)["pair_name"].tolist())
    worst_pairs = ", ".join(pair_df.tail(3)["pair_name"].tolist())
    representative_step = step_df.iloc[(step_df["mean_holdout_tau_rmse"] - step_df["mean_holdout_tau_rmse"].median()).abs().argmin()]["step_name"]
    balanced_rank = representativeness["balanced_pair_rank"]
    current_rank = representativeness["current_pair_rank"]

    summary_table = pd.DataFrame(
        [
            {"metric": "single-step median RMSE", "value": f"{split_summary['single_median_rmse']:.3f}"},
            {"metric": "leave-two-out median RMSE", "value": f"{split_summary['pair_median_rmse']:.3f}"},
            {"metric": "leave-two-out mean RMSE", "value": f"{split_summary['pair_mean_rmse']:.3f}"},
            {"metric": "current pair rank", "value": f"{current_rank} / {len(pair_df)}"},
            {"metric": "balanced pair rank", "value": f"{balanced_rank} / {len(pair_df)}"},
        ]
    )
    lines = [
        "# Tau All-Splits Assessment",
        "",
        "## Scope",
        "- The compact tau law class is unchanged: `[1, V, V_drive_minus_V]`.",
        "- Tau rollout is semi-observed: `tau(t)` is integrated while observed `V(t)` and `V_drive(t)` are supplied.",
        "- This report broadens evaluation across all usable RSFit-aligned p5838 steps rather than relying on one holdout pair.",
        "",
        "## Global summary",
        markdown_table(summary_table),
        "",
        f"- Easiest steps overall: {easiest_steps}.",
        f"- Hardest steps overall: {hardest_steps}.",
        f"- Best leave-two-out pairs: {best_pairs}.",
        f"- Worst leave-two-out pairs: {worst_pairs}.",
        f"- A representative single-step holdout by median difficulty is `{representative_step}`.",
        "",
        "## Step2 vs Step5",
        f"- `step2` is {'easier' if step2_vs_step5['step2']['global_mean_holdout_rmse'] < step2_vs_step5['step5']['global_mean_holdout_rmse'] else 'not easier'} overall: mean holdout RMSE `{step2_vs_step5['step2']['global_mean_holdout_rmse']:.3f}` vs `{step2_vs_step5['step5']['global_mean_holdout_rmse']:.3f}` for `step5`.",
        f"- `step2` has lower motion roughness after preparation: total-variation-per-second `{step2_vs_step5['step2']['tau_total_variation_per_s']:.3f}` vs `{step2_vs_step5['step5']['tau_total_variation_per_s']:.3f}`, and `dtau` std `{step2_vs_step5['step2']['dtau_std']:.3f}` vs `{step2_vs_step5['step5']['dtau_std']:.3f}`.",
        f"- `step5` has stronger forcing/context variation: `V_drive - V` std `{step2_vs_step5['step5']['V_drive_minus_V_std']:.3f}` vs `{step2_vs_step5['step2']['V_drive_minus_V_std']:.3f}`.",
        f"- `step2` is not easy because it has large amplitude; it is easier because its prepared trace is smoother and more monotone. Smoothing ratios: total variation prepared/raw `{step2_vs_step5['step2']['tau_total_variation_ratio_prepared_over_raw']:.3f}` for `step2` and `{step2_vs_step5['step5']['tau_total_variation_ratio_prepared_over_raw']:.3f}` for `step5`.",
        f"- `step5` fits worse visually mainly through shape mismatch and stronger curvature, not simply shorter duration. Peak-time-fraction error: `{step2_vs_step5['step2']['shape_features']['peak_time_fraction_error']:.3f}` for `step2` vs `{step2_vs_step5['step5']['shape_features']['peak_time_fraction_error']:.3f}` for `step5`.",
        "",
        "## Holdout fairness",
        f"- The original stress-test pair `step2 + step7` ranks `{current_rank}` out of `{len(pair_df)}` by leave-two-out mean rollout RMSE, so it remains unusually harsh.",
        f"- The balanced example `step2 + step5` ranks `{balanced_rank}` out of `{len(pair_df)}`. It is useful as a mixed-regime example, but it is {'optimistic rather than median-representative' if representativeness['balanced_pair_is_optimistic'] else 'close to the median and reasonably representative'} for overall pair difficulty.",
        f"- Median leave-two-out RMSE is `{representativeness['median_pair_rmse']:.3f}`; the original pair is `{representativeness['current_pair_rmse']:.3f}` and the balanced pair is `{representativeness['balanced_pair_rmse']:.3f}`.",
        "",
        "## Presentation guidance",
        "- Keep the original `step2 + step7` result as a low-motion stress-test reference.",
        "- Present the all-single-step and all-leave-two-out summaries as the main evidence for broad tau-law behavior.",
        "- Use one representative mixed-regime holdout example, but do not let one easy pair stand in for the whole story.",
        "- The strongest honest framing is that the compact tau law works broadly on many prepared steps, but performance is regime-dependent and degrades most on harsher multi-step low-motion or mismatch combinations.",
        "",
        "## Selected Leave-Three-Out Checks",
        markdown_table(selected_three_df[["holdout_steps", "mean_tau_rollout_rmse", "mean_derivative_rmse"]]),
        "",
        "## Step Difficulty Table",
        markdown_table(step_df[["difficulty_rank", "step_name", "mean_holdout_tau_rmse", "median_holdout_tau_rmse", "difficulty_label"]]),
        "",
        "## Pair Difficulty Table",
        markdown_table(pair_df[["pair_rank", "pair_name", "mean_pair_tau_rollout_rmse", "mean_pair_derivative_rmse"]].head(10)),
    ]
    return "\n".join(lines) + "\n"


def make_finalv3(report_text: str) -> None:
    ensure_directory(FINALV3_DIR)
    ensure_directory(FINALV3_FIG_DIR)
    for path in [FIG_SINGLE_RANK, FIG_PAIR_HEATMAP, FIG_STEP2_STEP5, FIG_COEFFS, FIG_DISTRIBUTION]:
        target = FINALV3_FIG_DIR / path.name
        target.write_bytes(path.read_bytes())
    (FINALV3_DIR / "tau_all_splits_assessment.md").write_text(report_text, encoding="utf-8")
    readme = "\n".join(
        [
            "# Finalv3",
            "",
            "This package adds an all-splits view of the compact tau law across the eight usable p5838 steps.",
            "",
            "Interpretation update:",
            "The compact tau equation still looks strongest overall, but the broad holdout sweep shows that performance varies by step and holdout composition. The original `step2 + step7` pair remains a harsh stress test, while mixed-regime examples such as `step2 + step5` are informative but somewhat optimistic relative to the leave-two-out median.",
            "",
            "Start with:",
            "- [tau_all_splits_assessment.md](./tau_all_splits_assessment.md)",
            "- [Figures/tau_single_step_holdout_ranking.png](./Figures/tau_single_step_holdout_ranking.png)",
            "- [Figures/tau_leave_two_out_heatmap.png](./Figures/tau_leave_two_out_heatmap.png)",
        ]
    )
    (FINALV3_DIR / "README.md").write_text(readme + "\n", encoding="utf-8")


def main() -> None:
    prepared_map = balanced_eval.load_prepared_map()
    step_diag = load_json(STEP_DIAG_JSON)
    split_diag = load_json(SPLIT_JSON)
    variability_df = pd.DataFrame(step_diag["variability_table"]).copy()

    single_rows = evaluate_holdout_family("single_step", [(step,) for step in ALL_STEPS], prepared_map)
    pair_rows = evaluate_holdout_family("leave_two_out", list(combinations(ALL_STEPS, 2)), prepared_map)
    triple_rows = evaluate_holdout_family("selected_leave_three_out", SELECTED_LEAVE_THREE_OUT, prepared_map)
    all_rows = single_rows + pair_rows + triple_rows

    step_df = summarize_step_difficulty(single_rows + pair_rows, variability_df)
    pair_df = summarize_pair_difficulty(pair_rows)
    representativeness = classify_pair_representativeness(pair_df)
    pair_df["distance_to_pair_median_rmse"] = np.abs(pair_df["mean_pair_tau_rollout_rmse"] - representativeness["median_pair_rmse"])

    all_table = split_rows_table(all_rows)
    selected_three_df = pd.DataFrame(
        [
            {
                "holdout_steps": " + ".join(row["holdout_steps"]),
                "mean_tau_rollout_rmse": row["mean_tau_rollout_rmse"],
                "mean_derivative_rmse": row["mean_derivative_rmse"],
            }
            for row in triple_rows
        ]
    )
    step2_vs_step5 = build_step2_vs_step5_explanation(step_df, variability_df, single_rows, prepared_map)

    plot_single_step_ranking(step_df)
    plot_pair_heatmap(pair_df)
    plot_step2_vs_step5(single_rows, prepared_map)
    plot_coefficients_across_splits(all_rows)
    plot_split_distribution(all_rows)
    plot_easy_hard_examples(pair_df, pair_rows, prepared_map)

    split_summary = {
        "single_mean_rmse": float(np.mean([row["mean_tau_rollout_rmse"] for row in single_rows])),
        "single_median_rmse": float(np.median([row["mean_tau_rollout_rmse"] for row in single_rows])),
        "pair_mean_rmse": float(np.mean([row["mean_tau_rollout_rmse"] for row in pair_rows])),
        "pair_median_rmse": float(np.median([row["mean_tau_rollout_rmse"] for row in pair_rows])),
        "pair_best": pair_df.iloc[0]["pair_name"],
        "pair_worst": pair_df.iloc[-1]["pair_name"],
        "current_pair_rank": int(representativeness["current_pair_rank"]),
        "balanced_pair_rank": int(representativeness["balanced_pair_rank"]),
    }

    report_text = build_report(step_df, pair_df, representativeness, step2_vs_step5, split_summary, selected_three_df)

    OUTPUT_TABLE.write_text(all_table.to_csv(index=False), encoding="utf-8")
    STEP_TABLE.write_text(step_df.to_csv(index=False), encoding="utf-8")
    PAIR_TABLE.write_text(pair_df.to_csv(index=False), encoding="utf-8")
    OUTPUT_MD.write_text(report_text, encoding="utf-8")

    summary_payload = {
        "scope": {
            "all_steps": ALL_STEPS,
            "single_step_holdouts": len(single_rows),
            "leave_two_out_pairs": len(pair_rows),
            "selected_leave_three_out": [list(row) for row in SELECTED_LEAVE_THREE_OUT],
        },
        "split_summary": split_summary,
        "step_difficulty": step_df,
        "pair_difficulty": pair_df,
        "selected_leave_three_out_rows": triple_rows,
        "step2_vs_step5": step2_vs_step5,
        "representativeness": representativeness,
        "stress_test_references": {
            "original_pair": list(CURRENT_PAIR),
            "balanced_pair": list(BALANCED_PAIR),
            "current_pair_rollout_rmse_from_saved_diag": split_diag["leave_two_out_summary"]["current_pair_rollout_rmse"],
            "saved_best_representative_pair": split_diag["leave_two_out_summary"]["best_representative_pair"],
        },
        "generated_files": {
            "report_md": str(OUTPUT_MD),
            "report_json": str(OUTPUT_JSON),
            "all_table": str(OUTPUT_TABLE),
            "step_table": str(STEP_TABLE),
            "pair_table": str(PAIR_TABLE),
            "figures": [
                str(FIG_SINGLE_RANK),
                str(FIG_PAIR_HEATMAP),
                str(FIG_STEP2_STEP5),
                str(FIG_COEFFS),
                str(FIG_DISTRIBUTION),
                str(FIG_EASY_HARD),
            ],
        },
    }
    write_json(OUTPUT_JSON, summary_payload)

    finalv3_ready = (
        math.isfinite(representativeness["median_pair_rmse"])
        and len(pair_df) == math.comb(len(ALL_STEPS), 2)
        and FIG_SINGLE_RANK.exists()
        and FIG_PAIR_HEATMAP.exists()
    )
    if finalv3_ready:
        make_finalv3(report_text)

    print(
        json.dumps(
            json_ready(
                {
                    "step_easiest": step_df.iloc[0]["step_name"],
                    "step_hardest": step_df.iloc[-1]["step_name"],
                    "current_pair_rank": representativeness["current_pair_rank"],
                    "balanced_pair_rank": representativeness["balanced_pair_rank"],
                    "median_pair_rmse": representativeness["median_pair_rmse"],
                    "finalv3_created": finalv3_ready,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
