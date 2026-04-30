from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
PREPARED_TAU_CHECKPOINT = RESULTS_DIR / "proposal_equation_checkpoints" / "prepared_segments.pkl"

PARTIAL_JSON = RESULTS_DIR / "v_reduced_partial_results.json"
PARTIAL_CSV = RESULTS_DIR / "v_reduced_partial_table.csv"
SUMMARY_MD = RESULTS_DIR / "v_reduced_summary.md"
SUMMARY_JSON = RESULTS_DIR / "v_reduced_summary.json"
STEP_TABLE = RESULTS_DIR / "v_step_difficulty_table.csv"
PAIR_TABLE = RESULTS_DIR / "v_pair_difficulty_table.csv"

FIG_SINGLE = RESULTS_DIR / "v_single_step_holdout_ranking_reduced.png"
FIG_PAIR = RESULTS_DIR / "v_leave_two_out_heatmap_reduced.png"
FIG_DIST = RESULTS_DIR / "v_split_distribution_summary_reduced.png"
FIG_EXAMPLES = RESULTS_DIR / "v_easy_vs_hard_examples_reduced.png"
FIG_CONTEXT = RESULTS_DIR / "v_step_context_comparison_reduced.png"

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


def load_prepared_map() -> dict[str, pd.DataFrame]:
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
        "coefficients_physical": coefficients,
        "coefficients_z": coefficients_z.tolist(),
        "active_terms": active_terms,
        "train_mse": train_mse,
        "equation": proposal.format_equation("dV/dt", coefficients, model_def["ordered_terms"]),
        "deltaS_orth_map": {},
    }


def rollout_arrays(model_row: dict, seg: pd.DataFrame) -> dict:
    series = showcase.rollout_velocity_series(model_row, seg)
    time = series["time"]
    observed_v = series["observed_v"]
    predicted_v = series["predicted_v"]
    abs_error = np.abs(predicted_v - observed_v)
    sigma_obs = float(np.std(observed_v))
    threshold = 3.0 * max(sigma_obs, 1e-6)
    divergence_index = len(observed_v)
    for index, (pred, obs) in enumerate(zip(predicted_v, observed_v)):
        if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
            divergence_index = index
            break
    stable_fraction = float(divergence_index / len(observed_v)) if len(observed_v) else 0.0
    derivative_err = series["predicted_dv"] - series["observed_dv"]
    return {
        "time": time,
        "rel_time": series["rel_time"],
        "observed_v": observed_v,
        "predicted_v": predicted_v,
        "observed_tau": series["observed_tau"],
        "observed_sigma": seg["sigmaN"].to_numpy(dtype=float),
        "observed_gap": (seg["V_drive"] - seg["V"]).to_numpy(dtype=float),
        "abs_v_error": abs_error,
        "velocity_rollout_rmse": float(np.sqrt(np.mean((predicted_v - observed_v) ** 2))),
        "velocity_rollout_mae": float(np.mean(abs_error)),
        "velocity_max_abs_error": float(np.max(abs_error)),
        "derivative_rmse": float(np.sqrt(np.mean(derivative_err**2))),
        "onset_timing_error_s": abs(proposal.onset_time(predicted_v, time) - proposal.onset_time(observed_v, time)),
        "peak_timing_error_s": abs(proposal.peak_time(predicted_v, time) - proposal.peak_time(observed_v, time)),
        "stable_fraction": stable_fraction,
    }


def load_partial_state() -> dict:
    if not PARTIAL_JSON.exists():
        return {"completed_splits": {}, "split_rows": [], "meta": {"branch": "reduced_rsf"}}
    return json.loads(PARTIAL_JSON.read_text(encoding="utf-8"))


def save_partial_state(state: dict) -> None:
    write_json(PARTIAL_JSON, state)
    pd.DataFrame(state["split_rows"]).to_csv(PARTIAL_CSV, index=False)


def split_plan() -> list[dict]:
    rows = []
    for holdout in [(step,) for step in ALL_STEPS]:
        rows.append({"family": "single_step", "holdout_steps": list(holdout)})
    for holdout in combinations(ALL_STEPS, 2):
        rows.append({"family": "leave_two_out", "holdout_steps": list(holdout)})
    return rows


def split_id(family: str, holdout_steps: list[str]) -> str:
    return f"{family}__{'__'.join(holdout_steps)}"


def evaluate_split(prepared_map: dict[str, pd.DataFrame], family: str, holdout_steps: list[str]) -> dict:
    train_steps = [step for step in ALL_STEPS if step not in holdout_steps]
    train_segments = [prepared_map[step].copy() for step in train_steps]
    model = fit_reduced_fixed_threshold(train_segments)
    holdout_rows = []
    for step in holdout_steps:
        arr = rollout_arrays(model, prepared_map[step])
        holdout_rows.append(
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
        "family": family,
        "train_steps": train_steps,
        "holdout_steps": holdout_steps,
        "equation": model["equation"],
        "coefficients_physical": model["coefficients_physical"],
        "holdout_rows": holdout_rows,
        "mean_velocity_rollout_rmse": float(np.mean([row["velocity_rollout_rmse"] for row in holdout_rows])),
        "mean_velocity_rollout_mae": float(np.mean([row["velocity_rollout_mae"] for row in holdout_rows])),
        "mean_velocity_max_abs_error": float(np.mean([row["velocity_max_abs_error"] for row in holdout_rows])),
        "mean_derivative_rmse": float(np.mean([row["derivative_rmse"] for row in holdout_rows])),
        "mean_onset_timing_error_s": float(np.mean([row["onset_timing_error_s"] for row in holdout_rows])),
        "mean_peak_timing_error_s": float(np.mean([row["peak_timing_error_s"] for row in holdout_rows])),
        "mean_stable_fraction": float(np.mean([row["stable_fraction"] for row in holdout_rows])),
    }


def summarize_step_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for step in ALL_STEPS:
        sub = frame.loc[frame["step_name"] == step].copy()
        timing = 0.5 * (sub["onset_timing_error_s"] + sub["peak_timing_error_s"])
        records.append(
            {
                "step_name": step,
                "n_holdout_appearances": int(len(sub)),
                "mean_holdout_velocity_rmse": float(sub["velocity_rollout_rmse"].mean()),
                "median_holdout_velocity_rmse": float(sub["velocity_rollout_rmse"].median()),
                "std_holdout_velocity_rmse": float(sub["velocity_rollout_rmse"].std(ddof=0)),
                "mean_timing_error_s": float(timing.mean()),
                "mean_stable_fraction": float(sub["stable_fraction"].mean()),
            }
        )
    out = pd.DataFrame(records).sort_values("mean_holdout_velocity_rmse").reset_index(drop=True)
    q1 = float(out["mean_holdout_velocity_rmse"].quantile(1 / 3))
    q2 = float(out["mean_holdout_velocity_rmse"].quantile(2 / 3))
    labels = []
    for value in out["mean_holdout_velocity_rmse"]:
        if value <= q1:
            labels.append("easy")
        elif value <= q2:
            labels.append("medium")
        else:
            labels.append("hard")
    out["difficulty_label"] = labels
    out["difficulty_rank"] = np.arange(1, len(out) + 1)
    return out


def summarize_pair_difficulty(state: dict) -> pd.DataFrame:
    rows = []
    for split_key, split in state["completed_splits"].items():
        if split["family"] != "leave_two_out":
            continue
        rows.append(
            {
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
    out = pd.DataFrame(rows).sort_values("mean_pair_velocity_rollout_rmse").reset_index(drop=True)
    if not out.empty:
        out["pair_rank"] = np.arange(1, len(out) + 1)
    return out


def load_visual_arrays(prepared_map: dict[str, pd.DataFrame], state: dict, holdout_steps: list[str]) -> list[dict]:
    split = state["completed_splits"][split_id("leave_two_out" if len(holdout_steps) == 2 else "single_step", holdout_steps)]
    model = {
        **reduced_model_def(),
        "coefficients_physical": split["coefficients_physical"],
        "deltaS_orth_map": {},
    }
    return [rollout_arrays(model, prepared_map[step]) for step in holdout_steps]


def plot_single_step_ranking(step_df: pd.DataFrame) -> None:
    frame = step_df.sort_values("mean_holdout_velocity_rmse")
    colors = ["tab:green" if x == "easy" else "tab:orange" if x == "medium" else "tab:red" for x in frame["difficulty_label"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame["step_name"], frame["mean_holdout_velocity_rmse"], yerr=frame["std_holdout_velocity_rmse"], color=colors, capsize=3, alpha=0.85)
    ax.set_title("Reduced RSF: average holdout velocity RMSE by step")
    ax.set_ylabel("mean holdout velocity rollout RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_SINGLE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_heatmap(pair_df: pd.DataFrame) -> None:
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
    ax.set_title("Reduced RSF leave-two-out velocity rollout RMSE")
    fig.tight_layout()
    fig.savefig(FIG_PAIR, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(state: dict) -> None:
    frame = pd.DataFrame(
        [{"family": split["family"], "rmse": split["mean_velocity_rollout_rmse"]} for split in state["completed_splits"].values()]
    )
    families = list(dict.fromkeys(frame["family"].tolist()))
    bins = np.linspace(float(frame["rmse"].min()), float(frame["rmse"].max()), 16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"single_step": "#457b9d", "leave_two_out": "#e76f51"}
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
    fig.suptitle("Reduced RSF dynamic velocity rollout performance across splits", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIST, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_examples(prepared_map: dict[str, pd.DataFrame], state: dict, pair_df: pd.DataFrame) -> None:
    chosen = pair_df.sort_values("mean_pair_velocity_rollout_rmse").iloc[[0, 1, -2, -1]]
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=False)
    for ridx, (_, row) in enumerate(chosen.iterrows()):
        arrays = load_visual_arrays(prepared_map, state, [row["step_a"], row["step_b"]])
        for cidx, arr in enumerate(arrays):
            ax = axes[ridx, cidx]
            ax.plot(arr["rel_time"], arr["observed_v"], linewidth=1.2, label="observed V")
            ax.plot(arr["rel_time"], arr["predicted_v"], linewidth=1.05, label="predicted V")
            ax.set_title(f"{row['pair_name']}: {row['step_a'] if cidx == 0 else row['step_b']}")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("V")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle("Reduced RSF easy vs hard holdout examples", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_EXAMPLES, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_context(step_df: pd.DataFrame, prepared_map: dict[str, pd.DataFrame]) -> None:
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
    fig.savefig(FIG_CONTEXT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summaries(prepared_map: dict[str, pd.DataFrame], state: dict) -> None:
    if not state["split_rows"]:
        return
    frame = pd.DataFrame(state["split_rows"])
    step_df = summarize_step_difficulty(frame)
    pair_df = summarize_pair_difficulty(state)
    step_df.to_csv(STEP_TABLE, index=False)
    pair_df.to_csv(PAIR_TABLE, index=False)
    plot_single_step_ranking(step_df)
    plot_pair_heatmap(pair_df)
    plot_distribution(state)
    if len(pair_df) >= 4:
        plot_examples(prepared_map, state, pair_df)
    plot_context(step_df, prepared_map)
    current_rank = int(pair_df.loc[pair_df["is_current_pair"], "pair_rank"].iloc[0]) if not pair_df.empty else None
    summary_payload = {
        "completed_split_count": len(state["completed_splits"]),
        "expected_split_count": 36,
        "step_difficulty": step_df,
        "pair_difficulty": pair_df,
        "current_pair_rank": current_rank,
    }
    write_json(SUMMARY_JSON, summary_payload)
    lines = [
        "# Reduced RSF V Summary",
        "",
        f"- Completed splits: `{len(state['completed_splits'])} / 36`",
        f"- Original `step2 + step7` pair rank: `{current_rank}`" if current_rank is not None else "- Original `step2 + step7` pair rank: `not available yet`",
        "",
        "## Step Difficulty",
        markdown_table(step_df[["difficulty_rank", "step_name", "mean_holdout_velocity_rmse", "mean_timing_error_s", "mean_stable_fraction", "difficulty_label"]]),
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute completed splits")
    args = parser.parse_args()

    prepared_map = load_prepared_map()
    state = load_partial_state()
    plan = split_plan()
    started = time.perf_counter()

    for index, item in enumerate(plan, start=1):
        sid = split_id(item["family"], item["holdout_steps"])
        if (not args.force) and sid in state["completed_splits"]:
            continue
        split_started = time.perf_counter()
        result = evaluate_split(prepared_map, item["family"], item["holdout_steps"])
        result["split_id"] = sid
        state["completed_splits"][sid] = result
        for row in result["holdout_rows"]:
            state["split_rows"].append(
                {
                    "split_id": sid,
                    "family": item["family"],
                    "train_steps": "|".join(result["train_steps"]),
                    "holdout_steps": "|".join(result["holdout_steps"]),
                    "equation": result["equation"],
                    "step_name": row["step_name"],
                    "velocity_rollout_rmse": row["velocity_rollout_rmse"],
                    "velocity_rollout_mae": row["velocity_rollout_mae"],
                    "velocity_max_abs_error": row["velocity_max_abs_error"],
                    "derivative_rmse": row["derivative_rmse"],
                    "onset_timing_error_s": row["onset_timing_error_s"],
                    "peak_timing_error_s": row["peak_timing_error_s"],
                    "stable_fraction": row["stable_fraction"],
                    "split_mean_velocity_rollout_rmse": result["mean_velocity_rollout_rmse"],
                }
            )
        save_partial_state(state)
        elapsed = time.perf_counter() - started
        split_elapsed = time.perf_counter() - split_started
        print(
            f"[reduced_v] completed {sid} ({index}/36) split_elapsed_s={split_elapsed:.2f} total_elapsed_s={elapsed:.2f} saved={PARTIAL_JSON.name}",
            flush=True,
        )

    build_summaries(prepared_map, state)
    print(json.dumps({"completed_splits": len(state["completed_splits"]), "partial_json": str(PARTIAL_JSON)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
