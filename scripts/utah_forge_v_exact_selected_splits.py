from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.derivatives import derivative_savgol
from src.exact_rsf import fit_exact_rsf_inverse_model, load_workflow_context, pack_initial_vector, prepare_exact_segments, simulate_exact_rsf_segment
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
CHECKPOINT_DIR = ensure_directory(RESULTS_DIR / "v_exact_selected_checkpoints")

PARTIAL_JSON = RESULTS_DIR / "v_exact_partial_results.json"
PARTIAL_CSV = RESULTS_DIR / "v_exact_partial_table.csv"
SUMMARY_MD = RESULTS_DIR / "v_exact_summary.md"
SUMMARY_JSON = RESULTS_DIR / "v_exact_summary.json"
STEP_TABLE = RESULTS_DIR / "v_exact_step_difficulty_table.csv"
PAIR_TABLE = RESULTS_DIR / "v_exact_pair_difficulty_table.csv"

FIG_SINGLE = RESULTS_DIR / "v_single_step_holdout_ranking_exact.png"
FIG_SELECTED = RESULTS_DIR / "v_selected_pairs_examples_exact.png"

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
SELECTED_PAIRS = [
    ("p5838_step2", "p5838_step5"),
    ("p5838_step3", "p5838_step10"),
    ("p5838_step4", "p5838_step5"),
    ("p5838_step2", "p5838_step10"),
    ("p5838_step5", "p5838_step7"),
    ("p5838_step2", "p5838_step7"),
]
CURRENT_PAIR = ("p5838_step2", "p5838_step7")
MAX_NFEV = 80
N_STARTS = 2
SEED = 798


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


def load_context():
    inventory_df, segments, steps, rsfit_globals = load_workflow_context()
    filtered_segments = {step: segments[step] for step in ALL_STEPS if step in segments}
    return inventory_df, filtered_segments, steps, rsfit_globals


def make_starts(base_initial: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    starts = [base_initial.copy()]
    for _ in range(N_STARTS - 1):
        trial = base_initial.copy()
        trial[0] *= float(rng.uniform(0.5, 1.5))
        trial[1] *= float(rng.uniform(0.5, 2.0))
        trial[2] *= float(rng.uniform(0.85, 1.15))
        trial[3] *= float(rng.uniform(0.5, 1.5))
        trial[4] *= float(rng.uniform(0.5, 1.5))
        trial[5] *= float(rng.uniform(0.5, 1.5))
        if len(trial) > 6:
            trial[6:] += rng.normal(0.0, 0.2, size=len(trial) - 6)
        starts.append(trial)
    return starts


def prepare_split(context, train_steps: list[str], holdout_steps: list[str]):
    _, segments, steps, rsfit_globals = context
    train_raw = [segments[step].copy() for step in train_steps]
    holdout_raw = [segments[step].copy() for step in holdout_steps]
    train_segments, holdout_segments, _ = prepare_exact_segments(train_raw, holdout_raw, steps, rsfit_globals)
    return train_segments, holdout_segments


def load_partial_state() -> dict:
    if not PARTIAL_JSON.exists():
        return {"completed_splits": {}, "split_rows": [], "meta": {"branch": "exact_rsf_selected"}}
    return json.loads(PARTIAL_JSON.read_text(encoding="utf-8"))


def save_partial_state(state: dict) -> None:
    write_json(PARTIAL_JSON, state)
    pd.DataFrame(state["split_rows"]).to_csv(PARTIAL_CSV, index=False)


def split_plan() -> list[dict]:
    rows = [{"family": "single_step", "holdout_steps": [step]} for step in ALL_STEPS]
    rows.extend({"family": "selected_pair", "holdout_steps": list(pair)} for pair in SELECTED_PAIRS)
    return rows


def split_id(family: str, holdout_steps: list[str]) -> str:
    return f"{family}__{'__'.join(holdout_steps)}"


def exact_arrays(segment, params: dict, theta_offsets: dict[str, float], acoustic_zscores: dict[str, float]) -> dict:
    sim = simulate_exact_rsf_segment(
        segment,
        params,
        delta_log_theta0=float(theta_offsets.get(segment.step_name, 0.0)),
        acoustic_z=float(acoustic_zscores.get(segment.step_name, 0.0)),
    )
    pred_v = sim["V"]
    obs_v = segment.V
    abs_err = np.abs(pred_v - obs_v)
    pred_dv = derivative_savgol(pred_v, t=segment.time, window=15, polyorder=3)
    deriv_err = pred_dv - segment.dV_dt
    sigma_obs = float(np.std(obs_v))
    threshold = 3.0 * max(sigma_obs, 1e-6)
    divergence_index = len(obs_v)
    for index, (pred, obs) in enumerate(zip(pred_v, obs_v)):
        if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
            divergence_index = index
            break
    stable_fraction = float(divergence_index / len(obs_v)) if len(obs_v) else 0.0
    return {
        "rel_time": segment.time - float(segment.time[0]),
        "observed_v": obs_v,
        "predicted_v": pred_v,
        "velocity_rollout_rmse": float(np.sqrt(np.nanmean((pred_v - obs_v) ** 2))) if np.isfinite(pred_v).any() else float("inf"),
        "velocity_rollout_mae": float(np.nanmean(abs_err)) if np.isfinite(abs_err).any() else float("inf"),
        "velocity_max_abs_error": float(np.nanmax(abs_err)) if np.isfinite(abs_err).any() else float("inf"),
        "derivative_rmse": float(np.sqrt(np.nanmean(deriv_err**2))) if np.isfinite(deriv_err).any() else float("inf"),
        "onset_timing_error_s": abs(proposal.onset_time(pred_v, segment.time) - proposal.onset_time(obs_v, segment.time)),
        "peak_timing_error_s": abs(proposal.peak_time(pred_v, segment.time) - proposal.peak_time(obs_v, segment.time)),
        "stable_fraction": stable_fraction,
    }


def evaluate_split(context, family: str, holdout_steps: list[str]) -> dict:
    train_steps = [step for step in ALL_STEPS if step not in holdout_steps]
    train_segments, holdout_segments = prepare_split(context, train_steps, holdout_steps)
    base_initial, _, _ = pack_initial_vector(train_segments, use_acoustic=False)
    starts = make_starts(base_initial)
    best_payload = None
    best_key = None
    for index, start in enumerate(starts):
        stage_name = f"v_exact_selected_{family}_{'__'.join(holdout_steps)}_start{index}"
        payload = fit_exact_rsf_inverse_model(
            train_segments,
            holdout_segments,
            use_acoustic=False,
            checkpoint_dir=CHECKPOINT_DIR,
            stage_name=stage_name,
            max_nfev=MAX_NFEV,
            initial_vector=start,
        )
        key = (float(payload["optimization"]["cost"]), float(np.mean([row["V_rmse"] for row in payload["holdout_rows"]])))
        if (best_key is None) or (key < best_key):
            best_payload = payload
            best_key = key
    assert best_payload is not None
    rows = []
    for segment in holdout_segments:
        arr = exact_arrays(segment, best_payload["parameters"], best_payload.get("theta_offsets_train", {}), best_payload["acoustic_zscores"])
        rows.append(
            {
                "step_name": segment.step_name,
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
        "fit_cost": float(best_payload["optimization"]["cost"]),
        "fit_success": bool(best_payload["optimization"]["success"]),
        "holdout_rows": rows,
        "mean_velocity_rollout_rmse": float(np.mean([row["velocity_rollout_rmse"] for row in rows])),
        "mean_velocity_rollout_mae": float(np.mean([row["velocity_rollout_mae"] for row in rows])),
        "mean_velocity_max_abs_error": float(np.mean([row["velocity_max_abs_error"] for row in rows])),
        "mean_derivative_rmse": float(np.mean([row["derivative_rmse"] for row in rows])),
        "mean_onset_timing_error_s": float(np.mean([row["onset_timing_error_s"] for row in rows])),
        "mean_peak_timing_error_s": float(np.mean([row["peak_timing_error_s"] for row in rows])),
        "mean_stable_fraction": float(np.mean([row["stable_fraction"] for row in rows])),
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


def summarize_pairs(state: dict) -> pd.DataFrame:
    rows = []
    for split in state["completed_splits"].values():
        if split["family"] != "selected_pair":
            continue
        rows.append(
            {
                "pair_name": " + ".join(split["holdout_steps"]),
                "step_a": split["holdout_steps"][0],
                "step_b": split["holdout_steps"][1],
                "mean_pair_velocity_rollout_rmse": split["mean_velocity_rollout_rmse"],
                "mean_pair_stable_fraction": split["mean_stable_fraction"],
                "is_current_pair": tuple(split["holdout_steps"]) == CURRENT_PAIR,
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_pair_velocity_rollout_rmse").reset_index(drop=True)
    if not out.empty:
        out["pair_rank"] = np.arange(1, len(out) + 1)
    return out


def build_summaries(state: dict) -> None:
    if not state["split_rows"]:
        return
    frame = pd.DataFrame(state["split_rows"])
    step_df = summarize_step_difficulty(frame)
    pair_df = summarize_pairs(state)
    step_df.to_csv(STEP_TABLE, index=False)
    pair_df.to_csv(PAIR_TABLE, index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(step_df["step_name"], step_df["mean_holdout_velocity_rmse"], color="#6c91bf")
    ax.set_title("Exact RSF: selected-split velocity difficulty")
    ax.set_ylabel("mean holdout velocity RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_SINGLE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not pair_df.empty:
        chosen = pair_df.sort_values("mean_pair_velocity_rollout_rmse").iloc[[0, -1]] if len(pair_df) > 1 else pair_df.iloc[[0]]
        fig, axes = plt.subplots(len(chosen), 2, figsize=(11.5, 4.2 * len(chosen)), sharex=False)
        if len(chosen) == 1:
            axes = np.array([axes])
        for ridx, (_, row) in enumerate(chosen.iterrows()):
            sid = split_id("selected_pair", [row["step_a"], row["step_b"]])
            split = state["completed_splits"][sid]
            for cidx, step in enumerate([row["step_a"], row["step_b"]]):
                step_row = next(item for item in split["holdout_rows"] if item["step_name"] == step)
                axes[ridx, cidx].text(0.5, 0.5, f"{step}\nRMSE={step_row['velocity_rollout_rmse']:.3f}\nStable={step_row['stable_fraction']:.3f}", ha="center", va="center")
                axes[ridx, cidx].set_title(f"{row['pair_name']}: {step}")
                axes[ridx, cidx].set_axis_off()
        fig.suptitle("Exact RSF selected pair summaries", y=0.995)
        fig.tight_layout()
        fig.savefig(FIG_SELECTED, dpi=200, bbox_inches="tight")
        plt.close(fig)

    current_rank = int(pair_df.loc[pair_df["is_current_pair"], "pair_rank"].iloc[0]) if (not pair_df.empty and pair_df["is_current_pair"].any()) else None
    payload = {
        "completed_split_count": len(state["completed_splits"]),
        "expected_split_count": 14,
        "selected_pairs": [list(pair) for pair in SELECTED_PAIRS],
        "step_difficulty": step_df,
        "pair_difficulty": pair_df,
        "current_pair_rank": current_rank,
    }
    write_json(SUMMARY_JSON, payload)
    lines = [
        "# Exact RSF V Summary",
        "",
        "- Exact RSF is a secondary exact-form comparison, not the primary final V model.",
        f"- Completed splits: `{len(state['completed_splits'])} / 14`",
        "- This script evaluates all 8 single-step holdouts plus selected informative leave-two-out pairs only.",
        f"- Original `step2 + step7` selected-pair rank: `{current_rank}`" if current_rank is not None else "- Original `step2 + step7` selected-pair rank: `not available yet`",
        "",
        "## Step Difficulty",
        markdown_table(step_df[["difficulty_rank", "step_name", "mean_holdout_velocity_rmse", "mean_timing_error_s", "mean_stable_fraction", "difficulty_label"]]),
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    context = load_context()
    state = load_partial_state()
    plan = split_plan()
    started = time.perf_counter()
    total = len(plan)

    for index, item in enumerate(plan, start=1):
        sid = split_id(item["family"], item["holdout_steps"])
        if (not args.force) and sid in state["completed_splits"]:
            continue
        split_started = time.perf_counter()
        result = evaluate_split(context, item["family"], item["holdout_steps"])
        result["split_id"] = sid
        state["completed_splits"][sid] = result
        for row in result["holdout_rows"]:
            state["split_rows"].append(
                {
                    "split_id": sid,
                    "family": item["family"],
                    "train_steps": "|".join(result["train_steps"]),
                    "holdout_steps": "|".join(result["holdout_steps"]),
                    "step_name": row["step_name"],
                    "velocity_rollout_rmse": row["velocity_rollout_rmse"],
                    "velocity_rollout_mae": row["velocity_rollout_mae"],
                    "velocity_max_abs_error": row["velocity_max_abs_error"],
                    "derivative_rmse": row["derivative_rmse"],
                    "onset_timing_error_s": row["onset_timing_error_s"],
                    "peak_timing_error_s": row["peak_timing_error_s"],
                    "stable_fraction": row["stable_fraction"],
                    "split_mean_velocity_rollout_rmse": result["mean_velocity_rollout_rmse"],
                    "fit_cost": result["fit_cost"],
                    "fit_success": result["fit_success"],
                }
            )
        save_partial_state(state)
        elapsed = time.perf_counter() - started
        split_elapsed = time.perf_counter() - split_started
        print(
            f"[exact_v] completed {sid} ({index}/{total}) split_elapsed_s={split_elapsed:.2f} total_elapsed_s={elapsed:.2f} saved={PARTIAL_JSON.name}",
            flush=True,
        )

    build_summaries(state)
    print(json.dumps({"completed_splits": len(state["completed_splits"]), "partial_json": str(PARTIAL_JSON)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
