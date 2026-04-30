from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exact_rsf import (
    fit_exact_rsf_inverse_model,
    json_ready,
    load_checkpoint,
    load_workflow_context,
    pack_initial_vector,
    prepare_exact_segments,
    save_checkpoint,
    split_segments,
)
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(REPO_ROOT / "results" / "utah_forge")
CHECKPOINT_DIR = ensure_directory(RESULTS_DIR / "exact_rsf_inverse_fit_checkpoints")
MULTISTART_DIR = ensure_directory(RESULTS_DIR / "exact_rsf_multistart_checkpoints")
MAX_NFEV = 300
N_STARTS = 4
SEED = 798


def make_starts(base_initial: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    starts = [base_initial.copy()]
    for _ in range(N_STARTS - 1):
        trial = base_initial.copy()
        trial[0] *= float(rng.uniform(0.5, 1.5))   # k
        trial[1] *= float(rng.uniform(0.5, 2.0))   # m
        trial[2] *= float(rng.uniform(0.85, 1.15))  # mu0
        trial[3] *= float(rng.uniform(0.5, 1.5))   # a
        trial[4] *= float(rng.uniform(0.5, 1.5))   # b
        trial[5] *= float(rng.uniform(0.5, 1.5))   # Dc
        if len(trial) > 6:
            trial[6:] += rng.normal(0.0, 0.2, size=len(trial) - 6)
        starts.append(trial)
    return starts


def parameter_stability(rows: list[dict]) -> dict:
    param_names = list(rows[0]["parameters"].keys())
    out = {}
    for name in param_names:
        values = [float(row["parameters"][name]) for row in rows]
        mean = float(np.mean(values))
        std = float(np.std(values))
        cv = float(std / (abs(mean) + 1e-12))
        out[name] = {"mean": mean, "std": std, "cv": cv, "values": values}
    return out


def summarize_run(index: int, payload: dict) -> dict:
    holdout_df = pd.DataFrame(payload["holdout_rows"])
    ident = payload["identifiability"]
    return {
        "start_index": index,
        "success": bool(payload["optimization"]["success"]),
        "status": int(payload["optimization"]["status"]),
        "message": str(payload["optimization"]["message"]),
        "nfev": int(payload["optimization"]["nfev"]),
        "cost": float(payload["optimization"]["cost"]),
        "optimality": float(payload["optimization"]["optimality"]),
        "parameters": payload["parameters"],
        "theta0": payload["per_event_theta0"],
        "mean_holdout_error": float(holdout_df["combined_rollout_error"].mean()),
        "mean_holdout_peak_timing_error_s": float(holdout_df["peak_timing_error_s"].mean()),
        "mean_holdout_onset_timing_error_s": float(holdout_df["onset_timing_error_s"].mean()),
        "mean_holdout_stable_fraction": float(holdout_df["stable_fraction"].mean()),
        "sigma_too_constant": bool(ident["sigma_too_constant_for_mu_a_b_separation"]),
        "parameter_confounding_flag": bool(ident["parameter_confounding_flag"]),
        "jtj_condition_number": float(ident["jtj_condition_number"]),
        "jtj_rank": int(ident["jtj_rank"]),
        "theta_offset_sensitivity_mean": float(np.mean(list(ident["theta_offset_sensitivity"].values()))),
        "theta_term_meaningfully_active": bool(float(payload["parameters"]["b"]) > 1e-3),
    }


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
    train_segments = prepared["train_segments"]
    holdout_segments = prepared["holdout_segments"]

    base_initial, _, _ = pack_initial_vector(train_segments, use_acoustic=False)
    starts = make_starts(base_initial)

    rows = []
    payloads = []
    for index, start in enumerate(starts):
        stage_name = f"exact_fit_multistart_{index}"
        payload = fit_exact_rsf_inverse_model(
            train_segments,
            holdout_segments,
            use_acoustic=False,
            checkpoint_dir=MULTISTART_DIR,
            stage_name=stage_name,
            max_nfev=MAX_NFEV,
            initial_vector=start,
        )
        payloads.append(payload)
        rows.append(summarize_run(index, payload))

    best_row = min(rows, key=lambda row: row["cost"])
    best_payload = payloads[best_row["start_index"]]
    baseline_payload = load_checkpoint(CHECKPOINT_DIR, "exact_fit_base")
    if baseline_payload is None:
        raise RuntimeError("Missing baseline exact_fit_base checkpoint.")

    stability = parameter_stability(rows)
    baseline_ident = baseline_payload["identifiability"]
    best_ident = best_payload["identifiability"]

    any_success = any(row["success"] for row in rows)
    theta_active = any(row["theta_term_meaningfully_active"] for row in rows)
    cond_improved = bool(best_ident["jtj_condition_number"] < baseline_ident["jtj_condition_number"])
    rank_improved = bool(best_ident["jtj_rank"] > baseline_ident["jtj_rank"])
    estimates_stable = all(stats["cv"] < 0.25 for name, stats in stability.items() if name in {"k", "m", "mu0", "a", "b", "Dc"})
    conclusion_changed = bool(
        any_success
        and not best_ident["parameter_confounding_flag"]
        and not best_ident["sigma_too_constant_for_mu_a_b_separation"]
        and theta_active
    )

    summary = {
        "settings": {"max_nfev": MAX_NFEV, "n_starts": N_STARTS, "seed": SEED},
        "runs": rows,
        "best_run": best_row,
        "parameter_stability": stability,
        "checks": {
            "fit_converges_more_cleanly": any_success or best_row["nfev"] < baseline_payload["optimization"]["nfev"],
            "theta_term_becomes_meaningfully_active": theta_active,
            "parameter_estimates_stabilize_across_starts": estimates_stable,
            "identifiability_metrics_improve": bool(cond_improved or rank_improved),
            "final_scientific_conclusion_changes": conclusion_changed,
        },
        "baseline": {
            "success": bool(baseline_payload["optimization"]["success"]),
            "nfev": int(baseline_payload["optimization"]["nfev"]),
            "cost": float(baseline_payload["optimization"]["cost"]),
            "jtj_condition_number": float(baseline_ident["jtj_condition_number"]),
            "jtj_rank": int(baseline_ident["jtj_rank"]),
        },
        "best_identifiability": {
            "jtj_condition_number": float(best_ident["jtj_condition_number"]),
            "jtj_rank": int(best_ident["jtj_rank"]),
            "sigma_too_constant": bool(best_ident["sigma_too_constant_for_mu_a_b_separation"]),
            "parameter_confounding_flag": bool(best_ident["parameter_confounding_flag"]),
        },
        "final_statement": (
            "Equation (2) remains non-identifiable even after direct latent-state inverse fitting with bounded multistart optimization."
            if not conclusion_changed
            else "The bounded multistart check changed the scientific conclusion."
        ),
    }

    summary_path = RESULTS_DIR / "exact_rsf_multistart_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

    lines = [
        "# Exact RSF Multistart Check",
        "",
        f"- Settings: `max_nfev={MAX_NFEV}`, `n_starts={N_STARTS}`",
        f"- Best run success: `{best_row['success']}`",
        f"- Best run message: `{best_row['message']}`",
        f"- Best run cost: `{best_row['cost']:.6e}`",
        f"- Best run mean holdout rollout error: `{best_row['mean_holdout_error']:.6e}`",
        "",
        "## Requested checks",
        f"- Fit converges more cleanly: `{summary['checks']['fit_converges_more_cleanly']}`",
        f"- Theta term becomes meaningfully active: `{summary['checks']['theta_term_becomes_meaningfully_active']}`",
        f"- Parameter estimates stabilize across starts: `{summary['checks']['parameter_estimates_stabilize_across_starts']}`",
        f"- Identifiability metrics improve: `{summary['checks']['identifiability_metrics_improve']}`",
        f"- Final scientific conclusion changes: `{summary['checks']['final_scientific_conclusion_changes']}`",
        "",
        "## Baseline vs best multistart",
        f"- Baseline success / nfev / cost: `{summary['baseline']['success']}` / `{summary['baseline']['nfev']}` / `{summary['baseline']['cost']:.6e}`",
        f"- Best success / nfev / cost: `{best_row['success']}` / `{best_row['nfev']}` / `{best_row['cost']:.6e}`",
        f"- Baseline JTJ condition number: `{summary['baseline']['jtj_condition_number']:.6e}`",
        f"- Best JTJ condition number: `{summary['best_identifiability']['jtj_condition_number']:.6e}`",
        f"- Baseline JTJ rank: `{summary['baseline']['jtj_rank']}`",
        f"- Best JTJ rank: `{summary['best_identifiability']['jtj_rank']}`",
        f"- SigmaN too constant in best run: `{summary['best_identifiability']['sigma_too_constant']}`",
        f"- Parameter confounding in best run: `{summary['best_identifiability']['parameter_confounding_flag']}`",
        "",
        "## Final statement",
        f"- `{summary['final_statement']}`",
    ]
    (RESULTS_DIR / "exact_rsf_multistart_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["checks"], indent=2), flush=True)


if __name__ == "__main__":
    main()
