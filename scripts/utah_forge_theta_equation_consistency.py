from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_reviewer_ablation as reviewer_ablation
from src.derivatives import derivative_savgol


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
EPS = 1e-12


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


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in frame.astype(object).fillna("").to_numpy().tolist():
        body.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join([header, divider, *body])


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


def prepare_theta_valid_segments(
    segments: dict[str, pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    rsfit_globals: dict,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    config = reviewer_ablation.MODEL_B_CONFIG
    train_prepared = reviewer_ablation.prepare_model_segments(
        segments,
        steps,
        rsfit_globals,
        reviewer_ablation.TRAIN_STEPS,
        "C",
        config["smoothing"],
        config["memory_window"],
        config["ema_span"],
        None,
    )
    holdout_prepared = reviewer_ablation.prepare_model_segments(
        segments,
        steps,
        rsfit_globals,
        reviewer_ablation.HOLDOUT_STEPS,
        "C",
        config["smoothing"],
        config["memory_window"],
        config["ema_span"],
        None,
    )
    return train_prepared, holdout_prepared


def add_theta_context(prepared_segments: list[pd.DataFrame], steps: dict[str, delay_ref.RSFitStep]) -> list[pd.DataFrame]:
    enriched: list[pd.DataFrame] = []
    for prepared_df in prepared_segments:
        working = prepared_df.copy()
        theta = np.clip(working["theta_approx"].to_numpy(dtype=float), 1e-10, None)
        dtheta_dt = derivative_savgol(theta, t=working["time"].to_numpy(dtype=float), window=15, polyorder=3)
        step_name = str(working["step_name"].iloc[0])
        params = reviewer_ablation.effective_step_params(steps[step_name])
        working["theta"] = theta
        working["dtheta_dt"] = dtheta_dt
        working["Vtheta"] = working["V"].to_numpy(dtype=float) * theta
        working["Dc_ref"] = float(params["Dc"])
        enriched.append(working)
    return enriched


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    yt = y_true[mask]
    yp = y_pred[mask]
    mse = float(np.mean((yp - yt) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yp - yt)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum((yp - yt) ** 2) / (denom + 1e-12))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def fit_linear_law(train_segments: list[pd.DataFrame], holdout_segments: list[pd.DataFrame], feature_names: list[str], label: str) -> dict:
    train_df = pd.concat(train_segments, ignore_index=True)
    holdout_df = pd.concat(holdout_segments, ignore_index=True)
    x_train = np.column_stack(
        [
            np.ones(len(train_df), dtype=float) if name == "1" else train_df[name].to_numpy(dtype=float)
            for name in feature_names
        ]
    )
    y_train = train_df["dtheta_dt"].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    coeff_map = {name: float(value) for name, value in zip(feature_names, coefficients)}
    yhat_train = x_train @ coefficients
    x_holdout = np.column_stack(
        [
            np.ones(len(holdout_df), dtype=float) if name == "1" else holdout_df[name].to_numpy(dtype=float)
            for name in feature_names
        ]
    )
    y_holdout = holdout_df["dtheta_dt"].to_numpy(dtype=float)
    yhat_holdout = x_holdout @ coefficients
    c1 = coeff_map.get("Vtheta", 0.0)
    dc_hat = float(-1.0 / c1) if c1 < 0 else float("nan")
    active_terms = [name for name in feature_names if abs(coeff_map.get(name, 0.0)) > 1e-12]
    return {
        "label": label,
        "feature_names": feature_names,
        "coefficients": coeff_map,
        "active_terms": active_terms,
        "train_metrics": metric_bundle(y_train, yhat_train),
        "holdout_metrics": metric_bundle(y_holdout, yhat_holdout),
        "c0_close_to_1_abs_error": float(abs(coeff_map.get("1", 0.0) - 1.0)),
        "c1_negative": bool(c1 < 0.0),
        "Dc_hat": dc_hat,
        "equation": format_equation("dtheta/dt", coeff_map, feature_names),
    }


def format_equation(lhs: str, coefficient_map: dict[str, float], feature_names: list[str]) -> str:
    pieces: list[str] = []
    for index, name in enumerate(feature_names):
        coefficient = float(coefficient_map.get(name, 0.0))
        if abs(coefficient) <= 1e-14:
            continue
        if name == "1":
            pieces.append(f"{coefficient:.6e}")
        else:
            sign = "+" if coefficient >= 0 else "-"
            pieces.append(f"{sign} {abs(coefficient):.6e}*{name}")
    if not pieces:
        return f"{lhs} = 0"
    if pieces[0].startswith("+ "):
        pieces[0] = pieces[0][2:]
    return f"{lhs} = " + " ".join(pieces)


def subset_summary(train_segments: list[pd.DataFrame], holdout_segments: list[pd.DataFrame]) -> dict:
    rows = []
    for split_name, segment_list in (("train", train_segments), ("holdout", holdout_segments)):
        for segment_df in segment_list:
            step_name = str(segment_df["step_name"].iloc[0])
            theta = np.clip(segment_df["theta"].to_numpy(dtype=float), 1e-12, None)
            rows.append(
                {
                    "split": split_name,
                    "step_name": step_name,
                    "n_rows": int(len(segment_df)),
                    "theta_cv": float(np.std(theta) / (np.mean(theta) + EPS)),
                    "logtheta_std": float(np.std(np.log(theta))),
                    "Dc_ref": float(segment_df["Dc_ref"].iloc[0]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["split", "step_name"]).reset_index(drop=True)
    return {
        "rows": frame.to_dict(orient="records"),
        "mean_theta_cv": float(frame["theta_cv"].mean()) if not frame.empty else float("nan"),
        "mean_logtheta_std": float(frame["logtheta_std"].mean()) if not frame.empty else float("nan"),
        "median_Dc_ref": float(frame["Dc_ref"].median()) if not frame.empty else float("nan"),
    }


def write_outputs(payload: dict) -> None:
    table_df = pd.DataFrame(payload["table_rows"])
    table_df.to_csv(RESULTS_DIR / "theta_equation_consistency_table.csv", index=False)
    (RESULTS_DIR / "theta_equation_consistency.json").write_text(
        json.dumps(json_ready(payload), indent=2),
        encoding="utf-8",
    )
    equations_text = "\n".join(
        [
            "Tiny-library theta equation",
            payload["tiny_library"]["equation"],
            "",
            "Expanded-library theta equation",
            payload["expanded_library"]["equation"],
            "",
        ]
    )
    (RESULTS_DIR / "theta_equation_consistency_equations.txt").write_text(equations_text, encoding="utf-8")

    md_lines = [
        "# Theta Equation Consistency Report",
        "",
        "## Scope",
        "- This is a conditional consistency check, not independent hidden-state discovery.",
        "- `theta(t)` is supplied from externally reconstructed RSFit theta on the theta-valid Model C subset.",
        "- The question is whether that supplied `theta(t)` is itself reasonably consistent with the RSF state law.",
        "",
        "## Usable subset",
        f"- Train steps: `{', '.join(payload['usable_subset']['train_steps'])}`",
        f"- Holdout steps: `{', '.join(payload['usable_subset']['holdout_steps'])}`",
        f"- Mean theta coefficient of variation: `{payload['subset_summary']['mean_theta_cv']:.6e}`",
        f"- Mean std of log(theta): `{payload['subset_summary']['mean_logtheta_std']:.6e}`",
        f"- Median reference Dc across usable events: `{payload['subset_summary']['median_Dc_ref']:.6e}`",
        "",
        "## Tiny-library fit",
        f"- Equation: `{payload['tiny_library']['equation']}`",
        f"- Implied Dc: `{payload['tiny_library']['Dc_hat']:.6e}`",
        f"- `c0` absolute error from 1: `{payload['tiny_library']['c0_close_to_1_abs_error']:.6e}`",
        f"- `c1` negative: `{payload['tiny_library']['c1_negative']}`",
        "",
        "## Expanded-library ablation",
        f"- Equation: `{payload['expanded_library']['equation']}`",
        f"- Implied Dc from `Vtheta` term: `{payload['expanded_library']['Dc_hat']:.6e}`",
        f"- `c0` absolute error from 1: `{payload['expanded_library']['c0_close_to_1_abs_error']:.6e}`",
        f"- `c1` negative: `{payload['expanded_library']['c1_negative']}`",
        "",
        "## Comparison table",
        markdown_table(table_df),
        "",
        "## Interpretation",
        "- If the tiny library already works well, that supports internal self-consistency of RSFit theta with the RSF state law.",
        "- If the expanded library is required to fit well, then the supplied theta trajectory is only weakly consistent with the expected state dynamics after alignment/filtering.",
        f"- Expanded library changed the conclusion materially: `{payload['conclusions']['expanded_library_changes_conclusion']}`",
        f"- Overall consistency-check strength: `{payload['conclusions']['consistency_strength']}`",
        "",
        "## Presentation Q/A paragraph",
        "This test does not discover theta from raw data; it only asks whether externally reconstructed RSFit theta behaves the way the third RSF equation says it should. We fit `dtheta/dt` using the minimal physical form `1 - (V*theta)/Dc` and compared it against a slightly expanded nuisance-term ablation. If the tiny law already matches well, that supports RSFit theta as a self-consistent state signal. If it needs the expanded library or gives an implausible `Dc`, then the theta reconstruction is only weakly consistent with the expected state dynamics after alignment and filtering.",
        "",
    ]
    (RESULTS_DIR / "theta_equation_consistency_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    ensure_layout()
    print("[theta-consistency] loading theta-valid Model C subset", flush=True)
    segments, steps, rsfit_globals = load_segments_without_writing()
    train_prepared, holdout_prepared = prepare_theta_valid_segments(segments, steps, rsfit_globals)
    train_segments = add_theta_context(train_prepared, steps)
    holdout_segments = add_theta_context(holdout_prepared, steps)

    tiny_library = fit_linear_law(train_segments, holdout_segments, ["1", "Vtheta"], "tiny_library")
    expanded_library = fit_linear_law(train_segments, holdout_segments, ["1", "Vtheta", "V", "theta"], "expanded_library")
    subset = subset_summary(train_segments, holdout_segments)

    tiny_improvement = expanded_library["holdout_metrics"]["rmse"] < 0.95 * tiny_library["holdout_metrics"]["rmse"]
    payload = {
        "tiny_library": tiny_library,
        "expanded_library": expanded_library,
        "subset_summary": subset,
        "usable_subset": {
            "train_steps": [str(frame["step_name"].iloc[0]) for frame in train_segments],
            "holdout_steps": [str(frame["step_name"].iloc[0]) for frame in holdout_segments],
        },
        "table_rows": [
            {
                "variant": "tiny_library",
                "equation": tiny_library["equation"],
                "active_terms": "|".join(tiny_library["active_terms"]),
                "train_rmse": tiny_library["train_metrics"]["rmse"],
                "holdout_rmse": tiny_library["holdout_metrics"]["rmse"],
                "holdout_r2": tiny_library["holdout_metrics"]["r2"],
                "c0": tiny_library["coefficients"].get("1", 0.0),
                "c1_Vtheta": tiny_library["coefficients"].get("Vtheta", 0.0),
                "Dc_hat": tiny_library["Dc_hat"],
            },
            {
                "variant": "expanded_library",
                "equation": expanded_library["equation"],
                "active_terms": "|".join(expanded_library["active_terms"]),
                "train_rmse": expanded_library["train_metrics"]["rmse"],
                "holdout_rmse": expanded_library["holdout_metrics"]["rmse"],
                "holdout_r2": expanded_library["holdout_metrics"]["r2"],
                "c0": expanded_library["coefficients"].get("1", 0.0),
                "c1_Vtheta": expanded_library["coefficients"].get("Vtheta", 0.0),
                "Dc_hat": expanded_library["Dc_hat"],
            },
        ],
        "conclusions": {
            "expanded_library_changes_conclusion": bool(tiny_improvement),
            "tiny_library_reasonably_consistent": bool(
                tiny_library["c1_negative"]
                and tiny_library["c0_close_to_1_abs_error"] < 0.5
                and tiny_library["holdout_metrics"]["r2"] > 0.25
            ),
            "consistency_strength": (
                "good"
                if tiny_library["c1_negative"] and tiny_library["c0_close_to_1_abs_error"] < 0.5 and tiny_library["holdout_metrics"]["r2"] > 0.25
                else "weak"
            ),
        },
    }
    write_outputs(payload)
    print("[theta-consistency] wrote consistency artifacts to results/utah_forge", flush=True)


if __name__ == "__main__":
    main()
