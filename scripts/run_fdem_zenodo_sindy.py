from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FDEM_ZENODO_CONFIG
from src.datasets.fdem_zenodo import build_fdem_zenodo_summary
from src.io.load_fdem_zenodo import (
    FDEM_BINARY_SHAPE,
    FDEM_NSS_COLUMN,
    FDEM_SENSOR_BLOCK_WIDTH,
    FDEM_SENSOR_VALUE_COLUMNS,
    FDEM_TIME_COLUMN,
    inspect_fdem_helper_scripts,
    load_fdem_zenodo_dataset,
    locate_fdem_binary,
    locate_fdem_zenodo_files,
)
from src.sindy.models import SINDyModel
from src.utils.paths import ensure_directory


RESULTS_DIR = ensure_directory(FDEM_ZENODO_CONFIG.results_dir)
RAW_DIAGNOSTICS_PATH = RESULTS_DIR / "raw_array_diagnostics.txt"
VARIABLE_DIAGNOSTICS_PATH = RESULTS_DIR / "variable_diagnostics.json"
CYCLE_DIAGNOSTICS_PATH = RESULTS_DIR / "cycle_diagnostics.csv"
ROLLOUT_PLOT = RESULTS_DIR / "baseline_rollout.png"
THRESHOLD = 1e-4
MAX_ITER = 12
MIN_CYCLE_LENGTH = 500
N_CYCLES_TO_USE = 3
EPS = 1e-12


@dataclass(frozen=True)
class SelectedCycle:
    cycle_id: str
    start_idx: int
    end_idx: int
    duration_s: float
    normalized_drop: float
    drop_amplitude: float
    cycle_df: pd.DataFrame


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_inventory() -> None:
    inventory = locate_fdem_zenodo_files()
    payload = {
        "dataset": FDEM_ZENODO_CONFIG.name,
        "source_url": FDEM_ZENODO_CONFIG.source_url,
        "files": {
            key: [
                {
                    "name": path.name,
                    "path": str(path),
                    "suffix": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                }
                for path in value
            ]
            for key, value in inventory.items()
        },
    }
    write_json(RESULTS_DIR / "raw_file_inventory.json", payload)


def save_raw_array_diagnostics() -> None:
    helper_summary = inspect_fdem_helper_scripts()
    binary_path = locate_fdem_binary()
    raw = np.fromfile(binary_path, dtype=np.float64).reshape(FDEM_BINARY_SHAPE)
    feature_block = raw[:, :FDEM_SENSOR_VALUE_COLUMNS]
    time_col = raw[:, FDEM_TIME_COLUMN]
    nss_col = raw[:, FDEM_NSS_COLUMN]
    lines = [
        "# FDEM raw array diagnostics",
        "",
        f"Raw file: {binary_path}",
        f"Shape: {raw.shape}",
        f"Helper summary: {json.dumps(helper_summary, indent=2)}",
        "",
        "First 5 rows of the raw array:",
        np.array2string(raw[:5], threshold=10_000_000, max_line_width=1000),
        "",
        "Column stats:",
        f"- all sensor features mean/std/min/max: {feature_block.mean():.6e}, {feature_block.std():.6e}, {feature_block.min():.6e}, {feature_block.max():.6e}",
        f"- time mean/std/min/max: {time_col.mean():.6e}, {time_col.std():.6e}, {time_col.min():.6e}, {time_col.max():.6e}",
        f"- nss mean/std/min/max: {nss_col.mean():.6e}, {nss_col.std():.6e}, {nss_col.min():.6e}, {nss_col.max():.6e}",
    ]
    RAW_DIAGNOSTICS_PATH.write_text("\n".join(lines), encoding="utf-8")


def segment_by_nss_drops(df: pd.DataFrame, min_cycle_length: int = MIN_CYCLE_LENGTH) -> tuple[list[tuple[int, int]], float]:
    mu = df["mu"].to_numpy(dtype=float)
    diffs = np.diff(mu)
    major_drop_threshold = min(-0.02, float(np.quantile(diffs, 0.001)))
    reset_points = np.where(diffs < major_drop_threshold)[0] + 1
    boundaries = [0, *reset_points.tolist(), len(df)]
    segments: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start >= min_cycle_length:
            segments.append((int(start), int(end)))
    return segments, major_drop_threshold


def write_variable_diagnostics(df: pd.DataFrame, segments: list[tuple[int, int]], threshold: float) -> dict:
    dt = np.diff(df["time"].to_numpy(dtype=float))
    mu_std = float(df["mu"].std()) or 1.0
    ek_std = float(df["Ek"].std()) or 1.0
    diagnostics = {
        "raw": {
            "mu": {
                "mean": float(df["mu"].mean()),
                "std": mu_std,
                "min": float(df["mu"].min()),
                "max": float(df["mu"].max()),
                "physical_units": "published normalized shear stress / friction-like target from the final binary column",
            },
            "Ek": {
                "mean": float(df["Ek"].mean()),
                "std": ek_std,
                "min": float(df["Ek"].min()),
                "max": float(df["Ek"].max()),
                "physical_units": "computed kinetic energy = 0.5 * sum(vx^2 + vy^2)",
            },
            "std_ratio_mu_over_Ek": mu_std / ek_std,
            "dt_median_seconds": float(np.median(dt[dt > 0])) if np.any(dt > 0) else None,
            "n_cycles_found": int(len(segments)),
            "nss_drop_threshold": float(threshold),
            "five_longest_cycle_lengths": [int(value) for value in sorted((end - start for start, end in segments), reverse=True)[:5]],
        }
    }
    return diagnostics


def zscore(values: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    return (values - mean) / std, {"mean": mean, "std": std}


def choose_cycles(df: pd.DataFrame, segments: list[tuple[int, int]]) -> tuple[list[SelectedCycle], pd.DataFrame]:
    rows: list[dict] = []
    candidates: list[SelectedCycle] = []
    for index, (start, end) in enumerate(segments, start=1):
        cycle_df = df.iloc[start:end].reset_index(drop=True).copy()
        mu = cycle_df["mu"].to_numpy(dtype=float)
        amplitude = float(np.max(mu) - np.min(mu))
        mu_std = float(np.std(mu)) or 1.0
        normalized_drop = amplitude / mu_std
        cycle = SelectedCycle(
            cycle_id=f"fdem_cycle_{index:03d}",
            start_idx=start,
            end_idx=end,
            duration_s=float(cycle_df["time"].iloc[-1] - cycle_df["time"].iloc[0]),
            normalized_drop=normalized_drop,
            drop_amplitude=amplitude,
            cycle_df=cycle_df,
        )
        candidates.append(cycle)
        rows.append(
            {
                "cycle_id": cycle.cycle_id,
                "start_idx": start,
                "end_idx": end,
                "n_samples": int(len(cycle_df)),
                "duration_s": cycle.duration_s,
                "mu_drop_amplitude": amplitude,
                "mu_std": mu_std,
                "normalized_mu_drop": normalized_drop,
                "Ek_range": float(np.max(cycle_df["Ek"]) - np.min(cycle_df["Ek"])),
            }
        )

    selected = sorted(candidates, key=lambda item: (item.normalized_drop, item.drop_amplitude, len(item.cycle_df)), reverse=True)[:N_CYCLES_TO_USE]
    diag_df = pd.DataFrame(rows)
    if not diag_df.empty:
        diag_df["selected"] = diag_df["cycle_id"].isin({cycle.cycle_id for cycle in selected})
        diag_df = diag_df.sort_values(["selected", "normalized_mu_drop"], ascending=[False, False]).reset_index(drop=True)
    diag_df.to_csv(CYCLE_DIAGNOSTICS_PATH, index=False)

    for export_index, cycle in enumerate(selected, start=1):
        export_df = cycle.cycle_df.copy()
        export_df.insert(0, "cycle_id", cycle.cycle_id)
        export_df.to_csv(RESULTS_DIR / f"selected_cycle_{export_index:03d}.csv", index=False)
    return selected, diag_df


def prepare_cycles(selected_cycles: list[SelectedCycle]) -> tuple[list[pd.DataFrame], dict]:
    frames = [cycle.cycle_df.assign(cycle_id=cycle.cycle_id).copy() for cycle in selected_cycles]
    combined = pd.concat(frames, ignore_index=True)
    combined["logEk_raw"] = np.log(np.clip(combined["Ek"], EPS, None))

    _, mu_stats = zscore(combined["mu"])
    _, ek_stats = zscore(combined["Ek"])
    _, logek_stats = zscore(combined["logEk_raw"])
    scaling = {"mu": mu_stats, "Ek": ek_stats, "logEk_raw": logek_stats}

    normalized_frames: list[pd.DataFrame] = []
    for frame in frames:
        working = frame.copy()
        working["mu_z"] = (working["mu"] - mu_stats["mean"]) / mu_stats["std"]
        working["Ek_z"] = (working["Ek"] - ek_stats["mean"]) / ek_stats["std"]
        working["logEk_z"] = (np.log(np.clip(working["Ek"], EPS, None)) - logek_stats["mean"]) / logek_stats["std"]
        normalized_frames.append(working.reset_index(drop=True))
    return normalized_frames, scaling


def build_library_and_targets(normalized_frames: list[pd.DataFrame]) -> tuple[np.ndarray, list[str], np.ndarray]:
    terms = ["1", "mu_z", "Ek_z", "mu_z^2", "Ek_z^2", "mu_z*Ek_z", "logEk_z", "mu_z*logEk_z"]
    libs: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    for frame in normalized_frames:
        mu_z = frame["mu_z"].to_numpy(dtype=float)
        ek_z = frame["Ek_z"].to_numpy(dtype=float)
        logek_z = frame["logEk_z"].to_numpy(dtype=float)
        libs.append(
            np.column_stack(
                [
                    np.ones(len(frame)),
                    mu_z,
                    ek_z,
                    mu_z**2,
                    ek_z**2,
                    mu_z * ek_z,
                    logek_z,
                    mu_z * logek_z,
                ]
            )
        )
        time = frame["time"].to_numpy(dtype=float)
        tgts.append(
            np.column_stack(
                [
                    CubicSpline(time, mu_z)(time, 1),
                    CubicSpline(time, ek_z)(time, 1),
                ]
            )
        )
    return np.vstack(libs), terms, np.vstack(tgts)


def backtransform_equation(coeff_vector: np.ndarray, target_std: float, scaling: dict) -> tuple[str, dict[str, float]]:
    coeffs = {name: float(value) for name, value in zip(["1", "mu_z", "Ek_z", "mu_z^2", "Ek_z^2", "mu_z*Ek_z", "logEk_z", "mu_z*logEk_z"], coeff_vector)}
    mu0 = scaling["mu"]["mean"]
    smu = scaling["mu"]["std"]
    e0 = scaling["Ek"]["mean"]
    se = scaling["Ek"]["std"]
    l0 = scaling["logEk_raw"]["mean"]
    sl = scaling["logEk_raw"]["std"]

    constant = (
        coeffs["1"]
        - coeffs["mu_z"] * mu0 / smu
        - coeffs["Ek_z"] * e0 / se
        + coeffs["mu_z^2"] * mu0**2 / smu**2
        + coeffs["Ek_z^2"] * e0**2 / se**2
        + coeffs["mu_z*Ek_z"] * mu0 * e0 / (smu * se)
        - coeffs["logEk_z"] * l0 / sl
        + coeffs["mu_z*logEk_z"] * mu0 * l0 / (smu * sl)
    )
    mu_term = (
        coeffs["mu_z"] / smu
        - 2.0 * coeffs["mu_z^2"] * mu0 / smu**2
        - coeffs["mu_z*Ek_z"] * e0 / (smu * se)
        - coeffs["mu_z*logEk_z"] * l0 / (smu * sl)
    )
    ek_term = coeffs["Ek_z"] / se - 2.0 * coeffs["Ek_z^2"] * e0 / se**2 - coeffs["mu_z*Ek_z"] * mu0 / (smu * se)
    mu2_term = coeffs["mu_z^2"] / smu**2
    ek2_term = coeffs["Ek_z^2"] / se**2
    mu_ek_term = coeffs["mu_z*Ek_z"] / (smu * se)
    logek_term = coeffs["logEk_z"] / sl - coeffs["mu_z*logEk_z"] * mu0 / (smu * sl)
    mu_logek_term = coeffs["mu_z*logEk_z"] / (smu * sl)

    physical = {
        "1": target_std * constant,
        "mu": target_std * mu_term,
        "Ek": target_std * ek_term,
        "mu^2": target_std * mu2_term,
        "Ek^2": target_std * ek2_term,
        "mu*Ek": target_std * mu_ek_term,
        "log(Ek+eps)": target_std * logek_term,
        "mu*log(Ek+eps)": target_std * mu_logek_term,
    }
    pieces = []
    for name, value in physical.items():
        if abs(value) <= 0:
            continue
        sign = "+" if value >= 0 else "-"
        piece = f"{sign} {abs(value):.3e}*{name}"
        pieces.append(piece if pieces else piece.lstrip("+ ").strip())
    return " ".join(pieces), physical


def rollout_cycle(frame: pd.DataFrame, coefficients: np.ndarray, scaling: dict) -> tuple[np.ndarray, dict]:
    time = frame["time"].to_numpy(dtype=float)
    truth = frame[["mu_z", "Ek_z"]].to_numpy(dtype=float)
    pred = np.zeros_like(truth)
    pred[0] = truth[0]
    dt_all = np.diff(time)
    fallback_dt = float(np.median(dt_all[dt_all > 0])) if np.any(dt_all > 0) else 1.0
    divergence_time = float(time[-1] - time[0])

    for idx in range(len(time) - 1):
        dt = float(time[idx + 1] - time[idx])
        dt = dt if dt > 0 else fallback_dt
        mu_raw = scaling["mu"]["mean"] + scaling["mu"]["std"] * float(pred[idx, 0])
        ek_raw = scaling["Ek"]["mean"] + scaling["Ek"]["std"] * float(pred[idx, 1])
        logek_z = (math.log(max(ek_raw, EPS)) - scaling["logEk_raw"]["mean"]) / scaling["logEk_raw"]["std"]
        phi = np.array(
            [
                1.0,
                pred[idx, 0],
                pred[idx, 1],
                pred[idx, 0] ** 2,
                pred[idx, 1] ** 2,
                pred[idx, 0] * pred[idx, 1],
                logek_z,
                pred[idx, 0] * logek_z,
            ],
            dtype=float,
        )
        derivative = phi @ coefficients
        pred[idx + 1] = pred[idx] + dt * derivative
        point_error = max(abs(pred[idx + 1, 0] - truth[idx + 1, 0]), abs(pred[idx + 1, 1] - truth[idx + 1, 1]))
        if point_error > 0.1 and divergence_time == float(time[-1] - time[0]):
            divergence_time = float(time[idx + 1] - time[0])

    residual = truth - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    relative_error = float(np.linalg.norm(residual) / (np.linalg.norm(truth) + 1e-16))
    if not np.isfinite(rmse):
        rmse = float("inf")
    if not np.isfinite(mae):
        mae = float("inf")
    if not np.isfinite(relative_error):
        relative_error = float("inf")
    return pred, {
        "rmse": rmse,
        "mae": mae,
        "relative_error": relative_error,
        "divergence_time_s": divergence_time,
    }


def plot_rollout(frame: pd.DataFrame, pred: np.ndarray) -> None:
    time = frame["time"].to_numpy(dtype=float)
    truth = frame[["mu_z", "Ek_z"]].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time, truth[:, 0], label="mu_z true", linewidth=1.0)
    axes[0].plot(time, pred[:, 0], label="mu_z rollout", linewidth=1.0)
    axes[0].set_ylabel("mu_z")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(time, truth[:, 1], label="Ek_z true", linewidth=1.0)
    axes[1].plot(time, pred[:, 1], label="Ek_z rollout", linewidth=1.0)
    axes[1].set_ylabel("Ek_z")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(ROLLOUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def baseline_interpretation(divergence_time: float) -> str:
    if divergence_time < 5.0:
        return "proxy-state model, weak physical interpretation"
    if divergence_time > 10.0:
        return "reasonable proxy baseline"
    return "proxy-state model, moderate conditioning but limited physical interpretation"


def write_equations(normalized_eqs: list[str], physical_eqs: list[str]) -> None:
    lines = ["# Normalized-domain equations", *normalized_eqs, "", "# Physical-unit equations", *physical_eqs, ""]
    (RESULTS_DIR / "discovered_equations.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_directory(RESULTS_DIR)
    write_inventory()
    save_raw_array_diagnostics()

    dataset_summary = build_fdem_zenodo_summary()
    write_json(RESULTS_DIR / "dataset_summary.json", dataset_summary)

    df, load_summary = load_fdem_zenodo_dataset()
    if df is None:
        print(json.dumps({"success": False, "reason": load_summary.get("suitability_note", "binary missing")}, indent=2))
        return

    segments, threshold = segment_by_nss_drops(df, min_cycle_length=MIN_CYCLE_LENGTH)
    diagnostics = write_variable_diagnostics(df, segments, threshold)
    if not segments:
        diagnostics["normalized"] = None
        write_json(VARIABLE_DIAGNOSTICS_PATH, diagnostics)
        print(json.dumps({"success": False, "reason": "No cycles with at least 500 timesteps were found."}, indent=2))
        return

    selected_cycles, cycle_diag = choose_cycles(df, segments)
    normalized_frames, scaling = prepare_cycles(selected_cycles)
    diagnostics["normalized"] = {
        "mu_z": {"mean": 0.0, "std": 1.0},
        "Ek_z": {"mean": 0.0, "std": 1.0},
        "std_ratio_mu_over_Ek": 1.0,
        "normalization_constants": {
            "mu_mean": scaling["mu"]["mean"],
            "mu_std": scaling["mu"]["std"],
            "Ek_mean": scaling["Ek"]["mean"],
            "Ek_std": scaling["Ek"]["std"],
        },
    }
    write_json(VARIABLE_DIAGNOSTICS_PATH, diagnostics)

    library, terms, targets = build_library_and_targets(normalized_frames)
    model = SINDyModel(threshold=THRESHOLD, max_iter=MAX_ITER)
    fit_summary = model.fit(library, targets, terms)
    normalized_eqs = model.equations(["mu_z", "Ek_z"])

    mu_eq_phys, mu_coeffs_phys = backtransform_equation(model.coefficients[:, 0], scaling["mu"]["std"], scaling)
    ek_eq_phys, ek_coeffs_phys = backtransform_equation(model.coefficients[:, 1], scaling["Ek"]["std"], scaling)
    physical_eqs = [f"dmu/dt = {mu_eq_phys}", f"dEk/dt = {ek_eq_phys}"]
    write_equations(normalized_eqs, physical_eqs)

    rollout_rows: list[dict] = []
    first_pred = None
    first_frame = None
    for frame in normalized_frames:
        pred, metrics = rollout_cycle(frame, model.coefficients, scaling)
        rollout_rows.append({"cycle_id": str(frame["cycle_id"].iloc[0]), **metrics})
        if first_pred is None:
            first_pred = pred
            first_frame = frame
    if first_pred is not None and first_frame is not None:
        plot_rollout(first_frame, first_pred)

    rollout_df = pd.DataFrame(rollout_rows)
    mean_rollout = {
        "rmse": float(rollout_df["rmse"].mean()),
        "mae": float(rollout_df["mae"].mean()),
        "relative_error": float(rollout_df["relative_error"].mean()),
        "divergence_time_s": float(rollout_df["divergence_time_s"].mean()),
    }

    baseline_summary = {
        "dataset": FDEM_ZENODO_CONFIG.name,
        "state_variables": ["mu", "Ek"],
        "state_columns": ["mu", "Ek"],
        "state_labels": ["mu_z", "Ek_z"],
        "scaling": {
            "means": {"mu": scaling["mu"]["mean"], "Ek": scaling["Ek"]["mean"]},
            "stds": {"mu": scaling["mu"]["std"], "Ek": scaling["Ek"]["std"]},
        },
        "poly_degree": 2,
        "threshold": THRESHOLD,
        "max_iter": MAX_ITER,
        "library_terms": terms,
        "equations": normalized_eqs,
        "physical_equations": physical_eqs,
        "relative_error": [float(value) for value in fit_summary["residuals"]],
        "rmse_like": [float(value) for value in fit_summary["residuals"]],
        "rollout_metrics": mean_rollout,
        "rollout_metrics_by_cycle": rollout_rows,
        "rollout_plot": str(ROLLOUT_PLOT),
        "mu_mean": scaling["mu"]["mean"],
        "mu_std": scaling["mu"]["std"],
        "Ek_mean": scaling["Ek"]["mean"],
        "Ek_std": scaling["Ek"]["std"],
        "n_cycles_used": int(len(selected_cycles)),
        "normalization_applied": True,
        "selected_cycle_ids": [cycle.cycle_id for cycle in selected_cycles],
        "selected_cycle_paths": [str(RESULTS_DIR / f"selected_cycle_{index:03d}.csv") for index in range(1, len(selected_cycles) + 1)],
        "segmentation": {
            "strategy": "major_nss_drop",
            "n_cycles_detected": int(len(segments)),
            "min_cycle_length": MIN_CYCLE_LENGTH,
            "drop_threshold": float(threshold),
        },
        "proxy_notes": load_summary.get("proxy_notes"),
        "physical_interpretation": baseline_interpretation(mean_rollout["divergence_time_s"]),
        "physical_coefficients": {"dmu_dt": mu_coeffs_phys, "dEk_dt": ek_coeffs_phys},
    }
    write_json(RESULTS_DIR / "baseline_summary.json", baseline_summary)

    comparison_lines = [
        "# FDEM Zenodo SINDy vs LightGBM comparison",
        "",
        "## Dataset role",
        "- `fdem_zenodo` is a simulated granular-fault comparison dataset rather than a physical lab dataset.",
        "- The helper scripts use the final binary column as the pointwise friction-like prediction target; this SINDy baseline now models that same `mu` signal dynamically.",
        "",
        "## SINDy baseline",
        f"- State variables: `mu`, `Ek`",
        f"- Cycles used: `{', '.join(baseline_summary['selected_cycle_ids'])}`",
        f"- Mean rollout divergence time: `{mean_rollout['divergence_time_s']:.6f}` s",
        f"- Mean rollout RMSE: `{mean_rollout['rmse']:.6f}`",
        f"- Interpretation: `{baseline_summary['physical_interpretation']}`",
        "",
        "## Published LightGBM benchmark",
        "- Huang et al. reported pointwise prediction metrics on the same FDEM target, including `R^2 = 0.94` for the final optimized/statistics model and `RMSE = 0.0045` for an optimized-feature testing case.",
        "- Source: https://www.mdpi.com/2077-1312/12/2/246 and https://zenodo.org/records/7370626",
        "",
        "## Comparison note",
        "- LightGBM's reported performance is a pointwise regression metric on `mu`, while SINDy's divergence time is a dynamical rollout metric.",
        "- The comparison is therefore more direct than the earlier proxy-state run, but it still mixes regression and rollout criteria rather than one shared score.",
    ]
    (RESULTS_DIR / "sindy_vs_lgbm_comparison.md").write_text("\n".join(comparison_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "success": True,
                "normalized_equations": normalized_eqs,
                "physical_equations": physical_eqs,
                "rollout_metrics": mean_rollout,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
