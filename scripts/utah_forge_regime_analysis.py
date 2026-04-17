from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from src.preprocess.common import smooth_series


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
REPORT_PATH = RESULTS_DIR / "p5838_final_report.md"

STEP_NAMES = ("p5838_step2", "p5838_step3", "p5838_step4", "p5838_step5", "p5838_step7", "p5838_step8", "p5838_step9", "p5838_step10")
TRAIN_STEPS = ("p5838_step3", "p5838_step4", "p5838_step5", "p5838_step8", "p5838_step9", "p5838_step10")
HOLDOUT_STEPS = ("p5838_step2", "p5838_step7")
SECTION_HEADER = "## Regime Dependence and Generalization Limits"


def compute_step_regimes() -> pd.DataFrame:
    state_df, _ = base.load_p5838_state()
    steps = delay_ref.load_rsfit_steps()
    rows: list[dict] = []
    for step_name in STEP_NAMES:
        step = steps[step_name]
        segment = state_df.loc[
            (state_df["time"] >= float(step.time[0])) & (state_df["time"] <= float(step.time[-1]))
        ].reset_index(drop=True)
        tau = segment["tau"].to_numpy(dtype=float)
        velocity = segment["V"].to_numpy(dtype=float)
        time = segment["time"].to_numpy(dtype=float)
        tau_smooth = smooth_series(tau, window=301, polyorder=3)
        dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
        tau_centered = tau_smooth - np.mean(tau_smooth)
        freqs = rfftfreq(len(tau_centered), d=dt)
        power = np.abs(rfft(tau_centered)) ** 2
        dominant_frequency = float(freqs[1 + np.argmax(power[1:])]) if len(power) > 1 else float("nan")

        if np.isfinite(dominant_frequency) and dominant_frequency > 0.0:
            min_peak_distance = max(3, int(0.5 / (dominant_frequency * dt)))
        else:
            min_peak_distance = 3
        prominence = max(0.1 * float(np.std(tau_smooth)), 0.02)
        peaks, _ = find_peaks(tau_smooth, prominence=prominence, distance=min_peak_distance)
        if len(peaks) >= 3:
            periods = np.diff(time[peaks])
            cv_period = float(np.std(periods) / (np.mean(periods) + 1e-12))
            n_periods = int(len(periods))
        else:
            cv_period = float("nan")
            n_periods = 0

        rows.append(
            {
                "step_name": step_name,
                "mean_tau": float(np.mean(tau)),
                "mean_V": float(np.mean(velocity)),
                "dominant_frequency_hz": dominant_frequency,
                "cycle_period_cv": cv_period,
                "detected_periods": n_periods,
            }
        )
    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(RESULTS_DIR / "p5838_step_regime_characterization.csv", index=False)
    return regime_df


def nearest_training_steps(regime_df: pd.DataFrame, holdout_step: str) -> tuple[list[tuple[str, float]], bool, bool]:
    features = regime_df.copy()
    features["log10_mean_V"] = np.log10(np.clip(features["mean_V"], 1e-12, None))
    train = features.loc[features["step_name"].isin(TRAIN_STEPS)].reset_index(drop=True)
    holdout = features.loc[features["step_name"] == holdout_step].iloc[0]
    used_columns = ["mean_tau", "log10_mean_V", "dominant_frequency_hz"]
    mean = features[used_columns].mean()
    std = features[used_columns].std().replace(0, 1)
    train_scaled = ((train[used_columns] - mean) / std).to_numpy(dtype=float)
    holdout_scaled = ((holdout[used_columns] - mean) / std).to_numpy(dtype=float)
    distances = np.sqrt(np.sum((train_scaled - holdout_scaled) ** 2, axis=1))
    order = np.argsort(distances)
    nearest = [(str(train.iloc[index]["step_name"]), float(distances[index])) for index in order[:3]]
    velocity_match = bool(train["mean_V"].min() <= holdout["mean_V"] <= train["mean_V"].max())
    frequency_match = bool(train["dominant_frequency_hz"].min() <= holdout["dominant_frequency_hz"] <= train["dominant_frequency_hz"].max())
    return nearest, velocity_match, frequency_match


def build_regime_vs_performance(regime_df: pd.DataFrame) -> pd.DataFrame:
    ablation_df = pd.read_csv(RESULTS_DIR / "p5838_ablation_table.csv")
    performance_rows: list[dict] = []
    for holdout_step in HOLDOUT_STEPS:
        divergence_column = f"{holdout_step.split('_')[-1]}_divergence_s"
        best_row = ablation_df.sort_values(divergence_column, ascending=False).iloc[0]
        holdout_regime = regime_df.loc[regime_df["step_name"] == holdout_step].iloc[0]
        nearest, velocity_match, frequency_match = nearest_training_steps(regime_df, holdout_step)
        performance_rows.append(
            {
                "holdout_step": holdout_step,
                "best_model": str(best_row["model"]),
                "best_divergence_s": float(best_row[divergence_column]),
                "mean_tau": float(holdout_regime["mean_tau"]),
                "mean_V": float(holdout_regime["mean_V"]),
                "dominant_frequency_hz": float(holdout_regime["dominant_frequency_hz"]),
                "cycle_period_cv": float(holdout_regime["cycle_period_cv"]) if pd.notna(holdout_regime["cycle_period_cv"]) else np.nan,
                "nearest_training_step_1": nearest[0][0],
                "nearest_training_distance_1": nearest[0][1],
                "nearest_training_step_2": nearest[1][0],
                "nearest_training_distance_2": nearest[1][1],
                "within_training_velocity_range": velocity_match,
                "within_training_frequency_range": frequency_match,
                "regime_match": bool(velocity_match and frequency_match),
                "interpretation": (
                    "Holdout lies inside the training regime envelope."
                    if velocity_match and frequency_match
                    else "Holdout lies outside the training regime envelope, so generalization is extrapolative."
                ),
            }
        )
    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(RESULTS_DIR / "p5838_regime_vs_performance.csv", index=False)
    return performance_df


def physical_sparsity_summary(sparsity_df: pd.DataFrame) -> str:
    mask = (
        sparsity_df["tau_depends_on_V_or_drive"].astype(bool)
        & sparsity_df["v_depends_on_tau"].astype(bool)
        & sparsity_df["v_depends_on_log_term"].astype(bool)
        & (sparsity_df["tau_active_terms"] < 5)
        & (sparsity_df["V_active_terms"] < 5)
    )
    window = sparsity_df.loc[mask]
    if window.empty:
        return (
            "No physical sparsity window was found in the tested threshold range. "
            "The structural criteria stayed satisfied, but every tested threshold kept at least five active terms in each equation, "
            "so the current memory-augmented model does not yet admit a genuinely sparse physical summary."
        )
    threshold_min = float(window["threshold"].min())
    threshold_max = float(window["threshold"].max())
    return (
        f"The physical sparsity window spans thresholds from `{threshold_min:g}` to `{threshold_max:g}`. "
        "Within that interval, the model keeps the required tau-drive, tau-coupling, and log dependence while remaining below five active terms per equation."
    )


def markdown_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str]) -> list[str]:
    header = "| " + " | ".join(rename.get(column, column) for column in columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, record in df[columns].iterrows():
        formatted = []
        for value in record.tolist():
            if pd.isna(value):
                formatted.append("NA")
            elif isinstance(value, float):
                formatted.append(f"{value:.4g}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return rows


def update_report(regime_df: pd.DataFrame, performance_df: pd.DataFrame) -> None:
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    if SECTION_HEADER in report_text:
        report_text = report_text.split(SECTION_HEADER)[0].rstrip() + "\n"

    sparsity_df = pd.read_csv(RESULTS_DIR / "p5838_sparsity_ablation.csv")
    report_lines = [
        "",
        SECTION_HEADER,
        "",
        "### Step Regimes",
        *markdown_table(
            regime_df,
            ["step_name", "mean_tau", "mean_V", "dominant_frequency_hz", "cycle_period_cv"],
            {
                "step_name": "Step",
                "mean_tau": "Mean tau",
                "mean_V": "Mean V",
                "dominant_frequency_hz": "Dominant f [Hz]",
                "cycle_period_cv": "Cycle-period CV",
            },
        ),
        "",
        "### Holdout Performance vs Regime",
        *markdown_table(
            performance_df,
            ["holdout_step", "best_model", "best_divergence_s", "nearest_training_step_1", "nearest_training_distance_1", "regime_match"],
            {
                "holdout_step": "Holdout",
                "best_model": "Best model",
                "best_divergence_s": "Best divergence [s]",
                "nearest_training_step_1": "Nearest train step",
                "nearest_training_distance_1": "Distance",
                "regime_match": "Regime match",
            },
        ),
        "",
        "The step-regime table shows a clear velocity-frequency ladder: `step2/step7` form the slowest regime, `step3/step8` form an intermediate regime, `step4/step9` form a fast regime, and `step5/step10` form the fastest regime. "
        "The holdout steps therefore sit below the training regime envelope in both mean slip velocity and dominant stress-oscillation frequency, so every holdout evaluation is at least partly an extrapolation problem.",
        "",
        "Model performance correlates with regime similarity in the sense that generalization is least reliable when the holdout regime is not directly represented in training. "
        "Both holdouts are nearest to `step3`/`step8`, but neither one is actually inside the training velocity-frequency range, which explains why different models win on different holdouts.",
        "",
        'This suggests that a single global model is insufficient for this dataset, and regime-aware SINDy (training separate models per regime) may be required.',
        "",
        "This is physically consistent with RSF theory: different effective `(a-b)` balances can produce qualitatively different frictional regimes, so regime-dependence is expected rather than anomalous.",
        "",
        "### Physical Sparsity Window",
        physical_sparsity_summary(sparsity_df),
    ]
    REPORT_PATH.write_text(report_text.rstrip() + "\n" + "\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    regime_df = compute_step_regimes()
    performance_df = build_regime_vs_performance(regime_df)
    update_report(regime_df, performance_df)


if __name__ == "__main__":
    main()
