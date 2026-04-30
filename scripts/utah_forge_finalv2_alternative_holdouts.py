from __future__ import annotations

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

from scripts import utah_forge_finalv2_refresh_package as finalv2
from src.utils.paths import ensure_directory


RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
FINALV2_DIR = RESULTS_DIR / "Finalv2"
ALT_DIR = ensure_directory(FINALV2_DIR / "Figures" / "Proposal_Tau" / "Alternative_Holdouts")
PAIR_TABLE = RESULTS_DIR / "tau_pair_difficulty_table.csv"
ALL_SPLITS_TABLE = RESULTS_DIR / "tau_all_splits_table.csv"
STEP_TABLE = RESULTS_DIR / "tau_step_difficulty_table.csv"

PAIR_CHOICES = [
    ("p5838_step2", "p5838_step5", "best_pair"),
    ("p5838_step3", "p5838_step10", "median_pair"),
    ("p5838_step4", "p5838_step5", "median_pair_alt"),
    ("p5838_step2", "p5838_step7", "stress_test"),
    ("p5838_step2", "p5838_step10", "hard_pair"),
]
SINGLE_CHOICES = [
    ("p5838_step3", "easy_single"),
    ("p5838_step5", "medium_single"),
    ("p5838_step2", "hard_single"),
]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(PAIR_TABLE),
        pd.read_csv(ALL_SPLITS_TABLE),
        pd.read_csv(STEP_TABLE),
    )


def parse_holdout_steps(text: str) -> tuple[str, ...]:
    return tuple(text.split("|"))


def get_split_coeffs(all_splits: pd.DataFrame, holdout_steps: tuple[str, ...], family: str) -> dict[str, float]:
    mask = (all_splits["family"] == family) & (all_splits["holdout_steps"] == "|".join(holdout_steps))
    row = all_splits.loc[mask].iloc[0]
    return {
        "1": float(row["coef_1"]),
        "V": float(row["coef_V"]),
        "V_drive_minus_V": float(row["coef_V_drive_minus_V"]),
    }


def plot_pair_rollout(pair_name: str, label: str, coeffs: dict[str, float], steps: tuple[str, ...], prepared_map: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(len(steps), 2, figsize=(12.5, 4.2 * len(steps)), sharex=False)
    if len(steps) == 1:
        axes = np.array([axes])
    for row_idx, step in enumerate(steps):
        seg = prepared_map[step]
        rel_time, pred = finalv2.rollout_tau_prediction(coeffs, seg)
        tau_true = seg["tau"].to_numpy(dtype=float)
        err = np.abs(pred - tau_true)
        rmse = float(np.sqrt(np.mean((pred - tau_true) ** 2)))

        axes[row_idx, 0].plot(rel_time, tau_true, label="observed tau", linewidth=1.35)
        axes[row_idx, 0].plot(rel_time, pred, label="predicted tau", linewidth=1.15)
        axes[row_idx, 0].set_title(f"{pair_name}: {step}")
        axes[row_idx, 0].set_ylabel("tau")
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].legend(fontsize=8)
        axes[row_idx, 0].text(0.01, 0.03, f"RMSE={rmse:.3f}", transform=axes[row_idx, 0].transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"})

        axes[row_idx, 1].plot(rel_time, err, color="tab:red", linewidth=1.15)
        axes[row_idx, 1].plot(rel_time, seg["V_drive"] - seg["V"], color="tab:blue", linewidth=0.95, alpha=0.7)
        axes[row_idx, 1].set_title(f"{step}: |tau error| with forcing context")
        axes[row_idx, 1].set_ylabel("error / forcing")
        axes[row_idx, 1].grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle(f"Proposal Tau alternative holdout: {label}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_gallery(pair_specs: list[dict], prepared_map: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(len(pair_specs), 2, figsize=(12.5, 3.6 * len(pair_specs)), sharex=False)
    for row_idx, spec in enumerate(pair_specs):
        step = spec["steps"][0]
        seg = prepared_map[step]
        rel_time, pred = finalv2.rollout_tau_prediction(spec["coeffs"], seg)
        tau_true = seg["tau"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((pred - tau_true) ** 2)))
        axes[row_idx, 0].plot(rel_time, tau_true, linewidth=1.25, label="observed")
        axes[row_idx, 0].plot(rel_time, pred, linewidth=1.05, label="predicted")
        axes[row_idx, 0].set_title(f"{spec['short_label']}: {step}")
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].set_ylabel("tau")
        axes[row_idx, 0].text(0.01, 0.03, f"RMSE={rmse:.3f}", transform=axes[row_idx, 0].transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"})

        step_b = spec["steps"][1]
        seg_b = prepared_map[step_b]
        rel_time_b, pred_b = finalv2.rollout_tau_prediction(spec["coeffs"], seg_b)
        tau_true_b = seg_b["tau"].to_numpy(dtype=float)
        rmse_b = float(np.sqrt(np.mean((pred_b - tau_true_b) ** 2)))
        axes[row_idx, 1].plot(rel_time_b, tau_true_b, linewidth=1.25, label="observed")
        axes[row_idx, 1].plot(rel_time_b, pred_b, linewidth=1.05, label="predicted")
        axes[row_idx, 1].set_title(f"{spec['short_label']}: {step_b}")
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_ylabel("tau")
        axes[row_idx, 1].text(0.01, 0.03, f"RMSE={rmse_b:.3f}", transform=axes[row_idx, 1].transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"})
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle("Proposal Tau alternative holdout gallery inside Finalv2", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_gallery(single_specs: list[dict], prepared_map: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(len(single_specs), 2, figsize=(12.5, 3.6 * len(single_specs)), sharex=False)
    if len(single_specs) == 1:
        axes = np.array([axes])
    for row_idx, spec in enumerate(single_specs):
        step = spec["step"]
        seg = prepared_map[step]
        rel_time, pred = finalv2.rollout_tau_prediction(spec["coeffs"], seg)
        tau_true = seg["tau"].to_numpy(dtype=float)
        err = np.abs(pred - tau_true)
        rmse = float(np.sqrt(np.mean((pred - tau_true) ** 2)))
        axes[row_idx, 0].plot(rel_time, tau_true, linewidth=1.25, label="observed")
        axes[row_idx, 0].plot(rel_time, pred, linewidth=1.05, label="predicted")
        axes[row_idx, 0].set_title(f"{spec['label']}: {step}")
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].set_ylabel("tau")
        axes[row_idx, 0].text(0.01, 0.03, f"RMSE={rmse:.3f}", transform=axes[row_idx, 0].transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"})
        axes[row_idx, 1].plot(rel_time, err, color="tab:red", linewidth=1.15)
        axes[row_idx, 1].set_title(f"{step}: absolute error")
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_ylabel("|tau error|")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time since step start [s]")
    fig.suptitle("Proposal Tau single-step holdout examples inside Finalv2", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pair_df, all_splits, step_df = load_tables()
    prepared_map = finalv2.load_tau_prepared_map()

    pair_specs = []
    for step_a, step_b, short_label in PAIR_CHOICES:
        coeffs = get_split_coeffs(all_splits, (step_a, step_b), "leave_two_out")
        pair_name = f"{step_a} + {step_b}"
        pair_row = pair_df.loc[(pair_df["step_a"] == step_a) & (pair_df["step_b"] == step_b)].iloc[0]
        label = f"{short_label} | rank {int(pair_row['pair_rank'])}/28 | pair RMSE {float(pair_row['mean_pair_tau_rollout_rmse']):.3f}"
        out_path = ALT_DIR / f"Proposal_Tau_holdout_{step_a.replace('p5838_', '')}_{step_b.replace('p5838_', '')}.png"
        plot_pair_rollout(pair_name, label, coeffs, (step_a, step_b), prepared_map, out_path)
        pair_specs.append(
            {
                "steps": (step_a, step_b),
                "coeffs": coeffs,
                "short_label": f"{short_label} (rank {int(pair_row['pair_rank'])})",
                "path": out_path,
            }
        )

    single_specs = []
    for step, label in SINGLE_CHOICES:
        coeffs = get_split_coeffs(all_splits, (step,), "single_step")
        out_path = ALT_DIR / f"Proposal_Tau_single_{step.replace('p5838_', '')}.png"
        single_specs.append({"step": step, "label": label, "coeffs": coeffs, "path": out_path})

    plot_pair_gallery(pair_specs, prepared_map, ALT_DIR / "Proposal_Tau_alternative_holdout_gallery.png")
    plot_single_gallery(single_specs, prepared_map, ALT_DIR / "Proposal_Tau_single_step_holdout_gallery.png")

    summary_lines = [
        "# Alternative Holdouts",
        "",
        "This folder adds a few alternative holdout graphs inside Finalv2 without changing the compact tau law class.",
        "",
        "Included pair examples:",
    ]
    for spec in pair_specs:
        summary_lines.append(f"- {spec['steps'][0]} + {spec['steps'][1]}: {spec['short_label']}")
    summary_lines.extend(
        [
            "",
            "Included single-step examples:",
            "- p5838_step3: easy single-step holdout",
            "- p5838_step5: medium single-step holdout",
            "- p5838_step2: hard single-step holdout",
            "",
            "The original Finalv2 conclusions are unchanged; these figures only broaden the visual holdout coverage.",
        ]
    )
    (ALT_DIR / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Created alternative holdout figures in Finalv2.")


if __name__ == "__main__":
    main()
