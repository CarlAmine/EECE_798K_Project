from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
FINALV4_DIR = RESULTS_DIR / "Finalv4"
FIG_DIR = FINALV4_DIR / "Figures"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def maybe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    parts = [fmt_row(headers), divider]
    parts.extend(fmt_row(row) for row in rows)
    return "\n".join(parts)


def build_comparison_summary(reduced_step: pd.DataFrame | None, exact_step: pd.DataFrame | None) -> tuple[str, pd.DataFrame]:
    rows = []
    lines = [
        "# V Comparison Summary",
        "",
        "- Reduced RSF is the primary usable V model and is evaluated exhaustively.",
        "- Exact RSF is a secondary exact-form comparison and is evaluated only on all single-step holdouts plus selected informative pairs.",
        "- This asymmetry is intentional for computational reliability and because exact RSF is not the main final V model.",
        "",
    ]
    if reduced_step is not None:
        rows.append(
            {
                "branch": "Reduced RSF",
                "best_step": reduced_step.iloc[0]["step_name"],
                "hardest_step": reduced_step.iloc[-1]["step_name"],
                "mean_step_rmse": float(reduced_step["mean_holdout_velocity_rmse"].mean()),
                "median_step_rmse": float(reduced_step["mean_holdout_velocity_rmse"].median()),
            }
        )
    if exact_step is not None:
        rows.append(
            {
                "branch": "Exact RSF",
                "best_step": exact_step.iloc[0]["step_name"],
                "hardest_step": exact_step.iloc[-1]["step_name"],
                "mean_step_rmse": float(exact_step["mean_holdout_velocity_rmse"].mean()),
                "median_step_rmse": float(exact_step["mean_holdout_velocity_rmse"].median()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        lines.append(dataframe_to_markdown(table))
        lines.append("")
        if len(table) == 2:
            winner = "Reduced RSF" if float(table.loc[table["branch"] == "Reduced RSF", "median_step_rmse"].iloc[0]) <= float(table.loc[table["branch"] == "Exact RSF", "median_step_rmse"].iloc[0]) else "Exact RSF"
            lines.append(f"- Lower median step RMSE in the available summaries: `{winner}`.")
    return "\n".join(lines) + "\n", table


def build_reduced_vs_exact_figure(reduced_step: pd.DataFrame | None, exact_step: pd.DataFrame | None, out_path: Path) -> None:
    if reduced_step is None or exact_step is None:
        return
    step_order = reduced_step.sort_values("difficulty_rank")["step_name"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(step_order))
    axes[0].plot(x, reduced_step.set_index("step_name").loc[step_order, "mean_holdout_velocity_rmse"], marker="o", label="Reduced RSF")
    axes[0].plot(x, exact_step.set_index("step_name").loc[step_order, "mean_holdout_velocity_rmse"], marker="s", label="Exact RSF")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(step_order, rotation=35, ha="right")
    axes[0].set_title("Per-step mean holdout V RMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    summary = pd.DataFrame(
        [
            {"branch": "Reduced RSF", "mean": float(reduced_step["mean_holdout_velocity_rmse"].mean()), "median": float(reduced_step["mean_holdout_velocity_rmse"].median())},
            {"branch": "Exact RSF", "mean": float(exact_step["mean_holdout_velocity_rmse"].mean()), "median": float(exact_step["mean_holdout_velocity_rmse"].median())},
        ]
    )
    idx = np.arange(len(summary))
    width = 0.35
    axes[1].bar(idx - width / 2, summary["mean"], width, label="mean step RMSE")
    axes[1].bar(idx + width / 2, summary["median"], width, label="median step RMSE")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(summary["branch"])
    axes[1].set_title("Branch summary")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Reduced vs exact V summary", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    reduced_step = load_optional_csv(RESULTS_DIR / "v_step_difficulty_table.csv")
    exact_step = load_optional_csv(RESULTS_DIR / "v_exact_step_difficulty_table.csv")
    reduced_pair = load_optional_csv(RESULTS_DIR / "v_pair_difficulty_table.csv")
    exact_pair = load_optional_csv(RESULTS_DIR / "v_exact_pair_difficulty_table.csv")

    copied = []
    for name in [
        "v_single_step_holdout_ranking_reduced.png",
        "v_leave_two_out_heatmap_reduced.png",
        "v_split_distribution_summary_reduced.png",
        "v_easy_vs_hard_examples_reduced.png",
        "v_step_context_comparison_reduced.png",
        "v_single_step_holdout_ranking_exact.png",
        "v_selected_pairs_examples_exact.png",
    ]:
        src = RESULTS_DIR / name
        dst = FIG_DIR / name
        if maybe_copy(src, dst):
            copied.append(name)

    comparison_md, master_table = build_comparison_summary(reduced_step, exact_step)
    build_reduced_vs_exact_figure(reduced_step, exact_step, RESULTS_DIR / "reduced_vs_exact_v_summary.png")
    if maybe_copy(RESULTS_DIR / "reduced_vs_exact_v_summary.png", FIG_DIR / "reduced_vs_exact_v_summary.png"):
        copied.append("reduced_vs_exact_v_summary.png")

    maybe_copy(RESULTS_DIR / "v_reduced_summary.md", FINALV4_DIR / "v_reduced_summary.md")
    maybe_copy(RESULTS_DIR / "v_exact_summary.md", FINALV4_DIR / "v_exact_summary.md")
    (FINALV4_DIR / "v_comparison_summary.md").write_text(comparison_md, encoding="utf-8")

    if not master_table.empty:
        master_table.to_csv(FINALV4_DIR / "v_master_table.csv", index=False)
    else:
        pd.DataFrame(columns=["branch", "best_step", "hardest_step", "mean_step_rmse", "median_step_rmse"]).to_csv(FINALV4_DIR / "v_master_table.csv", index=False)

    readme_lines = [
        "# Finalv4",
        "",
        "This folder packages the staged V evaluation workflow.",
        "",
        "Contents:",
        "- `v_reduced_summary.md`: exhaustive reduced-RSF results",
        "- `v_exact_summary.md`: selected-split exact-RSF results",
        "- `v_comparison_summary.md`: branch-level comparison",
        "- `v_master_table.csv`: compact branch summary table",
        "",
        "Reporting notes:",
        "- Reduced RSF is the primary usable V model and is evaluated exhaustively.",
        "- Exact RSF is a secondary exact-form comparison and is evaluated only on all single-step holdouts plus selected informative pairs.",
        "",
        "Copied figures:",
    ]
    readme_lines.extend([f"- `{name}`" for name in copied] or ["- `none available yet`"])
    (FINALV4_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(json.dumps({"finalv4_dir": str(FINALV4_DIR), "copied_figures": copied}, indent=2))


if __name__ == "__main__":
    main()
