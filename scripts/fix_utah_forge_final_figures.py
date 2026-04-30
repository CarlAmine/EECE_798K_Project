from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "utah_forge" / "Final"
FIGURES_DIR = FINAL_DIR / "Figures"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_figure(source: Path, dest: Path) -> str:
    ensure_dir(dest.parent)
    shutil.copy2(source, dest)
    return "copied"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def collect_current_final_tree() -> dict[str, list[str]]:
    files: list[str] = []
    folders: list[str] = []
    for path in sorted(FINAL_DIR.rglob("*")):
        rel_path = rel(path)
        if path.is_dir():
            folders.append(rel_path)
        else:
            files.append(rel_path)
    return {"files": files, "folders": folders}


def build_manifest() -> list[dict[str, str]]:
    pairs = [
        (
            ROOT / "results" / "utah_forge" / "overall_performance_comparison.png",
            FIGURES_DIR / "Overview" / "overall_performance_comparison.png",
            "project-level overview figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "train_vs_holdout_summary.png",
            FIGURES_DIR / "Overview" / "train_vs_holdout_summary.png",
            "project-level split comparison figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "step_difficulty_heatmap.png",
            FIGURES_DIR / "Overview" / "step_difficulty_heatmap.png",
            "project-level step difficulty figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "exact_vs_reduced_summary.png",
            FIGURES_DIR / "Overview" / "exact_vs_reduced_summary.png",
            "exact-vs-reduced comparison figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "baseline_rollout.png",
            FIGURES_DIR / "Model_A" / "Model_A_rollout.png",
            "best available Model A rollout/baseline figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "p5838_memory_rollouts" / "p5838_step2_memory_rollout.png",
            FIGURES_DIR / "Model_B" / "Model_B_rollout_step2.png",
            "saved Model B holdout rollout figure for step2",
        ),
        (
            ROOT / "results" / "utah_forge" / "p5838_memory_rollouts" / "p5838_step7_memory_rollout.png",
            FIGURES_DIR / "Model_B" / "Model_B_rollout_step7.png",
            "saved Model B holdout rollout figure for step7",
        ),
        (
            ROOT / "results" / "utah_forge" / "p5838_theta_comparison_plots" / "p5838_step2_theta_vs_memory.png",
            FIGURES_DIR / "Model_C" / "Model_C_theta_vs_memory_step2.png",
            "best available Model C theta-vs-memory comparison for step2",
        ),
        (
            ROOT / "results" / "utah_forge" / "p5838_theta_comparison_plots" / "p5838_step7_theta_vs_memory.png",
            FIGURES_DIR / "Model_C" / "Model_C_theta_vs_memory_step7.png",
            "best available Model C theta-vs-memory comparison for step7",
        ),
        (
            ROOT / "results" / "utah_forge" / "showcase_tau_fit.png",
            FIGURES_DIR / "Proposal_Tau" / "Proposal_Tau_rollout.png",
            "proposal tau rollout figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "showcase_derivative_scatter.png",
            FIGURES_DIR / "Proposal_Tau" / "Proposal_derivative_scatter.png",
            "proposal derivative scatter figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "proposal_equation_recovery_diagnostics.png",
            FIGURES_DIR / "Proposal_Tau" / "proposal_equation_recovery_diagnostics.png",
            "proposal recovery diagnostics figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "showcase_velocity_fit.png",
            FIGURES_DIR / "Reduced_RSF" / "Reduced_RSF_rollout.png",
            "reduced RSF rollout figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "showcase_exact_rsf_fit.png",
            FIGURES_DIR / "Exact_RSF" / "Exact_RSF_rollout.png",
            "exact RSF showcase rollout figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "showcase_phaseplot.png",
            FIGURES_DIR / "Exact_RSF" / "Exact_RSF_phaseplot.png",
            "exact RSF phase plot figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "exact_rsf_showcase_theta_examples.png",
            FIGURES_DIR / "Exact_RSF" / "Exact_RSF_theta_examples.png",
            "exact RSF theta example traces",
        ),
        (
            ROOT / "results" / "utah_forge" / "exact_rsf_inverse_fit_diagnostics.png",
            FIGURES_DIR / "Exact_RSF" / "exact_rsf_inverse_fit_diagnostics.png",
            "exact RSF inverse-fit diagnostics figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "theta_equation_consistency_table.csv",
            FIGURES_DIR / "Theta_Consistency" / "theta_equation_consistency_table.csv",
            "theta consistency supporting table copied into figure package area",
        ),
        (
            FINAL_DIR / "Theta_Consistency" / "Theta_consistency_plot.png",
            FIGURES_DIR / "Theta_Consistency" / "Theta_consistency_plot.png",
            "theta consistency plot",
        ),
        (
            FINAL_DIR / "BC_Tau_Fix" / "BC_tau_fix_summary.png",
            FIGURES_DIR / "BC_Tau_Fix" / "BC_tau_fix_summary.png",
            "B/C tau-fix summary figure",
        ),
        (
            FINAL_DIR / "C_Velocity_Isolation" / "C_velocity_isolation_summary.png",
            FIGURES_DIR / "C_Velocity_Isolation" / "C_velocity_isolation_summary.png",
            "C velocity-isolation summary figure",
        ),
        (
            ROOT / "results" / "utah_forge" / "multistep_tau_rollout_gallery.png",
            FIGURES_DIR / "Multistep_Assessment" / "multistep_tau_rollout_gallery.png",
            "multistep tau rollout gallery",
        ),
        (
            ROOT / "results" / "utah_forge" / "multistep_velocity_rollout_gallery.png",
            FIGURES_DIR / "Multistep_Assessment" / "multistep_velocity_rollout_gallery.png",
            "multistep velocity rollout gallery",
        ),
        (
            ROOT / "results" / "utah_forge" / "multistep_exact_rsf_gallery.png",
            FIGURES_DIR / "Multistep_Assessment" / "multistep_exact_rsf_gallery.png",
            "multistep exact RSF gallery",
        ),
        (
            ROOT / "results" / "utah_forge" / "multistep_phaseplots.png",
            FIGURES_DIR / "Multistep_Assessment" / "multistep_phaseplots.png",
            "multistep phase plots",
        ),
    ]
    manifest: list[dict[str, str]] = []
    for source, dest, note in pairs:
        entry = {
            "source_path": rel(source) if source.exists() else source.as_posix(),
            "destination_path": rel(dest),
            "status": "",
            "note": note,
        }
        if source.exists():
            entry["status"] = copy_figure(source, dest)
        else:
            entry["status"] = "missing_source"
        manifest.append(entry)
    return manifest


def expected_figures() -> list[str]:
    return [
        rel(FIGURES_DIR / "Overview" / "overall_performance_comparison.png"),
        rel(FIGURES_DIR / "Overview" / "train_vs_holdout_summary.png"),
        rel(FIGURES_DIR / "Overview" / "step_difficulty_heatmap.png"),
        rel(FIGURES_DIR / "Overview" / "exact_vs_reduced_summary.png"),
        rel(FIGURES_DIR / "Proposal_Tau" / "Proposal_Tau_rollout.png"),
        rel(FIGURES_DIR / "Reduced_RSF" / "Reduced_RSF_rollout.png"),
        rel(FIGURES_DIR / "Exact_RSF" / "Exact_RSF_rollout.png"),
        rel(FIGURES_DIR / "Exact_RSF" / "Exact_RSF_phaseplot.png"),
        rel(FIGURES_DIR / "Theta_Consistency" / "Theta_consistency_plot.png"),
        rel(FIGURES_DIR / "Multistep_Assessment" / "multistep_tau_rollout_gallery.png"),
        rel(FIGURES_DIR / "Multistep_Assessment" / "multistep_velocity_rollout_gallery.png"),
        rel(FIGURES_DIR / "Multistep_Assessment" / "multistep_exact_rsf_gallery.png"),
        rel(FIGURES_DIR / "Multistep_Assessment" / "multistep_phaseplots.png"),
    ]


def update_readme() -> None:
    content = """# Utah FORGE Final Folder

This folder is a clean aggregation of the full Utah FORGE p5838 model history.

Main files:
- `final_master_report.md`: human-readable master report.
- `final_master_report.json`: machine-readable summary.
- `final_master_table.csv`: one-row-per-model/result master table.
- `final_model_index.csv`: quick index of included folders.

Subfolders:
- `Model_A`, `Model_B`, `Model_C`
- `Proposal_Tau`, `Reduced_RSF`, `Exact_RSF`, `Theta_Consistency`
- `BC_Tau_Fix`, `C_Velocity_Isolation`, `Multistep_Assessment`
- `Figures/Overview`, `Figures/Model_A`, `Figures/Model_B`, `Figures/Model_C`
- `Figures/Proposal_Tau`, `Figures/Reduced_RSF`, `Figures/Exact_RSF`
- `Figures/Theta_Consistency`, `Figures/BC_Tau_Fix`, `Figures/C_Velocity_Isolation`
- `Figures/Multistep_Assessment`

## Where The Figures Are Located
- Project-level summary figures live under `Figures/Overview/`.
- Per-model and per-result figures live under the matching `Figures/<Name>/` folder.
- The figure copy manifest is `final_figure_manifest.csv`.
- The folder audit is in `final_folder_audit.md` and `final_folder_audit.json`.
- Package verification is in `final_package_verification.md`.
"""
    (FINAL_DIR / "README.md").write_text(content, encoding="utf-8")


def update_master_report() -> None:
    report = (FINAL_DIR / "final_master_report.md").read_text(encoding="utf-8")
    if "## Where the figures are located" not in report:
        report += """

## Where the figures are located
- Overview figures: `Figures/Overview/overall_performance_comparison.png`, `Figures/Overview/train_vs_holdout_summary.png`, `Figures/Overview/step_difficulty_heatmap.png`, `Figures/Overview/exact_vs_reduced_summary.png`
- Old-model figures: `Figures/Model_A/Model_A_rollout.png`, `Figures/Model_B/Model_B_rollout_step2.png`, `Figures/Model_B/Model_B_rollout_step7.png`, `Figures/Model_C/Model_C_theta_vs_memory_step2.png`, `Figures/Model_C/Model_C_theta_vs_memory_step7.png`
- Proposal and reduced-RSF figures: `Figures/Proposal_Tau/Proposal_Tau_rollout.png`, `Figures/Proposal_Tau/Proposal_derivative_scatter.png`, `Figures/Reduced_RSF/Reduced_RSF_rollout.png`
- Exact-RSF figures: `Figures/Exact_RSF/Exact_RSF_rollout.png`, `Figures/Exact_RSF/Exact_RSF_phaseplot.png`, `Figures/Exact_RSF/Exact_RSF_theta_examples.png`, `Figures/Exact_RSF/exact_rsf_inverse_fit_diagnostics.png`
- Theta/tau-fix/velocity-isolation figures: `Figures/Theta_Consistency/Theta_consistency_plot.png`, `Figures/BC_Tau_Fix/BC_tau_fix_summary.png`, `Figures/C_Velocity_Isolation/C_velocity_isolation_summary.png`
- Multistep figures: `Figures/Multistep_Assessment/multistep_tau_rollout_gallery.png`, `Figures/Multistep_Assessment/multistep_velocity_rollout_gallery.png`, `Figures/Multistep_Assessment/multistep_exact_rsf_gallery.png`, `Figures/Multistep_Assessment/multistep_phaseplots.png`
"""
    report = report.replace(
        "## Key Project-Level Figures\n- `overall_performance_comparison.png`\n- `train_vs_holdout_summary.png`\n- `step_difficulty_heatmap.png`\n- `exact_vs_reduced_summary.png`\n",
        "## Key Project-Level Figures\n- `Figures/Overview/overall_performance_comparison.png`\n- `Figures/Overview/train_vs_holdout_summary.png`\n- `Figures/Overview/step_difficulty_heatmap.png`\n- `Figures/Overview/exact_vs_reduced_summary.png`\n",
    )
    (FINAL_DIR / "final_master_report.md").write_text(report, encoding="utf-8")


def write_manifest(manifest: list[dict[str, str]]) -> None:
    manifest_path = FINAL_DIR / "final_figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "destination_path", "status", "note"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def write_audit(before: dict[str, list[str]], manifest: list[dict[str, str]]) -> None:
    current_after = collect_current_final_tree()
    expected = expected_figures()
    missing_expected = [p for p in expected if not (ROOT / p).exists()]

    report_refs = []
    for rel_doc in [
        "results/utah_forge/Final/README.md",
        "results/utah_forge/Final/final_master_report.md",
    ]:
        text = (ROOT / rel_doc).read_text(encoding="utf-8")
        refs = []
        for token in [
            "Figures/Overview/overall_performance_comparison.png",
            "Figures/Overview/train_vs_holdout_summary.png",
            "Figures/Overview/step_difficulty_heatmap.png",
            "Figures/Overview/exact_vs_reduced_summary.png",
            "Figures/Proposal_Tau/Proposal_Tau_rollout.png",
            "Figures/Reduced_RSF/Reduced_RSF_rollout.png",
            "Figures/Exact_RSF/Exact_RSF_rollout.png",
            "Figures/Exact_RSF/Exact_RSF_phaseplot.png",
            "Figures/Theta_Consistency/Theta_consistency_plot.png",
            "Figures/Multistep_Assessment/multistep_tau_rollout_gallery.png",
            "Figures/Multistep_Assessment/multistep_velocity_rollout_gallery.png",
            "Figures/Multistep_Assessment/multistep_exact_rsf_gallery.png",
            "Figures/Multistep_Assessment/multistep_phaseplots.png",
        ]:
            refs.append(
                {
                    "path": token,
                    "exists": (FINAL_DIR / token).exists(),
                }
            )
        report_refs.append({"document": rel_doc, "references": refs})

    audit = {
        "before": before,
        "after": current_after,
        "expected_figure_paths": expected,
        "missing_expected_figures": missing_expected,
        "manifest_summary": manifest,
        "report_reference_check": report_refs,
    }
    (FINAL_DIR / "final_folder_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Final Folder Audit",
        "",
        "## Files currently inside Final",
    ]
    md_lines.extend(f"- `{p}`" for p in current_after["files"])
    md_lines.append("")
    md_lines.append("## Subfolders currently inside Final")
    md_lines.extend(f"- `{p}`" for p in current_after["folders"])
    md_lines.append("")
    md_lines.append("## Expected figure files missing")
    if missing_expected:
        md_lines.extend(f"- `{p}`" for p in missing_expected)
    else:
        md_lines.append("- None")
    md_lines.append("")
    md_lines.append("## Report reference check")
    for doc in report_refs:
        md_lines.append(f"- `{doc['document']}`")
        for ref in doc["references"]:
            status = "exists" if ref["exists"] else "missing"
            md_lines.append(f"  - `{ref['path']}`: {status}")
    (FINAL_DIR / "final_folder_audit.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def write_verification() -> None:
    report_files = [
        FINAL_DIR / "README.md",
        FINAL_DIR / "final_master_report.md",
        FINAL_DIR / "final_master_report.json",
        FINAL_DIR / "final_master_table.csv",
        FINAL_DIR / "final_model_index.csv",
        FINAL_DIR / "final_folder_audit.md",
        FINAL_DIR / "final_folder_audit.json",
        FINAL_DIR / "final_figure_manifest.csv",
    ]
    figure_paths = expected_figures()
    all_reports_exist = all(path.exists() for path in report_files)
    all_figures_exist = all((ROOT / path).exists() for path in figure_paths)
    text = [
        "# Final Package Verification",
        "",
        f"- Report files present: {'yes' if all_reports_exist else 'no'}",
        f"- Key figures present: {'yes' if all_figures_exist else 'no'}",
        "- Verified report/table files:",
    ]
    text.extend(f"  - `{rel(path)}`" for path in report_files)
    text.append("- Verified key figure files:")
    text.extend(f"  - `{path}`" for path in figure_paths)
    text.append("- Markdown report paths were updated to point to files inside `results/utah_forge/Final/Figures/...`.")
    (FINAL_DIR / "final_package_verification.md").write_text(
        "\n".join(text) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dir(FIGURES_DIR)
    before = collect_current_final_tree()
    manifest = build_manifest()
    write_manifest(manifest)
    update_readme()
    update_master_report()
    write_audit(before, manifest)
    write_verification()
    print("[final-figures] audited and populated Final/Figures package")


if __name__ == "__main__":
    main()
