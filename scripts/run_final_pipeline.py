#!/usr/bin/env python
"""
run_final_pipeline.py
=====================
Final pipeline entry point for the Utah FORGE p5838 stick-slip SINDy project.

This wrapper:
1. Prints project purpose and overview
2. Checks expected data paths
3. Lists and optionally runs the final analysis scripts in order
4. Writes outputs to results/utah_forge/
5. Fails gracefully with instructions if data is missing

Usage:
    python scripts/run_final_pipeline.py           # Run full pipeline
    python scripts/run_final_pipeline.py --check   # Check only (no execution)
    python scripts/run_final_pipeline.py --list    # List scripts only

Course: EECE 798K - Data-Driven Modeling and Machine Learning for Science
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# ============================================================
# PROJECT INFO
# ============================================================

PROJECT_TITLE = """
=======================================================================
 Physics-Informed Sparse Identification of Laboratory
 Stick-Slip Friction Dynamics (Utah FORGE p5838)
 EECE 798K - Data-Driven Modeling and Machine Learning for Science
=======================================================================

Goal: Recover interpretable governing ODEs for shear stress (tau) and
      slip velocity (V) from laboratory stick-slip data using SINDy
      with RSF-informed candidate libraries.

Key finding: d(tau)/dt = k * (V_drive - V) is robustly identified.
See docs/project_summary.md for full scientific context.
"""

# ============================================================
# EXPECTED PATHS
# ============================================================

REPO_ROOT = Path(__file__).parent.parent

DATA_DIR = REPO_ROOT / "data" / "utah_forge"
RESULTS_DIR = REPO_ROOT / "results" / "utah_forge"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Final pipeline scripts in recommended execution order
FINAL_SCRIPTS = [
    (
        "01_main_sindy",
        "utah_forge_proposal_equation_recovery.py",
        "FINAL: Main SINDy analysis - polynomial baseline + physics-informed library",
    ),
    (
        "02_ablation",
        "utah_forge_reviewer_ablation.py",
        "FINAL: A/B/C model ablation study (observed-only vs memory vs theta)",
    ),
    (
        "03_tau_splits",
        "utah_forge_tau_all_splits_assessment.py",
        "FINAL: Tau equation cross-validation across all train/holdout splits",
    ),
    (
        "04_velocity_splits",
        "utah_forge_v_all_splits_assessment.py",
        "FINAL: Velocity equation cross-validation across all splits",
    ),
    (
        "05_rollout",
        "utah_forge_multistep_rollout_summary.py",
        "FINAL: Multi-step rollout validation summary",
    ),
    (
        "06_exact_rsf",
        "utah_forge_exact_rsf_showcase.py",
        "FINAL: Exact RSF inverse fitting showcase and identifiability analysis",
    ),
    (
        "07_regime",
        "utah_forge_regime_analysis.py",
        "IMPORTANT DIAGNOSTIC: Regime mismatch analysis",
    ),
    (
        "08_conditional_v",
        "utah_forge_conditional_v_diagnostic.py",
        "IMPORTANT DIAGNOSTIC: Conditional velocity diagnostics by regime",
    ),
]


def check_environment():
    """Check that the environment is set up correctly."""
    issues = []
    warnings = []

    # Check data directory
    if not DATA_DIR.exists():
        warnings.append(f"Data directory not found: {DATA_DIR}")
        warnings.append("  Create it and place Utah FORGE p5838 .mat files inside.")
        warnings.append("  See docs/datasets.md for download instructions.")
    else:
        mat_files = list(DATA_DIR.glob("*.mat"))
        if not mat_files:
            warnings.append(f"No .mat files found in {DATA_DIR}")
            warnings.append("  See docs/datasets.md for download instructions.")
        else:
            print(f"[OK] Found {len(mat_files)} .mat file(s) in {DATA_DIR}")

    # Check results directory
    if not RESULTS_DIR.exists():
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Created results directory: {RESULTS_DIR}")
        except Exception as e:
            issues.append(f"Cannot create results directory: {e}")
    else:
        print(f"[OK] Results directory exists: {RESULTS_DIR}")

    # Check src imports
    src_dir = REPO_ROOT / "src"
    if src_dir.exists():
        print(f"[OK] src/ package found")
    else:
        issues.append("src/ directory not found")

    # Check script files
    missing_scripts = []
    for step_id, script_name, _ in FINAL_SCRIPTS:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            missing_scripts.append(script_name)
    if missing_scripts:
        issues.append(f"Missing scripts: {', '.join(missing_scripts)}")
    else:
        print(f"[OK] All {len(FINAL_SCRIPTS)} final scripts found")

    return issues, warnings


def print_script_list():
    """Print the list of final pipeline scripts."""
    print("\nFinal Pipeline Scripts (in order):\n")
    print(f"{'Step':<8} {'Script':<55} {'Description'}")
    print("-" * 120)
    for step_id, script_name, description in FINAL_SCRIPTS:
        print(f"{step_id:<8} {script_name:<55} {description}")
    print()


def run_pipeline(dry_run=False):
    """Run the final pipeline scripts in order."""
    print("\nRunning final pipeline...\n")

    for step_id, script_name, description in FINAL_SCRIPTS:
        script_path = SCRIPTS_DIR / script_name
        print(f"\n{'='*60}")
        print(f"Step {step_id}: {script_name}")
        print(f"  {description}")
        print(f"{'='*60}")

        if dry_run:
            print(f"  [DRY RUN] Would run: python {script_path}")
            continue

        if not script_path.exists():
            print(f"  [SKIP] Script not found: {script_path}")
            continue

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(REPO_ROOT),
                check=False,
            )
            if result.returncode == 0:
                print(f"  [OK] Completed successfully")
            else:
                print(f"  [WARN] Script exited with code {result.returncode}")
        except Exception as e:
            print(f"  [ERROR] Failed to run: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Utah FORGE p5838 stick-slip SINDy final pipeline"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check environment only, do not run"
    )
    parser.add_argument(
        "--list", action="store_true", help="List scripts only, do not run"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    print(PROJECT_TITLE)

    # Add repo root to sys.path so src/ is importable
    sys.path.insert(0, str(REPO_ROOT))

    # Environment check
    print("\nEnvironment Check:\n")
    issues, warnings = check_environment()

    for w in warnings:
        print(f"[WARN] {w}")
    for issue in issues:
        print(f"[ERROR] {issue}")

    if args.list or args.check:
        print_script_list()
        if warnings:
            print(
                "\nData is missing. Scripts will fail without raw .mat files.\n"
                "See docs/datasets.md for download instructions.\n"
                "Preprocessed CSVs in results/utah_forge/ can be used for inspection.\n"
            )
        return

    if warnings and not args.dry_run:
        print(
            "\n[WARN] Raw data not found. Most scripts require .mat files.\n"
            "  - To inspect existing results: see results/utah_forge/\n"
            "  - To read documentation: see docs/\n"
            "  - To download data: see docs/datasets.md\n"
            "\nUse --dry-run to see what would be executed.\n"
            "Use --list to see the script list.\n"
        )
        sys.exit(0)

    print_script_list()
    run_pipeline(dry_run=args.dry_run)

    print("\nPipeline complete. Results written to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
