#!/usr/bin/env python
"""
smoke_test.py
=============
Smoke test for the EECE 798K Utah FORGE stick-slip SINDy project.

This script verifies:
1. Core source modules are importable
2. Expected directory structure exists
3. No import errors in key modules
4. Optionally checks for raw data files (warns if missing, does not fail)

Requirements: None beyond the standard library and installed dependencies.
Does NOT require raw .mat data files to pass.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

passed = []
warnings = []
failed = []


def check(name, condition, warn=False):
    if condition:
        passed.append(name)
        print(f"[OK]   {name}")
    elif warn:
        warnings.append(name)
        print(f"[WARN] {name}")
    else:
        failed.append(name)
        print(f"[FAIL] {name}")


print("\nSmoke Test: EECE 798K Utah FORGE SINDy Project")
print("=" * 50)
print()

# --------------------------------------------------
# 1. Directory structure
# --------------------------------------------------
print("--- Directory Checks ---")
check("src/ exists", (REPO_ROOT / "src").exists())
check("scripts/ exists", (REPO_ROOT / "scripts").exists())
check("results/ exists", (REPO_ROOT / "results").exists())
check("results/utah_forge/ exists", (REPO_ROOT / "results" / "utah_forge").exists())
check("docs/ exists", (REPO_ROOT / "docs").exists())
check(
    "data/utah_forge/ exists (raw data)",
    (REPO_ROOT / "data" / "utah_forge").exists(),
    warn=True,
)
check(
    "data/utah_forge/*.mat files present",
    len(list((REPO_ROOT / "data" / "utah_forge").glob("*.mat"))) > 0
    if (REPO_ROOT / "data" / "utah_forge").exists()
    else False,
    warn=True,
)

print()

# --------------------------------------------------
# 2. Core imports
# --------------------------------------------------
print("--- Import Checks ---")

try:
    import numpy
    check("numpy importable", True)
except ImportError as e:
    check(f"numpy importable ({e})", False)

try:
    import pandas
    check("pandas importable", True)
except ImportError as e:
    check(f"pandas importable ({e})", False)

try:
    import scipy
    check("scipy importable", True)
except ImportError as e:
    check(f"scipy importable ({e})", False)

try:
    import matplotlib
    check("matplotlib importable", True)
except ImportError as e:
    check(f"matplotlib importable ({e})", False)

try:
    import sklearn
    check("scikit-learn importable", True)
except ImportError as e:
    check(f"scikit-learn importable ({e})", False)

# --------------------------------------------------
# 3. src package imports
# --------------------------------------------------
print()
print("--- src Package Checks ---")

try:
    import src
    check("src package importable", True)
except Exception as e:
    check(f"src package ({e})", False)

try:
    from src import config
    check("src.config importable", True)
except Exception as e:
    check(f"src.config ({e})", False)

try:
    from src import derivatives
    check("src.derivatives importable", True)
except Exception as e:
    check(f"src.derivatives ({e})", False)

try:
    from src import exact_rsf
    check("src.exact_rsf importable", True)
except Exception as e:
    check(f"src.exact_rsf ({e})", False)

try:
    from src import io as src_io
    check("src.io importable", True)
except Exception as e:
    check(f"src.io ({e})", False)

try:
    from src import preprocess
    check("src.preprocess importable", True)
except Exception as e:
    check(f"src.preprocess ({e})", False)

try:
    from src import segmentation
    check("src.segmentation importable", True)
except Exception as e:
    check(f"src.segmentation ({e})", False)

try:
    from src import sindy
    check("src.sindy importable", True)
except Exception as e:
    check(f"src.sindy ({e})", False)

try:
    from src import utils
    check("src.utils importable", True)
except Exception as e:
    check(f"src.utils ({e})", False)

try:
    from src import datasets
    check("src.datasets importable", True)
except Exception as e:
    check(f"src.datasets ({e})", False)

# --------------------------------------------------
# 4. Results spot check
# --------------------------------------------------
print()
print("--- Results Spot Checks ---")
check(
    "results/utah_forge/p5838_final_report.md exists",
    (REPO_ROOT / "results" / "utah_forge" / "p5838_final_report.md").exists(),
)
check(
    "results/utah_forge/best_equations_showcase.md exists",
    (REPO_ROOT / "results" / "utah_forge" / "best_equations_showcase.md").exists(),
)

# --------------------------------------------------
# 5. Summary
# --------------------------------------------------
print()
print("=" * 50)
print(f"PASSED:   {len(passed)}")
print(f"WARNINGS: {len(warnings)} (data missing — see docs/datasets.md)")
print(f"FAILED:   {len(failed)}")

if failed:
    print("\nSmoke test FAILED. Check failed items above.")
    sys.exit(1)
else:
    print("\nSmoke test PASSED (import checks)")
    if warnings:
        print("Raw data not present. Pipeline will need data to run fully.")
    sys.exit(0)
