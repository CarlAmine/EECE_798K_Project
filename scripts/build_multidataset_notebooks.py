"""Compatibility wrapper for scripts/notebooks/build_multidataset_notebooks.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    print("[DEPRECATED] Use scripts/notebooks/build_multidataset_notebooks.py instead.")
    target = Path(__file__).resolve().parent / "notebooks" / "build_multidataset_notebooks.py"
    runpy.run_path(str(target), run_name="__main__")
