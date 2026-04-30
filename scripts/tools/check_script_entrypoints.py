"""Basic smoke check for selected script entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

ENTRYPOINTS = [
    ["python", "scripts/build_multidataset_notebooks.py"],
    ["python", "scripts/notebooks/build_multidataset_notebooks.py"],
]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def main() -> int:
    failures = 0
    for cmd in ENTRYPOINTS:
        code, output = run(cmd)
        label = "PASS" if code == 0 else "FAIL"
        print(f"[{label}] {' '.join(cmd)}")
        if code != 0:
            failures += 1
            print(output)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
