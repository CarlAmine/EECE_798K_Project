# Scripts Directory Guide

This folder is being reorganized to reduce one-off entrypoints and make automation safer.

## Layout

- `scripts/notebooks/`: notebook builders and notebook execution helpers.
- `scripts/tools/`: validation and developer tooling scripts.
- `scripts/*.py`: dataset and experiment entrypoints (active migration area).

## Migration Rules

1. Move scripts in small batches.
2. Leave compatibility wrappers at old paths for at least one cycle.
3. Print deprecation messages from wrappers.
4. Prefer reusable logic in `src/`, keep scripts as thin entrypoints.

## Current compatibility wrappers

- `scripts/run_notebooks.py` -> `scripts/notebooks/run_notebooks.py`
- `scripts/build_multidataset_notebooks.py` -> `scripts/notebooks/build_multidataset_notebooks.py`

## Quick check

Run this to verify active entrypoints parse and expose `--help` where expected:

```bash
python scripts/tools/check_script_entrypoints.py
```

