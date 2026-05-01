# Cleanup Manifest

This document records all changes made during the `repo-cleanup/readability-pass` branch. It is intended to satisfy the course requirement for documented iteration history.

**Cleanup performed:** May 2026  
**Branch:** `repo-cleanup/readability-pass`  
**Base commit (dev):** `2bd48c59d3f9356f340f470f7b280f6345acabb0`

---

## Files Created (New)

| Path | Reason |
|------|--------|
| `README.md` | Complete overhaul: replaced 19-byte minimal placeholder with full project landing page |
| `requirements.txt` | New: formal dependency manifest for reproducibility |
| `docs/project_summary.md` | New: scientific context, notation, RSF equations, honest assessment |
| `docs/repository_map.md` | New: full tree with explanations and new reader path |
| `docs/branch_strategy.md` | New: branch purposes, protection notes, cleanup recommendations |
| `docs/reproducibility.md` | New: environment setup, pipeline guide, smoke test instructions |
| `docs/datasets.md` | New: dataset descriptions, download sources, file locations |
| `docs/model_inventory.md` | New: all 10 model families with rationale, features, and results |
| `docs/results_index.md` | New: complete index of final vs exploratory vs archival results |
| `docs/iteration_history.md` | New: 9-phase project progression narrative |
| `docs/cleanup_manifest.md` | New: this file |
| `scripts/run_final_pipeline.py` | New: final pipeline entry point wrapper |
| `scripts/smoke_test.py` | New: import and directory verification script |
| `reports/README.md` | New: explanation of reports folder structure |
| `results/utah_forge/README.md` | Updated: expanded from minimal placeholder to comprehensive guide |

---

## Files NOT Modified

| Category | Action |
|----------|--------|
| All files in `src/` | No changes — scientific code untouched |
| All files in `scripts/` (existing) | No changes — scripts preserved as-is |
| All files in `results/utah_forge/` | No changes — results preserved as-is |
| All files in `notebooks/` | No changes |
| All files in `data/` | No changes |
| `REPO_OVERVIEW.md` | No changes — kept as-is alongside new docs |
| `.gitignore` | No changes |

---

## Files NOT Deleted

Per the project philosophy: **do not erase research history.**

The following files are obsolete or misplaced but were deliberately kept:

| File | Status | Reason Kept |
|------|--------|-------------|
| `explore_data.py` (root level) | Misplaced | Documents early exploration; moving would break history |
| `extract_nb_outputs.py` (root level) | Misplaced | Same |
| `test_write.txt` (root level) | Empty test file | Documents codex agent test; harmless |
| `scripts/utah_forge_finalv2_*.py` | Obsolete | Documents iteration history (v2 packaging attempt) |
| `scripts/utah_forge_finalize_project_package.py` | Obsolete | Documents v1 packaging attempt |
| `scripts/assemble_utah_forge_final_report.py` | Obsolete | Documents assembly iteration |
| `scripts/fix_utah_forge_final_figures.py` | Obsolete | Documents figure-fixing iteration |
| `scripts/improve_utah_forge_model.py` | Obsolete | Documents model improvement iteration |
| `scripts/refine_utah_forge_validation.py` | Obsolete | Documents validation refinement |
| `scripts/utah_forge_v_package_finalv4.py` | Obsolete | Documents finalv4 packaging attempt |

---

## Branches NOT Touched

| Branch | SHA | Status |
|--------|-----|--------|
| `malek-utah-forge` | `2074d62c39b917011565332f15740cbac26b427c` | **Untouched — protected** |
| `main` | `49baa1fdadbf4efaf48ed888b9977fcda892b614` | Untouched — cleanup is in separate branch |
| `Multidataset-validation` | `76467f23b574d55bd7d80839b515ca0b2e52ea23` | Untouched |
| `codex/*` branches | Various | Untouched |

---

## Pre-Cleanup SHA References (Archival)

For reference, the commit SHAs before cleanup:
- `dev` base: `2bd48c59d3f9356f340f470f7b280f6345acabb0`
- `main` base: `49baa1fdadbf4efaf48ed888b9977fcda892b614`
- `malek-utah-forge`: `2074d62c39b917011565332f15740cbac26b427c`
- `Multidataset-validation`: `76467f23b574d55bd7d80839b515ca0b2e52ea23`

---

## Known Remaining Issues (Deferred)

These were identified but deliberately deferred to avoid risky changes:

1. **Root-level scripts** (`explore_data.py`, `extract_nb_outputs.py`) should move to `scripts/` but weren't moved to preserve git history.
2. **`results/utah_forge/sindy_sweep_results.csv`** (75 MB) is very large for git. Should be added to `.gitignore` or moved to external storage, but removing it would alter committed history.
3. **`results/utah_forge/` structure** is flat with many mixed files. An `archive/` subfolder reorganization was planned but deferred to avoid breaking any script path references.
4. **`reports/` folder** is newly created but no paper content was moved into it (paper content may not be committed).
5. **`src/` docstrings:** Some modules lack top-level docstrings. Adding them is safe but deferred to avoid any inadvertent changes to scientific code.
