# Branch Strategy

## Current Branches

| Branch | SHA (at cleanup) | Status | Purpose |
|--------|-----------------|--------|---------|
| `main` | 49baa1fd | Active | Public-facing, should be clean final state |
| `dev` | 2bd48c59 | Active | Integration branch, active development |
| `malek-utah-forge` | 2074d62c | **PROTECTED** | Collaborator branch — DO NOT MODIFY |
| `Multidataset-validation` | 76467f23 | Historical | Multi-dataset exploration experiments |
| `codex/locate-file-modification-options-on-github` | 50d7b510 | Agent branch | Temporary Codex agent branch |
| `codex/locate-file-modification-options-on-github-64luqc` | 87bd1f62 | Agent branch | Temporary Codex agent branch (variant) |
| `repo-cleanup/readability-pass` | (this branch) | Cleanup | Documentation and readability overhaul |

---

## Branch Purposes

### `main`
Intended as the clean, public-facing final submission branch. Should contain:
- Final readable project structure
- Complete README and docs
- Source code
- Reproducibility instructions
- Final report assets
- Indexed results

Currently `main` is behind `dev` and does not have the full cleanup. After this PR is merged to `dev`, a follow-up PR from `dev` → `main` should be opened for the final submission.

### `dev`
Active development and integration branch. Contains all exploratory work alongside stable code. After this cleanup branch is merged, `dev` will be fully documented and navigable.

### `malek-utah-forge` — PROTECTED
**This branch must not be modified, rebased, force-pushed, or deleted.**  
It is a collaborator's branch and is preserved as-is. The SHA at time of cleanup is `2074d62c39b917011565332f15740cbac26b427c`.

### `Multidataset-validation`
Historical branch for multi-dataset validation experiments (LANL, PANGAEA, FDEM Zenodo alongside Utah FORGE). SHA: `76467f23b574d55bd7d80839b515ca0b2e52ea23`.

Recommendation: Do not delete. Create an archival tag if merging or closing:
```bash
# Suggested (do not run without confirming changes are captured)
git tag archive/multidataset-validation-pre-close 76467f23b574d55bd7d80839b515ca0b2e52ea23
git push origin archive/multidataset-validation-pre-close
```

### `codex/*` branches
Temporary agent branches created during automated code exploration. SHA references:
- `codex/locate-file-modification-options-on-github`: `50d7b510`
- `codex/locate-file-modification-options-on-github-64luqc`: `87bd1f62`

Recommendation: Verify no unique changes exist in these branches that aren't in `dev`. Then create archival tags and close/delete:
```bash
# Suggested workflow (verify first):
git tag archive/codex-locate-pre-close 50d7b5107c0941c32c3cce777707f87dc834648a
git tag archive/codex-locate-64luqc-pre-close 87bd1f620b16a830d6a397fbc1bf931c4def6aad
git push origin --tags
# Then delete branches via GitHub UI or:
# git push origin --delete codex/locate-file-modification-options-on-github
# git push origin --delete codex/locate-file-modification-options-on-github-64luqc
```

---

## Recommended Final State

```
main               ← Clean final paper/reproducible release
dev                ← Ongoing working branch (fully documented post-cleanup)
malek-utah-forge   ← Protected collaborator branch, UNTOUCHED FOREVER

[Eventually close after tagging:]
Multidataset-validation
codex/*
repo-cleanup/readability-pass (after merging to dev)
```

---

## Archive Tags Recommended

Before major cleanup of `main`, create these tags:
```bash
git tag archive/dev-before-cleanup-20250101 2bd48c59d3f9356f340f470f7b280f6345acabb0
git tag archive/main-before-cleanup-20250101 49baa1fdadbf4efaf48ed888b9977fcda892b614
git push origin --tags
```

The exact date should reflect when the cleanup is finalized.

---

## What Was NOT Done in This Cleanup Pass

- No branches were deleted
- No force pushes were performed
- `malek-utah-forge` was not touched
- No rebases were performed
- All historical commits are preserved
