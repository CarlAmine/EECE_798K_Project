# Physics-Informed Sparse Identification of Laboratory Stick-Slip Friction Dynamics

**Course:** EECE 798K — Data-Driven Modeling and Machine Learning for Science

---

## Scientific Summary

This project recovers interpretable ordinary differential equation (ODE) structure from Utah FORGE laboratory stick-slip friction data (experiment p5838) using Sparse Identification of Nonlinear Dynamics (SINDy). The candidate libraries are constructed from Rate-and-State Friction (RSF) theory, augmented with memory surrogates, proxy state variables, reduced-form velocity laws, and exact RSF inverse fitting. The goal is to identify sparse, physics-consistent governing equations for shear stress (τ) and slip velocity (V) dynamics during controlled stick-slip loading events — not to predict earthquakes, but to recover the governing structure from laboratory observations.

---

## New Reader? Start Here

1. **This README** — project overview, quick start, workflow
2. [`docs/project_summary.md`](docs/project_summary.md) — scientific context, notation, RSF equations, honest assessment
3. [`docs/repository_map.md`](docs/repository_map.md) — full repo tree with explanations
4. [`docs/model_inventory.md`](docs/model_inventory.md) — all model families, what was tried, why, and what was found
5. [`results/utah_forge/p5838_final_report.md`](results/utah_forge/p5838_final_report.md) — the core results report
6. [`results/utah_forge/best_equations_showcase.md`](results/utah_forge/best_equations_showcase.md) — the recovered equations
7. [`scripts/run_final_pipeline.py`](scripts/run_final_pipeline.py) — entry point wrapper
8. [`docs/results_index.md`](docs/results_index.md) — index of all result files (final vs archival)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/CarlAmine/EECE_798K_Project.git
cd EECE_798K_Project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run smoke test (no raw data required)
python scripts/smoke_test.py

# 5. Run final pipeline wrapper (requires raw data; will explain if data is missing)
python scripts/run_final_pipeline.py
```

**Note:** Raw `.mat` data files (Utah FORGE p5838) are not committed. See [`docs/datasets.md`](docs/datasets.md) for download instructions and expected file locations.

---

## Repository Map

```
.
├── README.md                  ← You are here
├── REPO_OVERVIEW.md           ← Legacy overview (see docs/ for full docs)
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── docs/                      ← Complete project documentation
│   ├── project_summary.md     ← Scientific context and notation
│   ├── repository_map.md      ← Full tree with explanations
│   ├── branch_strategy.md     ← Branch purposes and cleanup plan
│   ├── reproducibility.md     ← Environment setup and pipeline guide
│   ├── datasets.md            ← Dataset descriptions and download info
│   ├── model_inventory.md     ← All models: rationale, features, results
│   ├── results_index.md       ← Index of final and archival results
│   ├── iteration_history.md   ← Project progression narrative
│   └── cleanup_manifest.md    ← Record of all cleanup changes
│
├── src/                       ← Reusable Python package
│   ├── config.py              ← Paths, constants, experiment config
│   ├── derivatives.py         ← Derivative estimation methods
│   ├── exact_rsf.py           ← Exact RSF inverse fitting utilities
│   ├── io/                    ← Data loading (MATLAB, CSV, etc.)
│   ├── preprocess/            ← Signal filtering and normalization
│   ├── segmentation/          ← Stick-slip cycle detection
│   ├── sindy/                 ← SINDy library builders and STLSQ
│   ├── datasets/              ← Dataset-specific loaders
│   └── utils/                 ← Metrics, rollout, visualization helpers
│
├── scripts/                   ← Experiment scripts (Utah FORGE primary)
│   ├── README.md              ← Script catalog with phase labels
│   ├── run_final_pipeline.py  ← FINAL: pipeline entry point wrapper
│   ├── smoke_test.py          ← Import and path verification
│   ├── utah_forge_proposal_equation_recovery.py    ← FINAL: main analysis
│   ├── utah_forge_reviewer_ablation.py             ← FINAL: ablation study
│   ├── utah_forge_exact_rsf_showcase.py            ← FINAL: RSF showcase
│   ├── utah_forge_conditional_v_diagnostic.py      ← IMPORTANT DIAGNOSTIC
│   ├── utah_forge_regime_analysis.py               ← IMPORTANT DIAGNOSTIC
│   ├── [many other scripts...]                     ← See scripts/README.md
│   └── notebooks/             ← Notebook-generating utilities
│
├── notebooks/                 ← Jupyter inspection workflows
│   └── [lanl/, pangaea/, utah_forge/ subdirs]
│
├── data/                      ← Raw data placeholders (not committed)
│   ├── README.md
│   ├── utah_forge/            ← Utah FORGE p5838 .mat files (ignored)
│   ├── lanl/                  ← LANL seismic lab data (ignored)
│   ├── pangaea/               ← PANGAEA dataset (ignored)
│   └── fdem_zenodo/           ← FDEM Zenodo dataset (ignored)
│
├── results/                   ← Generated outputs (lightweight committed)
│   ├── utah_forge/            ← Primary results (see results_index.md)
│   │   ├── README.md          ← What is final vs historical
│   │   └── [figures, reports, metrics, CSVs...]
│   ├── lanl/
│   └── pangaea/
│
└── reports/                   ← Final paper and proposal assets
    ├── README.md
    ├── final_paper/           ← Final report assets
    ├── proposal/              ← Project proposal
    └── progress/              ← Progress report
```

---

## Main Scientific Workflow

```
Raw .mat data (Utah FORGE p5838)
        ↓
  [src/io] Loading and parsing
        ↓
  [src/preprocess] Signal filtering, normalization, unit conversion
        ↓
  [src/segmentation] Stick-slip cycle detection and segmentation
        ↓
  [src/derivatives] Derivative estimation (TVD, finite differences, SG filter)
        ↓
  [src/sindy] RSF-inspired candidate library construction
        ↓
  STLSQ sparse regression → discovered equations
        ↓
  [src/utils] Rollout validation on train and holdout cycles
        ↓
  Diagnostics: regime analysis, theta consistency, RSF identifiability
        ↓
  Results written to results/utah_forge/
```

---

## Model Families

All models target the Utah FORGE p5838 stick-slip dataset. See [`docs/model_inventory.md`](docs/model_inventory.md) for full details.

| Label | Description |
|-------|-------------|
| **Poly baseline** | Polynomial SINDy, no physics prior |
| **Model A** | Observed-only RSF-informed SINDy (τ, V, V_drive) |
| **Model B** | Memory-augmented (τ_avg, τ_ema surrogates) |
| **Model C** | Theta-proxy informed (augmented state with θ surrogate) |
| **Tau-spring** | Tau-isolated spring-loading law: dτ/dt = k(V_drive − V) |
| **V-reduced** | Reduced velocity law with logarithmic RSF structure |
| **Conditional-V** | Velocity variants conditioned on τ regime |
| **Exact RSF** | Exact RSF inverse fitting (nonlinear parameter estimation) |
| **Theta-check** | Theta equation consistency verification |
| **Regime diag** | Regime mismatch diagnostics |

---

## Key Findings

1. **Stress law is identifiable:** dτ/dt ≈ k(V_drive − V) is robustly recovered across splits (R² > 0.99 on τ derivative).
2. **Velocity law partially recovers RSF structure:** log(V) and interaction terms appear but coefficient stability is limited.
3. **Memory terms improve rollout:** but are not proof of true θ recovery — surrogates (τ_avg, τ_ema) help numerically without physical interpretation.
4. **Explicit θ recovery is weak:** θ is not directly observed; surrogate-based recovery is non-identifiable.
5. **Exact RSF fitting is parameter unstable:** multi-start optimization confirms poor identifiability from observed data alone.
6. **Regime mismatch explains holdout inconsistency:** training and holdout cycles differ in dynamic regime, limiting generalization.

---

## Data Note

Raw `.mat` files for Utah FORGE experiment p5838 are **not committed** to this repository (they are large binary files and may have access restrictions). See [`docs/datasets.md`](docs/datasets.md) for:
- Expected file paths
- Download sources
- How to place data for scripts to find it

Preprocessed cycle CSVs (short/medium/long) are committed in `results/utah_forge/` and can be used for model inspection without raw data.

---

## Results Note

Results in `results/utah_forge/` are organized as follows:
- **Final results:** `p5838_final_report.md`, `best_equations_showcase.md`, `p5838_paper_section.md`, key figures (`showcase_*.png`, `multistep_*.png`)
- **Historical/exploratory:** All other files are preserved as-is for iteration history
- **Archive subdirectory:** See `results/utah_forge/README.md` for the map of what is final vs historical

See [`docs/results_index.md`](docs/results_index.md) for a complete indexed listing.

---

## Branch Note

| Branch | Purpose |
|--------|--------|
| `main` | Clean final submission branch |
| `dev` | Active development / integration branch |
| `malek-utah-forge` | **Protected collaborator branch — DO NOT MODIFY** |
| `Multidataset-validation` | Historical multi-dataset exploration |
| `codex/*` | Temporary agent branches (see `docs/branch_strategy.md`) |

See [`docs/branch_strategy.md`](docs/branch_strategy.md) for full branch strategy and cleanup recommendations.

---

## Citation / Course Context

This project was developed for **EECE 798K: Data-Driven Modeling and Machine Learning for Science**. The scientific context involves laboratory-scale stick-slip friction governed by Rate-and-State Friction (RSF) equations, using data from the Utah FORGE geothermal research site (experiment p5838).

The SINDy framework follows:
> Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937.

RSF theory references:
> Dieterich, J. H. (1979). Modeling of rock friction. *JGR*, 84, 2161–2168.  
> Ruina, A. (1983). Slip instability and state variable friction laws. *JGR*, 88, 10359–10370.
