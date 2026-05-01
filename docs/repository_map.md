# Repository Map

This document explains what every major folder and file in the repository contains, and provides a recommended reading path for new users.

---

## New Reader Path

If you just opened this repository for the first time, follow this sequence:

1. **[`README.md`](../README.md)** — Project overview, quick start, workflow summary
2. **[`docs/project_summary.md`](project_summary.md)** — Scientific context: what is RSF, what is θ, what was found
3. **[`results/utah_forge/p5838_final_report.md`](../results/utah_forge/p5838_final_report.md)** — The core results document
4. **[`docs/model_inventory.md`](model_inventory.md)** — What models were tried, in what order, and why
5. **[`docs/results_index.md`](results_index.md)** — Which files are final vs exploratory vs archival
6. **[`scripts/run_final_pipeline.py`](../scripts/run_final_pipeline.py)** — Pipeline entry point

---

## Full Repository Tree

```
.
├── README.md                              Project landing page
├── REPO_OVERVIEW.md                       Legacy overview (kept for history)
├── requirements.txt                       Python dependencies
├── .gitignore                             Ignore rules
├── explore_data.py                        ROOT-LEVEL: early exploratory script (should be in scripts/)
├── extract_nb_outputs.py                  ROOT-LEVEL: utility script (should be in scripts/)
├── test_write.txt                         ROOT-LEVEL: empty test file (can be deleted)
│
├── docs/                                  Complete documentation suite
│   ├── project_summary.md                 Scientific system, notation, RSF equations
│   ├── repository_map.md                  This file
│   ├── branch_strategy.md                 Branch purposes and recommended cleanup
│   ├── reproducibility.md                 Environment setup, pipeline, smoke test
│   ├── datasets.md                        Dataset descriptions, download instructions
│   ├── model_inventory.md                 All model families with rationale and results
│   ├── results_index.md                   Indexed listing of all result files
│   ├── iteration_history.md               Project progression narrative
│   └── cleanup_manifest.md               Record of all cleanup changes made
│
├── src/                                   Reusable Python source package
│   ├── __init__.py                        Package init (imports key modules)
│   ├── config.py                          Paths, constants, experiment config
│   ├── derivatives.py                     Derivative estimation: TVD, SG, finite diff
│   ├── exact_rsf.py                       Exact RSF inverse fitting (nonlinear optim.)
│   ├── io/                                Data loading
│   │   └── [loaders for .mat, CSV, HDF5]
│   ├── preprocess/                        Signal processing
│   │   └── [filtering, normalization, unit conversion]
│   ├── segmentation/                      Cycle detection
│   │   └── [peak detection, boundary finding]
│   ├── sindy/                             SINDy regression
│   │   └── [library builders, STLSQ, feature selection]
│   ├── datasets/                          Dataset-specific utilities
│   │   └── [utah_forge, lanl, pangaea loaders]
│   └── utils/                             Shared utilities
│       └── [metrics, rollout, visualization]
│
├── scripts/                               Experiment scripts
│   ├── README.md                          Script catalog with phase labels
│   ├── run_final_pipeline.py              FINAL: main pipeline entry point
│   ├── smoke_test.py                      Import and directory verification
│   │
│   │   -- Utah FORGE Analysis Scripts --
│   ├── utah_forge_proposal_equation_recovery.py    FINAL: baseline + physics SINDy
│   ├── utah_forge_reviewer_ablation.py             FINAL: A/B/C ablation study
│   ├── utah_forge_exact_rsf_showcase.py            FINAL: exact RSF fitting
│   ├── utah_forge_conditional_v_diagnostic.py      IMPORTANT DIAGNOSTIC
│   ├── utah_forge_regime_analysis.py               IMPORTANT DIAGNOSTIC
│   ├── utah_forge_regime_balanced_tau_evaluation.py IMPORTANT DIAGNOSTIC
│   ├── utah_forge_tau_all_splits_assessment.py     IMPORTANT DIAGNOSTIC
│   ├── utah_forge_v_all_splits_assessment.py       IMPORTANT DIAGNOSTIC
│   ├── utah_forge_multistep_rollout_summary.py     IMPORTANT DIAGNOSTIC
│   ├── utah_forge_sparsity_frontier.py             EXPLORATORY
│   ├── utah_forge_memory_refinement.py             EXPLORATORY
│   ├── utah_forge_augmented_theta.py               EXPLORATORY
│   ├── utah_forge_model_c_velocity_isolation.py    EXPLORATORY
│   ├── utah_forge_model_bc_tau_fix_comparison.py   EXPLORATORY
│   ├── utah_forge_exact_rsf_inverse_fit.py         EXPLORATORY
│   ├── utah_forge_exact_rsf_multistart_check.py    EXPLORATORY
│   ├── utah_forge_theta_equation_consistency.py    EXPLORATORY
│   ├── utah_forge_step_variability_diagnostics.py  EXPLORATORY
│   ├── utah_forge_rollout_metric_explainer.py      EXPLORATORY
│   ├── utah_forge_showcase_fit_visuals.py          EXPLORATORY
│   ├── utah_forge_v_reduced_all_splits.py          EXPLORATORY
│   ├── utah_forge_v_exact_selected_splits.py       EXPLORATORY
│   ├── utah_forge_v_package_finalv4.py             OBSOLETE (superseded)
│   ├── utah_forge_finalv2_*.py                     OBSOLETE (superseded)
│   ├── utah_forge_finalize_project_package.py      OBSOLETE (superseded)
│   ├── assemble_utah_forge_final_report.py         OBSOLETE (superseded)
│   ├── fix_utah_forge_final_figures.py             OBSOLETE (superseded)
│   ├── improve_utah_forge_model.py                 OBSOLETE (superseded)
│   ├── refine_utah_forge_validation.py             OBSOLETE (superseded)
│   │
│   │   -- Other Dataset Scripts --
│   ├── run_fdem_zenodo_sindy.py                    EXPLORATORY (FDEM dataset)
│   │
│   │   -- Utilities --
│   ├── build_multidataset_notebooks.py             Notebook generation
│   ├── run_notebooks.py                            Notebook execution runner
│   ├── export_utah_forge_table.m                   MATLAB table export helper
│   ├── utah_forge_script_catalog.md                Script catalog (legacy)
│   ├── notebooks/                                  Notebook templates
│   └── tools/                                      Script utilities
│
├── notebooks/                             Jupyter notebooks
│   └── [utah_forge/, lanl/, pangaea/ subdirs]
│
├── data/                                  Data placeholders
│   ├── README.md                          Data location and download guide
│   └── [utah_forge/, lanl/, pangaea/, fdem_zenodo/ — mostly gitignored]
│
├── results/                               Generated outputs
│   ├── utah_forge/                        Primary results (see results_index.md)
│   │   ├── README.md                      What is final vs historical
│   │   ├── p5838_final_report.md          FINAL: core results report
│   │   ├── best_equations_showcase.md     FINAL: discovered equations
│   │   ├── p5838_paper_section.md         FINAL: paper-ready section
│   │   ├── project_performance_assessment.md  FINAL: performance summary
│   │   ├── [many figures and JSON files]  Mix of final and exploratory
│   │   └── [subfolders for plot galleries]
│   ├── lanl/
│   ├── pangaea/
│   ├── fdem_zenodo/
│   └── multidataset_status.md
│
└── reports/                               Final paper and report assets
    ├── README.md
    ├── final_paper/
    ├── proposal/
    └── progress/
```

---

## Notes on Root-Level Files

Three files at the root level are somewhat out of place and would ideally be moved to `scripts/`:
- `explore_data.py` — early exploratory data inspection script
- `extract_nb_outputs.py` — notebook output extraction utility
- `test_write.txt` — empty test file, can be deleted

These are **not moved** in this cleanup pass to preserve git history, but are documented here for clarity.
