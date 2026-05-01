# Data

This folder contains raw data placeholders and loading instructions. Raw data files are **not committed** to this repository.

See [`docs/datasets.md`](../docs/datasets.md) for full dataset descriptions and download instructions.

---

## Folder Structure

```
data/
├── README.md           This file
├── utah_forge/         Utah FORGE p5838 .mat files (gitignored)
├── lanl/               LANL seismic lab data (gitignored)
├── pangaea/            PANGAEA dataset (gitignored)
└── fdem_zenodo/        FDEM Zenodo dataset (gitignored)
```

---

## Utah FORGE (Primary Dataset)

**Expected path:** `data/utah_forge/p5838_*.mat`

Place Utah FORGE experiment p5838 MATLAB files here. These are loaded by all `scripts/utah_forge_*.py` scripts.

See `docs/datasets.md` for download instructions.

---

## LANL (Secondary)

**Expected path:** `data/lanl/`

LANL laboratory friction data. Used for secondary/comparison analyses.

---

## PANGAEA (Secondary)

**Expected path:** `data/pangaea/`

PANGAEA laboratory friction data. Used for secondary analyses.

---

## FDEM Zenodo (Exploratory)

**Expected path:** `data/fdem_zenodo/`

FDEM dataset from Zenodo. Used in `scripts/run_fdem_zenodo_sindy.py`.

---

## Note on Preprocessed Data

Preprocessed cycle CSVs are committed in `results/utah_forge/` and can be used for model inspection without raw data:
- `results/utah_forge/selected_cycle_short.csv`
- `results/utah_forge/selected_cycle_medium.csv`
- `results/utah_forge/selected_cycle_long.csv`
