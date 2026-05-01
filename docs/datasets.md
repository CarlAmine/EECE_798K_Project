# Datasets

## Overview

| Dataset | Role | Status | ODE Suitability |
|---------|------|--------|-----------------|
| Utah FORGE p5838 | **Primary** | Data present (gitignored) | Excellent |
| LANL | Secondary | Gitignored | Limited |
| PANGAEA | Secondary | Gitignored | Limited |
| FDEM Zenodo | Exploratory | Gitignored | Partial |

---

## Utah FORGE p5838 (Primary Dataset)

### Why Primary?
Utah FORGE experiment p5838 provides the cleanest laboratory stick-slip data for ODE recovery:
- **Controlled loading:** V_drive is known and controlled
- **Well-instrumented:** τ, V, and displacement are all measured at high sampling rate
- **Clear stick-slip cycles:** Natural segmentation into clean loading/slip events
- **Known physics:** RSF theory applies directly; parameters are approximately known
- **Sufficient cycles:** Enough repeat cycles for cross-validation

### Expected File Location
```
data/utah_forge/
  └── p5838_*.mat   (one or more MATLAB data files)
```

### File Format
MAT files (MATLAB binary). Loaded using `mat73` or `scipy.io.loadmat` depending on format version. Key channels: time, shear stress (τ), axial displacement, velocity (V or derived), normal force.

### Where to Download
Utah FORGE experimental data may be available through:
- The Utah FORGE data portal: https://utahforge.com/
- Contact the experiment PI for data sharing
- NGDS (National Geothermal Data System)

### Scripts that Use This Data
- All scripts in `scripts/utah_forge_*.py`
- All notebooks in `notebooks/utah_forge/`

### Derived Outputs (Already Committed)
Preprocessed CSVs that can be used without raw data:
- `results/utah_forge/selected_cycle_short.csv`
- `results/utah_forge/selected_cycle_medium.csv`
- `results/utah_forge/selected_cycle_long.csv`

---

## LANL (Secondary Dataset)

### What It Is
Los Alamos National Laboratory laboratory friction data from shear experiments. This dataset has been used in machine learning studies for seismic signal prediction (e.g., Rouet-Leduc et al. 2017).

### Why Secondary?
LANL is designed for prediction tasks ("predict time to failure"), not equation recovery. The measured signals and experimental protocol are less directly aligned with RSF ODE recovery. Useful for comparison and generalization testing.

### Expected File Location
```
data/lanl/
  └── [LANL experiment files]
```

### Where to Download
Originally from Kaggle competition: https://www.kaggle.com/c/LANL-Earthquake-Prediction

---

## PANGAEA (Secondary Dataset)

### What It Is
Laboratory friction data available from the PANGAEA data repository, covering various rock types and loading conditions.

### Why Secondary?
PANGAEA datasets have variable instrumentation and are not always set up for direct RSF ODE recovery. Useful for multi-dataset generalization but not the primary focus.

### Expected File Location
```
data/pangaea/
  └── [PANGAEA data files]
```

### Where to Download
https://www.pangaea.de/ (search for relevant friction experiments)

---

## FDEM Zenodo (Exploratory Dataset)

### What It Is
A friction dynamics dataset published on Zenodo, covering fault experiments with electromagnetic measurements (FDEM = Fault Dynamics Electromagnetic Monitoring).

### Role
Exploratory — one analysis script (`scripts/run_fdem_zenodo_sindy.py`) was written for this dataset. Results are in `results/fdem_zenodo/`.

### Expected File Location
```
data/fdem_zenodo/
  └── [Zenodo dataset files]
```

### Where to Download
Search Zenodo (https://zenodo.org/) for the specific dataset identifier.
