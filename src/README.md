# src Package

This package contains all reusable Python modules for the Utah FORGE stick-slip SINDy project.

---

## Module Overview

### `config.py`
Central configuration file containing:
- File paths (data, results, logs)
- Physical constants (machine stiffness, reference velocity, etc.)
- Default experiment parameters (window sizes, filter settings)
- Dataset identifiers

### `derivatives.py`
Derivative estimation methods for noisy time-series data:
- Total Variation Differentiation (TVD)
- Savitzky-Golay filter derivatives
- Finite difference methods
- Smoothed derivative pipelines

### `exact_rsf.py`
Exact Rate-and-State Friction (RSF) utilities:
- Full RSF ODE system integration
- Inverse fitting (nonlinear parameter estimation for a, b, D_c, mu0, k)
- Parameter identifiability analysis
- Multi-start optimization wrapper

### `io/`
Data loading modules:
- MATLAB `.mat` file loader (supports mat5 and mat73 formats)
- CSV/HDF5 loaders
- Channel extraction and parsing
- Utah FORGE specific loaders

### `preprocess/`
Signal processing:
- Low-pass and Butterworth filtering
- Signal normalization and standardization
- Unit conversion (physical to normalized)
- Outlier detection and handling

### `segmentation/`
Stick-slip cycle detection:
- Peak and trough detection in shear stress signal
- Cycle boundary identification
- Quality filtering (removing poor cycles)
- Cycle metadata extraction

### `sindy/`
SINDy regression:
- RSF-informed candidate library construction
- STLSQ (Sequentially Thresholded Least Squares)
- Hyperparameter sweep utilities
- Feature selection helpers
- Library with physics-motivated terms (log(V), tau*V, V_drive-V, etc.)

### `datasets/`
Dataset-specific utilities:
- Utah FORGE p5838 specific loading and preprocessing
- LANL dataset utilities
- PANGAEA dataset utilities
- Cross-dataset standardization

### `utils/`
Shared utilities:
- Rollout simulation (Euler/RK4 integration of discovered ODE)
- Metrics (R², RMSE, normalized rollout error)
- Visualization helpers
- Cross-validation splits
- Regime analysis tools

---

## Usage

```python
# Example: load data and estimate derivatives
from src import config
from src.io import load_mat_file
from src.segmentation import detect_cycles
from src.derivatives import compute_tvd_derivative
from src.sindy import build_rsf_library

# Load data
data = load_mat_file(config.DATA_PATH / "p5838_experiment.mat")

# Segment into cycles
cycles = detect_cycles(data["tau"], data["time"])

# Estimate derivatives for one cycle
tau_dot = compute_tvd_derivative(cycles[0]["tau"], cycles[0]["time"])
```

---

## Notes

- All numerical behavior is unchanged from the research version.
- Docstrings may be sparse in some modules — refer to the scripts that use them for usage examples.
- Do not refactor or restructure without running the smoke test: `python scripts/smoke_test.py`
