# src API Audit (Temporary)

This note records mismatches between `src/README.md` usage examples and the real module exports.

## 1) Imports shown in `src/README.md` that do not exist

From the usage block in `src/README.md`, these imports are not currently exported under those names:

- `from src.io import load_mat_file`
- `from src.segmentation import detect_cycles`
- `from src.derivatives import compute_tvd_derivative`
- `from src.sindy import build_rsf_library`

## 2) Real imports that should replace them

Closest real imports based on current modules:

- Replace `from src.io import load_mat_file` with one of the real dataset loaders in `src.io`, for example:
  - `from src.io import load_utah_forge_dataset`
  - `from src.io import load_lanl_train`
  - `from src.io import load_pangaea_dataset`
- Replace `from src.segmentation import detect_cycles` with dataset-specific segmentation exports such as:
  - `from src.segmentation import segment_utah_forge_events`
  - `from src.segmentation import segment_lanl_cycles`
  - `from src.segmentation import segment_pangaea_events`
- Replace `from src.derivatives import compute_tvd_derivative` with existing derivative utilities such as:
  - `from src.derivatives import compute_derivative`
  - `from src.derivatives import derivative_savgol`
  - `from src.derivatives import derivative_spline`
- Replace `from src.sindy import build_rsf_library` with the real library builder in `src/sindy/library.py`:
  - `from src.sindy.library import build_polynomial_library`

## 3) Whether `SINDyModel` exists

Yes. `SINDyModel` exists in `src/sindy/models.py`.

## 4) Real library-builder functions in `src/sindy/library.py`

- `build_polynomial_library`

## 5) Derivative functions in `src/derivatives.py`

- `finite_difference_forward`
- `finite_difference_backward`
- `finite_difference_central`
- `compute_derivative`
- `derivative_savgol`
- `derivative_spline`
- `estimate_derivatives_df`
- `estimate_velocity_from_accel`
