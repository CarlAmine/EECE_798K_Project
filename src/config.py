from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .utils.paths import dataset_data_dir, dataset_notebooks_dir, dataset_results_dir


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    label: str
    source_url: str
    raw_dir: Path
    results_dir: Path
    notebooks_dir: Path
    preferred_raw_names: tuple[str, ...]
    raw_globs: tuple[str, ...]
    column_aliases: dict[str, tuple[str, ...]]
    velocity_mode: str
    smoothing: dict
    segmentation: dict
    notes: str = ""

    def to_summary(self) -> dict:
        data = asdict(self)
        for key in ("raw_dir", "results_dir", "notebooks_dir"):
            data[key] = str(data[key])
        return data


LANL_CONFIG = DatasetConfig(
    name="lanl",
    label="LANL Earthquake Prediction",
    source_url="https://www.kaggle.com/competitions/LANL-Earthquake-Prediction",
    raw_dir=dataset_data_dir("lanl"),
    results_dir=dataset_results_dir("lanl"),
    notebooks_dir=dataset_notebooks_dir("lanl"),
    preferred_raw_names=("train.csv", "lanl_train.csvtdunczn5.part"),
    raw_globs=("*.csv", "*.part"),
    column_aliases={
        "time": ("time",),
        "acoustic_data": ("acoustic_data",),
        "time_to_failure": ("time_to_failure",),
        "state_1": ("tau_proxy",),
        "state_2": ("V_proxy",),
    },
    velocity_mode="proxy_from_smoothed_acoustic_data",
    smoothing={"window": 101, "polyorder": 3},
    segmentation={"strategy": "time_to_failure_reset", "min_cycle_length": 50_000},
    notes="Proxy-state dataset. tau_proxy is derived from acoustic_data and V_proxy is derived from the smoothed proxy.",
)

PANGAEA_CONFIG = DatasetConfig(
    name="pangaea",
    label="PANGAEA 915062",
    source_url="https://doi.org/10.1594/PANGAEA.915062",
    raw_dir=dataset_data_dir("pangaea"),
    results_dir=dataset_results_dir("pangaea"),
    notebooks_dir=dataset_notebooks_dir("pangaea"),
    preferred_raw_names=("PANGAEA.915062.txt", "pangaea_915062.txt", "pangaea_915062.tsv", "pangaea_915062.tab"),
    raw_globs=("*.txt", "*.tab", "*.tsv", "*.csv"),
    column_aliases={
        "time": ("time", "t"),
        "tau": ("tau", "shear stress", "shear_stress", "stress", "friction", "mu", "μ", "µ"),
        "displacement": ("u", "displacement", "slip", "shear displacement"),
        "velocity": ("v", "velocity", "sliding velocity", "slip velocity"),
    },
    velocity_mode="measured_or_derived_from_displacement",
    smoothing={"window": 31, "polyorder": 3},
    segmentation={"strategy": "stress_drop_or_peak_based", "min_cycle_length": 25},
    notes="The DOI dataset appears to publish rate-and-state fit outputs rather than raw time-series. Time-series friction analysis requires an additional raw file with time and mechanical variables.",
)

UTAH_FORGE_CONFIG = DatasetConfig(
    name="utah_forge",
    label="Utah FORGE Laboratory Shear Experiments",
    source_url="https://catalog.data.gov/dataset/utah-forge-laboratory-shear-experiments-linking-fault-roughness-friction-permeability-and-",
    raw_dir=dataset_data_dir("utah_forge"),
    results_dir=dataset_results_dir("utah_forge"),
    notebooks_dir=dataset_notebooks_dir("utah_forge"),
    preferred_raw_names=("0_README.txt",),
    raw_globs=("*_datatable.mat", "*.mat", "*.txt"),
    column_aliases={
        "time": ("time", "Time", "t", "seconds"),
        "tau": ("tau", "shear stress", "shear_stress"),
        "displacement": ("d_int", "displacement", "slip", "u", "shear_displacement", "d_ext"),
        "velocity": ("v_int", "velocity", "slip_velocity", "V", "load_point_velocity", "v_ext"),
        "friction_coefficient": ("mu", "friction coefficient", "friction_coefficient"),
    },
    velocity_mode="measured_or_derived_from_displacement",
    smoothing={"window": 31, "polyorder": 3},
    segmentation={"strategy": "stress_drop_or_displacement_event", "min_cycle_length": 25},
    notes="Mechanical/acoustic time-series are distributed as MATLAB .mat datatables plus a README that defines the variables and units.",
)


DATASET_CONFIGS = {
    "lanl": LANL_CONFIG,
    "pangaea": PANGAEA_CONFIG,
    "utah_forge": UTAH_FORGE_CONFIG,
}


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {name}") from exc
