from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return repo_root() / "data"


def results_root() -> Path:
    return repo_root() / "results"


def notebooks_root() -> Path:
    return repo_root() / "notebooks"


def dataset_data_dir(dataset: str) -> Path:
    return data_root() / dataset


def dataset_results_dir(dataset: str) -> Path:
    return results_root() / dataset


def dataset_notebooks_dir(dataset: str) -> Path:
    return notebooks_root() / dataset


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None

