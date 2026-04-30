import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"


def lines(text: str) -> list[str]:
    return [f"{line}\n" for line in textwrap.dedent(text).strip("\n").splitlines()]


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(source)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2) + "\n", encoding="utf-8")


def lanl_notebooks() -> dict[str, list[dict]]:
    nb00 = [
        markdown_cell(
            """
            # LANL 00 Load

            Repo-native inspection notebook for the LANL Earthquake Prediction dataset.
            This pipeline stays inside the `lanl` namespace and keeps the proxy-state assumption explicit.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import json
            import sys

            REPO_ROOT = Path.cwd().resolve()
            while not (REPO_ROOT / "src").exists():
                REPO_ROOT = REPO_ROOT.parent
            sys.path.insert(0, str(REPO_ROOT))

            from src.config import LANL_CONFIG
            from src.datasets.lanl import build_lanl_summary
            from src.io.lanl import load_lanl_train
            from src.preprocess.lanl import clean_lanl_dataframe
            from src.utils.plotting import plot_signal_panel

            RESULTS_DIR = LANL_CONFIG.results_dir
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            INSPECTION_ROWS = 200_000
            PLOT_ROWS = 20_000
            """
        ),
        code_cell(
            """
            summary = build_lanl_summary(nrows=INSPECTION_ROWS)
            summary_path = RESULTS_DIR / "dataset_summary.json"
            inventory_path = RESULTS_DIR / "raw_file_inventory.json"
            readme_path = RESULTS_DIR / "README.md"
            subset_path = RESULTS_DIR / "inspection_subset.csv"

            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            readme_path.write_text(
                "\\n".join(
                    [
                        "# LANL Dataset Summary",
                        "",
                        f"- Source URL: {summary['dataset']['source_url']}",
                        f"- Raw file: {summary['raw_file']}",
                        "- State variables: tau_proxy, V_proxy",
                        "- Velocity is derived from the smoothed acoustic proxy",
                        f"- Segmentation rule: {summary['dataset']['segmentation']['strategy']}",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )

            inspection_df = clean_lanl_dataframe(load_lanl_train(nrows=INSPECTION_ROWS))
            inspection_df.to_csv(subset_path, index=False)
            inventory = {
                "dataset": "lanl",
                "raw_dir": str(LANL_CONFIG.raw_dir),
                "filenames_found": sorted(path.name for path in LANL_CONFIG.raw_dir.iterdir() if path.is_file()),
                "file_types_found": sorted({path.suffix.lower() or "<no_extension>" for path in LANL_CONFIG.raw_dir.iterdir() if path.is_file()}),
                "loader": "load_lanl_train",
                "loader_status": "ok",
                "loader_raw_file": summary["raw_file"],
                "parsed_columns": summary["schema"]["columns"],
                "available_variables": summary["schema"]["columns"],
            }
            inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            print(inspection_df.head())
            """
        ),
        code_cell(
            """
            plot_df = inspection_df.iloc[:PLOT_ROWS].copy()
            plot_signal_panel(plot_df, "time", ["acoustic_data", "time_to_failure"], "LANL inspection subset")
            """
        ),
    ]

    nb01 = [
        markdown_cell(
            """
            # LANL 01 Preprocess And Segment

            Build the LANL proxy state, detect failure resets, export a selected cycle, and save the segmentation summary.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import json
            import sys

            import pandas as pd

            REPO_ROOT = Path.cwd().resolve()
            while not (REPO_ROOT / "src").exists():
                REPO_ROOT = REPO_ROOT.parent
            sys.path.insert(0, str(REPO_ROOT))

            from src.config import LANL_CONFIG
            from src.io.lanl import load_lanl_train, scan_lanl_reset_indices, sample_lanl_cycle
            from src.preprocess.lanl import clean_lanl_dataframe, add_lanl_proxies
            from src.segmentation.lanl import segment_lanl_cycles
            from src.segmentation.common import summarize_segments
            from src.utils.plotting import plot_signal_panel, plot_segment_boundaries

            RESULTS_DIR = LANL_CONFIG.results_dir
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            INITIAL_ROWS = 1_000_000
            RESET_BUFFER_ROWS = 10_000
            SELECTED_CYCLE_MAX_ROWS = 100_000
            """
        ),
        code_cell(
            """
            reset_indices = scan_lanl_reset_indices(count=2)
            if len(reset_indices) < 2:
                raise RuntimeError("LANL segmentation requires at least two resets to export a full cycle.")

            resolved_rows = max(INITIAL_ROWS, reset_indices[0] + RESET_BUFFER_ROWS)
            subset_df = add_lanl_proxies(clean_lanl_dataframe(load_lanl_train(nrows=resolved_rows)))
            segments = segment_lanl_cycles(
                subset_df,
                failure_col="time_to_failure",
                min_cycle_length=LANL_CONFIG.segmentation["min_cycle_length"],
            )
            segment_summary = summarize_segments(segments)

            cycle_df, cycle_step = sample_lanl_cycle(
                reset_indices[0],
                reset_indices[1],
                max_rows=SELECTED_CYCLE_MAX_ROWS,
            )
            cycle_df = add_lanl_proxies(clean_lanl_dataframe(cycle_df))

            cycle_path = RESULTS_DIR / "lanl_selected_cycle.csv"
            metadata_path = RESULTS_DIR / "lanl_selected_cycle_metadata.json"
            segmentation_path = RESULTS_DIR / "segmentation_summary.json"

            cycle_df.to_csv(cycle_path, index=False)
            metadata = {
                "dataset": "lanl",
                "state_columns": ["tau_proxy", "V_proxy"],
                "state_labels": ["tau_proxy", "V_proxy"],
                "velocity_mode": "derived_proxy",
                "cycle_step": int(cycle_step),
                "cycle_path": str(cycle_path),
                "selected_cycle_range": [int(reset_indices[0]), int(reset_indices[1])],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            segmentation_path.write_text(
                json.dumps(
                    {
                        "reset_indices": reset_indices,
                        "resolved_subset_rows": int(resolved_rows),
                        "segments": segments,
                        "segment_summary": segment_summary,
                        "selected_cycle_path": str(cycle_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print("Detected reset indices:", reset_indices)
            print("Resolved subset rows:", resolved_rows)
            print("Segment summary:", segment_summary)
            print("Saved selected cycle to:", cycle_path)
            """
        ),
        code_cell(
            """
            reset_window = subset_df.iloc[max(0, reset_indices[0] - 20_000): reset_indices[0] + 20_000].copy()
            plot_signal_panel(reset_window, "time", ["acoustic_data", "tau_proxy", "time_to_failure"], "LANL reset window")
            plot_segment_boundaries(subset_df.iloc[: min(len(subset_df), reset_indices[0] + 10_000)], "time", "time_to_failure", segments, "LANL segmentation boundaries")
            """
        ),
    ]

    nb02 = [
        markdown_cell(
            """
            # LANL 02 SINDy Baseline

            Fit a baseline SINDy model on a selected LANL proxy-state cycle and save discovered equations into the LANL namespace.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import json
            import sys

            import numpy as np
            import pandas as pd

            REPO_ROOT = Path.cwd().resolve()
            while not (REPO_ROOT / "src").exists():
                REPO_ROOT = REPO_ROOT.parent
            sys.path.insert(0, str(REPO_ROOT))

            from src.config import LANL_CONFIG
            from src.derivatives import estimate_derivatives_df
            from src.preprocess.common import standardize_columns
            from src.sindy import build_polynomial_library, SINDyModel, relative_error, compute_rollout_metrics, rollout_polynomial
            from src.utils.plotting import plot_signal_panel

            RESULTS_DIR = LANL_CONFIG.results_dir
            CYCLE_PATH = RESULTS_DIR / "lanl_selected_cycle.csv"
            METADATA_PATH = RESULTS_DIR / "lanl_selected_cycle_metadata.json"
            ROLLOUT_PLOT_PATH = RESULTS_DIR / "baseline_rollout.png"

            POLY_DEGREE = 3
            THRESHOLD = 1e-6
            MAX_ITER = 15
            DERIVATIVE_WINDOW = 101
            DERIVATIVE_POLYORDER = 3
            ROLLOUT_POINTS = 500
            """
        ),
        code_cell(
            """
            if not CYCLE_PATH.exists() or not METADATA_PATH.exists():
                raise FileNotFoundError("Run notebooks/lanl/01_preprocess_and_segment.ipynb first.")

            cycle_df = pd.read_csv(CYCLE_PATH)
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            state_cols = metadata["state_columns"]
            state_df = cycle_df[["time", *state_cols]].copy()
            standardized_df, scaling = standardize_columns(state_df, state_cols)
            derivatives, _ = estimate_derivatives_df(
                standardized_df,
                time_col="time",
                var_cols=state_cols,
                method="savgol",
                window=DERIVATIVE_WINDOW,
                polyorder=DERIVATIVE_POLYORDER,
                add_to_df=False,
            )

            X = standardized_df[state_cols].to_numpy(dtype=float)
            Xdot = np.column_stack([derivatives[column] for column in state_cols])
            library, descriptions = build_polynomial_library(X, degree=POLY_DEGREE, var_names=state_cols)
            model = SINDyModel(threshold=THRESHOLD, max_iter=MAX_ITER)
            diagnostics = model.fit(library, Xdot, descriptions)
            Xdot_pred = model.predict(library)
            equations = model.equations(state_cols)

            rollout_len = min(ROLLOUT_POINTS, len(standardized_df))
            rollout_time = standardized_df["time"].to_numpy(dtype=float)[:rollout_len]
            rollout_true = X[:rollout_len]
            rollout_pred = rollout_polynomial(model.coefficients, descriptions, X[0], rollout_time, {"state_1": state_cols[0], "state_2": state_cols[1]})
            rollout_metrics = compute_rollout_metrics(rollout_true, rollout_pred)

            summary = {
                "dataset": "lanl",
                "state_columns": state_cols,
                "state_labels": metadata["state_labels"],
                "scaling": scaling,
                "poly_degree": POLY_DEGREE,
                "threshold": THRESHOLD,
                "max_iter": MAX_ITER,
                "library_terms": descriptions,
                "equations": equations,
                "relative_error": [float(relative_error(Xdot[:, i], Xdot_pred[:, i])) for i in range(Xdot.shape[1])],
                "rmse_like": diagnostics["residuals"].tolist(),
                "rollout_metrics": rollout_metrics,
                "rollout_plot": str(ROLLOUT_PLOT_PATH),
            }
            (RESULTS_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (RESULTS_DIR / "discovered_equations.txt").write_text("\\n".join(equations) + "\\n", encoding="utf-8")

            print(json.dumps(summary, indent=2))
            """
        ),
        code_cell(
            """
            rollout_df = pd.DataFrame(
                {
                    "time": rollout_time,
                    f"true_{state_cols[0]}": rollout_true[:, 0],
                    f"pred_{state_cols[0]}": rollout_pred[:, 0],
                    f"true_{state_cols[1]}": rollout_true[:, 1],
                    f"pred_{state_cols[1]}": rollout_pred[:, 1],
                }
            )
            fig, _ = plot_signal_panel(
                rollout_df,
                "time",
                [f"true_{state_cols[0]}", f"pred_{state_cols[0]}", f"true_{state_cols[1]}", f"pred_{state_cols[1]}"],
                "LANL baseline rollout",
                figsize=(12, 10),
            )
            fig.savefig(ROLLOUT_PLOT_PATH, dpi=200, bbox_inches="tight")
            """
        ),
    ]
    return {"00_load.ipynb": nb00, "01_preprocess_and_segment.ipynb": nb01, "02_sindy_baseline.ipynb": nb02}


def physical_notebooks(dataset: str, dataset_label: str, summary_builder_import: str, loader_import: str, preprocess_import: str, segment_import: str) -> dict[str, list[dict]]:
    summary_func = f"build_{dataset}_summary"
    loader_func = f"load_{dataset}_dataset"
    preprocess_func = f"build_{dataset}_state"
    segment_func = f"segment_{dataset}_events"
    results_name = "utah_forge" if dataset == "utah_forge" else dataset
    upper_label = dataset_label

    nb00 = [
        markdown_cell(
            f"""
            # {upper_label} 00 Load

            Inspect raw files for the {upper_label} pipeline and save an explicit dataset summary under `results/{results_name}/`.
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import json
            import sys

            REPO_ROOT = Path.cwd().resolve()
            while not (REPO_ROOT / "src").exists():
                REPO_ROOT = REPO_ROOT.parent
            sys.path.insert(0, str(REPO_ROOT))

            from src.config import get_dataset_config
            from src.datasets.{dataset} import {summary_func}
            from src.io.{dataset} import {loader_func}
            from src.utils.plotting import plot_signal_panel

            CONFIG = get_dataset_config("{dataset}")
            RESULTS_DIR = CONFIG.results_dir
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        code_cell(
            f"""
            summary = {summary_func}()
            summary_path = RESULTS_DIR / "dataset_summary.json"
            inventory_path = RESULTS_DIR / "raw_file_inventory.json"
            readme_path = RESULTS_DIR / "README.md"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            readme_path.write_text(
                "\\n".join(
                    [
                        "# {upper_label} Dataset Summary",
                        "",
                        f"- Source URL: {{summary['dataset']['source_url']}}",
                        f"- Analysis ready: {{summary.get('analysis_ready', False)}}",
                        f"- Notes: {{summary.get('notes', summary['dataset'].get('notes', ''))}}",
                        f"- Raw file: {{summary.get('raw_file')}}",
                    ]
                )
                + "\\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2))
            inventory = {{
                "dataset": "{dataset}",
                "raw_dir": str(CONFIG.raw_dir),
                "filenames_found": sorted(path.name for path in CONFIG.raw_dir.iterdir() if path.is_file()),
                "file_types_found": sorted({{path.suffix.lower() or "<no_extension>" for path in CONFIG.raw_dir.iterdir() if path.is_file()}}),
                "loader": "{loader_func}",
                "loader_status": "ok" if summary.get("raw_file") and not summary.get("error") else "error",
                "loader_raw_file": summary.get("raw_file"),
                "parsed_columns": summary.get("schema", {{}}).get("columns", []),
                "available_variables": summary.get("available_variables", summary.get("schema", {{}}).get("columns", [])),
                "column_mapping": summary.get("column_mapping", {{}}),
            }}
            inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            if summary.get("raw_file"):
                raw_df, _ = {loader_func}()
                print(raw_df.head())
            """
        ),
        code_cell(
            f"""
            if summary.get("raw_file"):
                numeric_cols = raw_df.select_dtypes(include=["number"]).columns.tolist()
                if summary.get("column_mapping", {{}}).get("time") and summary.get("column_mapping", {{}}).get("tau"):
                    preview_df = raw_df[[summary["column_mapping"]["time"], summary["column_mapping"]["tau"]]].copy()
                    preview_df.columns = ["time", "tau"]
                    plot_signal_panel(preview_df.head(min(2000, len(preview_df))), "time", ["tau"], "{upper_label} primary signal preview", figsize=(12, 4))
                elif len(numeric_cols) >= 2:
                    preview_df = raw_df[numeric_cols[:2]].head(min(2000, len(raw_df))).copy()
                    preview_df.insert(0, "sample", range(len(preview_df)))
                    plot_signal_panel(preview_df, "sample", numeric_cols[:2], "{upper_label} numeric preview")
                else:
                    print("No numeric preview available for plotting.")
            else:
                print("Raw file missing. See dataset_summary.json for the exact file placement instructions.")
            """
        ),
    ]

    nb01 = [
        markdown_cell(
            f"""
            # {upper_label} 01 Preprocess And Segment

            Apply dataset-specific preprocessing, construct the physical state, segment events, and export a selected cycle.
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import json
            import sys

            import pandas as pd

            REPO_ROOT = Path.cwd().resolve()
            while not (REPO_ROOT / "src").exists():
                REPO_ROOT = REPO_ROOT.parent
            sys.path.insert(0, str(REPO_ROOT))

            from src.config import get_dataset_config
            from src.datasets.{dataset} import {summary_func}
            from src.io.{dataset} import {loader_func}
            from src.preprocess.{dataset} import {preprocess_func}
            from src.segmentation.{dataset} import {segment_func}
            from src.segmentation.common import summarize_segments, extract_segment
            from src.utils.plotting import plot_signal_panel, plot_segment_boundaries

            CONFIG = get_dataset_config("{dataset}")
            RESULTS_DIR = CONFIG.results_dir
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        code_cell(
            f"""
            summary = {summary_func}()
            if not summary.get("analysis_ready", False):
                raise RuntimeError(
                    "This dataset is not ready for preprocessing/segmentation. "
                    f"Check {{RESULTS_DIR / 'dataset_summary.json'}} and add the required raw file(s) under {{CONFIG.raw_dir}}."
                )

            raw_df, load_summary = {loader_func}()
            state_df, state_metadata = {preprocess_func}(raw_df, load_summary["column_mapping"])
            segments = {segment_func}(state_df, tau_col="tau", min_cycle_length=CONFIG.segmentation["min_cycle_length"])
            if not segments:
                segments = [(0, len(state_df))]
            selected_cycle = extract_segment(state_df, *segments[0])

            cycle_path = RESULTS_DIR / "selected_cycle_001.csv"
            cycle_meta_path = RESULTS_DIR / "selected_cycle_001_metadata.json"
            seg_summary_path = RESULTS_DIR / "segmentation_summary.json"
            selected_cycle.to_csv(cycle_path, index=False)
            cycle_metadata = {{
                "dataset": "{dataset}",
                "state_columns": ["tau", "V"],
                "state_labels": ["tau", "V"],
                "velocity_mode": state_metadata.get("velocity_mode", CONFIG.velocity_mode),
                "cycle_path": str(cycle_path),
                "selected_segment": [int(segments[0][0]), int(segments[0][1])],
            }}
            cycle_meta_path.write_text(json.dumps(cycle_metadata, indent=2), encoding="utf-8")
            seg_summary_path.write_text(
                json.dumps(
                    {{
                        "dataset": "{dataset}",
                        "segment_summary": summarize_segments(segments),
                        "segments": segments,
                        "selected_cycle_path": str(cycle_path),
                    }},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("Selected cycle saved to:", cycle_path)
            print("Segment summary:", summarize_segments(segments))
            """
        ),
        code_cell(
            f"""
            plot_signal_panel(state_df.head(min(4000, len(state_df))), "time", ["tau", "V"], "{upper_label} physical state preview")
            plot_segment_boundaries(state_df, "time", "tau", segments, "{upper_label} segmentation boundaries")
            """
        ),
    ]

    nb02 = [
        markdown_cell(
            f"""
            # {upper_label} 02 SINDy Baseline

            Fit a baseline sparse model for the selected {upper_label} cycle and save equations under `results/{results_name}/`.
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import json
            import sys

            import numpy as np
            import pandas as pd

            REPO_ROOT = Path.cwd().resolve()
            while not (REPO_ROOT / "src").exists():
                REPO_ROOT = REPO_ROOT.parent
            sys.path.insert(0, str(REPO_ROOT))

            from src.config import get_dataset_config
            from src.derivatives import estimate_derivatives_df
            from src.preprocess.common import standardize_columns
            from src.sindy import build_polynomial_library, SINDyModel, relative_error, compute_rollout_metrics, rollout_polynomial
            from src.utils.plotting import plot_signal_panel

            CONFIG = get_dataset_config("{dataset}")
            RESULTS_DIR = CONFIG.results_dir
            CYCLE_PATH = RESULTS_DIR / "selected_cycle_001.csv"
            META_PATH = RESULTS_DIR / "selected_cycle_001_metadata.json"
            ROLLOUT_PLOT_PATH = RESULTS_DIR / "baseline_rollout.png"

            POLY_DEGREE = 2
            THRESHOLD = 1e-4
            MAX_ITER = 12
            DERIVATIVE_WINDOW = 51
            DERIVATIVE_POLYORDER = 3
            ROLLOUT_POINTS = 300
            """
        ),
        code_cell(
            f"""
            if not CYCLE_PATH.exists() or not META_PATH.exists():
                raise FileNotFoundError(
                    f"Run notebooks/{dataset}/01_preprocess_and_segment.ipynb first."
                )

            cycle_df = pd.read_csv(CYCLE_PATH)
            metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
            state_cols = metadata["state_columns"]
            state_df = cycle_df[["time", *state_cols]].copy()
            standardized_df, scaling = standardize_columns(state_df, state_cols)
            derivatives, _ = estimate_derivatives_df(
                standardized_df,
                time_col="time",
                var_cols=state_cols,
                method="savgol",
                window=DERIVATIVE_WINDOW,
                polyorder=DERIVATIVE_POLYORDER,
                add_to_df=False,
            )

            X = standardized_df[state_cols].to_numpy(dtype=float)
            Xdot = np.column_stack([derivatives[column] for column in state_cols])
            library, descriptions = build_polynomial_library(X, degree=POLY_DEGREE, var_names=state_cols)
            model = SINDyModel(threshold=THRESHOLD, max_iter=MAX_ITER)
            diagnostics = model.fit(library, Xdot, descriptions)
            Xdot_pred = model.predict(library)
            equations = model.equations(state_cols)

            rollout_len = min(ROLLOUT_POINTS, len(standardized_df))
            rollout_time = standardized_df["time"].to_numpy(dtype=float)[:rollout_len]
            rollout_true = X[:rollout_len]
            rollout_pred = rollout_polynomial(model.coefficients, descriptions, X[0], rollout_time, {{"state_1": state_cols[0], "state_2": state_cols[1]}})
            rollout_metrics = compute_rollout_metrics(rollout_true, rollout_pred)

            summary = {{
                "dataset": "{dataset}",
                "state_columns": state_cols,
                "state_labels": metadata["state_labels"],
                "scaling": scaling,
                "poly_degree": POLY_DEGREE,
                "threshold": THRESHOLD,
                "max_iter": MAX_ITER,
                "library_terms": descriptions,
                "equations": equations,
                "relative_error": [float(relative_error(Xdot[:, i], Xdot_pred[:, i])) for i in range(Xdot.shape[1])],
                "rmse_like": diagnostics["residuals"].tolist(),
                "rollout_metrics": rollout_metrics,
                "rollout_plot": str(ROLLOUT_PLOT_PATH),
            }}
            (RESULTS_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (RESULTS_DIR / "discovered_equations.txt").write_text("\\n".join(equations) + "\\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
            """
        ),
        code_cell(
            f"""
            rollout_df = pd.DataFrame(
                {{
                    "time": rollout_time,
                    f"true_{{state_cols[0]}}": rollout_true[:, 0],
                    f"pred_{{state_cols[0]}}": rollout_pred[:, 0],
                    f"true_{{state_cols[1]}}": rollout_true[:, 1],
                    f"pred_{{state_cols[1]}}": rollout_pred[:, 1],
                }}
            )
            fig, _ = plot_signal_panel(
                rollout_df,
                "time",
                [f"true_{{state_cols[0]}}", f"pred_{{state_cols[0]}}", f"true_{{state_cols[1]}}", f"pred_{{state_cols[1]}}"],
                "{upper_label} baseline rollout",
                figsize=(12, 10),
            )
            fig.savefig(ROLLOUT_PLOT_PATH, dpi=200, bbox_inches="tight")
            """
        ),
    ]

    return {"00_load.ipynb": nb00, "01_preprocess_and_segment.ipynb": nb01, "02_sindy_baseline.ipynb": nb02}


def main() -> None:
    notebook_map = {
        "lanl": lanl_notebooks(),
        "pangaea": physical_notebooks("pangaea", "PANGAEA", "src.datasets.pangaea", "src.io.pangaea", "src.preprocess.pangaea", "src.segmentation.pangaea"),
        "utah_forge": physical_notebooks("utah_forge", "Utah FORGE", "src.datasets.utah_forge", "src.io.utah_forge", "src.preprocess.utah_forge", "src.segmentation.utah_forge"),
    }

    for dataset, notebooks in notebook_map.items():
        for filename, cells in notebooks.items():
            write_notebook(NOTEBOOK_ROOT / dataset / filename, cells)


if __name__ == "__main__":
    main()
