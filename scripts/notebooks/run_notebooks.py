import argparse
import base64
import io
import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stream_output(name: str, text: str) -> dict | None:
    if not text:
        return None
    return {
        "output_type": "stream",
        "name": name,
        "text": text.splitlines(keepends=True),
    }


def figure_output(fig) -> dict:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "text/plain": f"<Figure size {int(fig.get_figwidth() * 100)}x{int(fig.get_figheight() * 100)} with {len(fig.axes)} Axes>",
            "image/png": encoded,
        },
        "metadata": {},
    }


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    globals_dict: dict = {"__name__": "__main__"}
    execution_count = 1
    original_cwd = Path.cwd()

    os.chdir(path.parent)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell.get("source", []))
            cell["outputs"] = []
            cell["execution_count"] = execution_count
            execution_count += 1

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            display_outputs: list[dict] = []

            original_show = plt.show

            def patched_show(*args, **kwargs):
                for fig_number in plt.get_fignums():
                    fig = plt.figure(fig_number)
                    display_outputs.append(figure_output(fig))
                plt.close("all")

            plt.show = patched_show

            try:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    exec(compile(source, str(path), "exec"), globals_dict)
            except Exception as exc:
                tb = traceback.format_exc().splitlines()
                stdout = stream_output("stdout", stdout_buffer.getvalue())
                stderr = stream_output("stderr", stderr_buffer.getvalue())
                if stdout:
                    cell["outputs"].append(stdout)
                if stderr:
                    cell["outputs"].append(stderr)
                cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": exc.__class__.__name__,
                        "evalue": str(exc),
                        "traceback": tb,
                    }
                )
                path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
                raise
            finally:
                plt.show = original_show

            stdout = stream_output("stdout", stdout_buffer.getvalue())
            stderr = stream_output("stderr", stderr_buffer.getvalue())
            if stdout:
                cell["outputs"].append(stdout)
            if stderr:
                cell["outputs"].append(stderr)
            cell["outputs"].extend(display_outputs)

        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    finally:
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+")
    args = parser.parse_args()

    for notebook_path in args.notebooks:
        execute_notebook(Path(notebook_path).resolve())


if __name__ == "__main__":
    main()
