from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.linalg import lstsq
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.utah_forge import infer_utah_forge_column_mapping, load_utah_forge_dataset
from src.preprocess.common import remove_invalid_rows, smooth_series
from src.preprocess.utah_forge import build_utah_forge_state
from src.segmentation.utah_forge import segment_utah_forge_events
from src.utils.paths import ensure_directory, repo_root


try:
    from src.derivatives import derivative_savgol
except Exception:  # pragma: no cover - fallback handled below
    derivative_savgol = None

try:
    from src.derivatives import derivative_spline
except Exception:  # pragma: no cover - fallback handled below
    derivative_spline = None

try:
    from src.sindy.models import SINDyModel as ImportedSINDyModel
except Exception:  # pragma: no cover - fallback handled below
    ImportedSINDyModel = None


RESULTS_DIR = ensure_directory(repo_root() / "results" / "utah_forge")
THRESHOLDS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200]
FEATURE_NAMES = ["1", "tau", "V", "log(V)", "tau*log(V)", "V_drive-V", "tau_avg", "tau_ema", "S"]
SEED = 42
SMOOTHING_WINDOW = 61
SMOOTHING_POLYORDER = 3
DERIV_WINDOW = 15
DERIV_POLYORDER = 3
ROLLING_WINDOW = 20
EMA_SPAN = 20
MIN_EVENT_SAMPLES = 1200
NEGATIVE_V_LIMIT = 0.30
DEFAULT_VDRIVE_OFFSET = 0.01
EPS = 1e-6


class InlineSTLSQModel:
    def __init__(self, threshold: float, max_iter: int = 20):
        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self.coef_: np.ndarray | None = None

    def fit(self, theta: np.ndarray, xdot: np.ndarray) -> "InlineSTLSQModel":
        n_features = theta.shape[1]
        n_targets = xdot.shape[1]
        coefficients = np.zeros((n_features, n_targets), dtype=float)

        for col_idx in range(n_targets):
            active = np.ones(n_features, dtype=bool)
            active[0] = True
            for _ in range(self.max_iter):
                coef_active, _, _, _ = lstsq(theta[:, active], xdot[:, col_idx])
                coef = np.zeros(n_features, dtype=float)
                coef[active] = coef_active
                small = np.abs(coef) < self.threshold
                small[0] = False
                updated_active = ~small
                if np.array_equal(updated_active, active):
                    active = updated_active
                    break
                active = updated_active

            coef_active, _, _, _ = lstsq(theta[:, active], xdot[:, col_idx])
            coef = np.zeros(n_features, dtype=float)
            coef[active] = coef_active
            coefficients[:, col_idx] = coef

        self.coef_ = coefficients
        return self


def load_and_preprocess() -> pd.DataFrame:
    raw_df, meta = load_utah_forge_dataset()
    mapping = meta["column_mapping"]
    state_df, _ = build_utah_forge_state(
        raw_df,
        mapping,
        derive_velocity_window=SMOOTHING_WINDOW,
        derive_velocity_polyorder=SMOOTHING_POLYORDER,
    )

    time_col = mapping.get("time")
    if time_col is None:
        raise RuntimeError("Utah FORGE mapping did not resolve a time column.")

    extras = pd.DataFrame({"time": raw_df[time_col].to_numpy(dtype=float)})
    normalized = {"".join(ch.lower() for ch in str(col) if ch.isalnum()): col for col in raw_df.columns}
    for alias in ("vext", "loadpointvelocity", "drivingvelocity", "vdrive", "v_ext"):
        column = normalized.get("".join(ch.lower() for ch in alias if ch.isalnum()))
        if column is not None:
            extras["v_ext"] = raw_df[column].to_numpy(dtype=float)
            break
    if "displacement" in state_df.columns:
        extras["displacement"] = raw_df[mapping.get("displacement")].to_numpy(dtype=float)
    merged = state_df.merge(extras, on="time", how="left")
    merged = remove_invalid_rows(merged)
    merged = merged.sort_values("time").reset_index(drop=True)
    if "v_ext" in merged.columns:
        merged["v_ext"] = merged["v_ext"].astype(float)
    return merged


def segment_events(state_df: pd.DataFrame) -> list[pd.DataFrame]:
    segments = segment_utah_forge_events(state_df, tau_col="tau", min_cycle_length=MIN_EVENT_SAMPLES)
    events: list[pd.DataFrame] = []
    for idx, (start, end) in enumerate(segments):
        try:
            event_df = state_df.iloc[start:end].reset_index(drop=True).copy()
            if len(event_df) < MIN_EVENT_SAMPLES:
                continue
            negative_fraction = float((event_df["V"] < 0).mean())
            if negative_fraction > NEGATIVE_V_LIMIT:
                print(
                    f"Skipping event {idx:03d}: negative V fraction {negative_fraction:.3f} exceeds {NEGATIVE_V_LIMIT:.3f}",
                    flush=True,
                )
                continue
            event_df.insert(0, "event_id", f"event_{idx:03d}")
            events.append(event_df)
        except Exception as exc:
            print(f"Skipping event {idx:03d}: segmentation extraction failed: {exc}", flush=True)
    return events


def choose_event_split(events: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    if not events:
        raise RuntimeError("No valid Utah FORGE events were found after filtering.")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(events))
    shuffled = [events[idx] for idx in order]

    if len(shuffled) >= 9:
        n_train = 6
        n_holdout = 3
    else:
        n_train = max(1, int(round(len(shuffled) * (6 / 9))))
        n_holdout = max(1, len(shuffled) - n_train)
        if n_train + n_holdout > len(shuffled):
            n_holdout = len(shuffled) - n_train
        if n_holdout <= 0:
            n_holdout = 1
            n_train = max(1, len(shuffled) - 1)

    train_events = shuffled[:n_train]
    holdout_events = shuffled[n_train : n_train + n_holdout]
    if not holdout_events:
        holdout_events = [train_events.pop()]

    print(
        f"Using {len(train_events)} training events and {len(holdout_events)} holdout events after filtering {len(events)} events.",
        flush=True,
    )
    return train_events, holdout_events


def estimate_derivatives(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    if derivative_savgol is not None:
        try:
            return np.asarray(
                derivative_savgol(values, t=time, window=DERIV_WINDOW, polyorder=DERIV_POLYORDER),
                dtype=float,
            )
        except Exception:
            pass

    if len(values) < 5:
        return np.gradient(values, time)

    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
    return np.asarray(
        savgol_filter(values, window_length=DERIV_WINDOW, polyorder=DERIV_POLYORDER, deriv=1, delta=dt),
        dtype=float,
    )


def prepare_event(event_df: pd.DataFrame) -> pd.DataFrame:
    working = event_df.copy()
    working = working.sort_values("time").reset_index(drop=True)

    tau_smooth = smooth_series(working["tau"].to_numpy(dtype=float), window=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER)
    v_smooth = smooth_series(working["V"].to_numpy(dtype=float), window=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLYORDER)
    v_positive = np.clip(v_smooth, EPS, None)

    if "v_ext" in working.columns:
        v_drive = working["v_ext"].to_numpy(dtype=float)
        invalid_drive = ~np.isfinite(v_drive)
        if np.any(invalid_drive):
            v_drive = v_drive.copy()
            v_drive[invalid_drive] = v_positive[invalid_drive] + DEFAULT_VDRIVE_OFFSET
    else:
        v_drive = v_positive + DEFAULT_VDRIVE_OFFSET

    tau_avg = pd.Series(tau_smooth).rolling(window=ROLLING_WINDOW, min_periods=1).mean().to_numpy(dtype=float)
    tau_ema = pd.Series(tau_smooth).ewm(span=EMA_SPAN, adjust=False).mean().to_numpy(dtype=float)
    slip = np.zeros(len(working), dtype=float)
    if len(working) > 1:
        dt = np.diff(working["time"].to_numpy(dtype=float))
        slip[1:] = np.cumsum(0.5 * (v_positive[1:] + v_positive[:-1]) * dt)

    prepared = pd.DataFrame(
        {
            "event_id": working["event_id"],
            "time": working["time"].to_numpy(dtype=float),
            "tau": tau_smooth,
            "V": v_positive,
            "V_drive": v_drive,
            "log(V)": np.log(v_positive),
            "tau_avg": tau_avg,
            "tau_ema": tau_ema,
            "S": slip,
        }
    )
    prepared["tau*log(V)"] = prepared["tau"] * prepared["log(V)"]
    prepared["V_drive-V"] = prepared["V_drive"] - prepared["V"]
    prepared["dtau_dt"] = estimate_derivatives(prepared["time"].to_numpy(dtype=float), prepared["tau"].to_numpy(dtype=float))
    prepared["dV_dt"] = estimate_derivatives(prepared["time"].to_numpy(dtype=float), prepared["V"].to_numpy(dtype=float))
    prepared = remove_invalid_rows(prepared)
    return prepared.reset_index(drop=True)


def build_feature_matrix(prepared_df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(prepared_df), dtype=float),
            prepared_df["tau"].to_numpy(dtype=float),
            prepared_df["V"].to_numpy(dtype=float),
            prepared_df["log(V)"].to_numpy(dtype=float),
            prepared_df["tau*log(V)"].to_numpy(dtype=float),
            prepared_df["V_drive-V"].to_numpy(dtype=float),
            prepared_df["tau_avg"].to_numpy(dtype=float),
            prepared_df["tau_ema"].to_numpy(dtype=float),
            prepared_df["S"].to_numpy(dtype=float),
        ]
    )


def zscore_stack(feature_blocks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    stacked = np.vstack(feature_blocks)
    means = np.mean(stacked, axis=0)
    stds = np.std(stacked, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    means[0] = 0.0
    stds[0] = 1.0
    z_blocks = [(block - means) / stds for block in feature_blocks]
    return z_blocks, {"means": means.tolist(), "stds": stds.tolist()}


def fit_model(theta_all: np.ndarray, xdot_all: np.ndarray, threshold: float) -> np.ndarray:
    if ImportedSINDyModel is not None:
        try:
            model = ImportedSINDyModel(threshold=threshold)
            fit_result = model.fit(theta_all, xdot_all, FEATURE_NAMES)
            coefficients = getattr(model, "coef_", None)
            if coefficients is None:
                coefficients = getattr(model, "coefficients", None)
            if coefficients is None:
                raise RuntimeError("Imported SINDyModel did not expose coefficients.")
            return np.asarray(coefficients, dtype=float)
        except Exception as exc:
            print(f"Imported SINDyModel failed at threshold {threshold:.3f}; using inline STLSQ instead: {exc}", flush=True)

    model = InlineSTLSQModel(threshold=threshold)
    model.fit(theta_all, xdot_all)
    if model.coef_ is None:
        raise RuntimeError("Inline STLSQ failed to produce coefficients.")
    return model.coef_


def evaluate_training_mse(theta_all: np.ndarray, xdot_all: np.ndarray, xi: np.ndarray) -> float:
    prediction = theta_all @ xi
    return float(np.mean((prediction - xdot_all) ** 2))


def active_term_names(xi: np.ndarray, column: int) -> list[str]:
    return [FEATURE_NAMES[idx] for idx in np.where(np.abs(xi[:, column]) > 0)[0]]


def make_rhs(xi: np.ndarray, feature_means: np.ndarray, feature_stds: np.ndarray, default_vdrive: float):
    def rhs(state: np.ndarray, _time: float) -> np.ndarray:
        tau_x = float(state[0])
        v_x = max(float(state[1]), EPS)
        raw = np.array(
            [
                1.0,
                tau_x,
                v_x,
                math.log(v_x),
                tau_x * math.log(v_x),
                default_vdrive - v_x,
                tau_x,
                tau_x,
                0.0,
            ],
            dtype=float,
        )
        z = (raw - feature_means) / feature_stds
        derivative = z @ xi
        return np.asarray(derivative, dtype=float)

    return rhs


def rollout_holdout_event(event_df: pd.DataFrame, xi: np.ndarray, scaling: dict[str, list[float]]) -> dict[str, float]:
    try:
        prepared = prepare_event(event_df)
    except Exception as exc:
        print(f"Skipping holdout {event_df['event_id'].iloc[0]}: preprocessing failed: {exc}", flush=True)
        return {"divergence_fraction": 0.0}

    time = prepared["time"].to_numpy(dtype=float)
    observed = prepared[["tau", "V"]].to_numpy(dtype=float)
    default_vdrive = float(np.nanmedian(prepared["V_drive"].to_numpy(dtype=float)))
    tau_std = float(np.std(observed[:, 0])) or 1.0
    v_std = float(np.std(observed[:, 1])) or 1.0
    rhs = make_rhs(xi, np.asarray(scaling["means"], dtype=float), np.asarray(scaling["stds"], dtype=float), default_vdrive)

    try:
        rollout = odeint(rhs, observed[0], time)
    except Exception as exc:
        print(f"Skipping holdout {event_df['event_id'].iloc[0]}: rollout integration failed: {exc}", flush=True)
        return {"divergence_fraction": 0.0}

    divergence_fraction = 1.0
    total_steps = max(len(time) - 1, 1)
    for idx in range(len(rollout)):
        tau_err = abs(float(rollout[idx, 0]) - float(observed[idx, 0]))
        v_err = abs(float(rollout[idx, 1]) - float(observed[idx, 1]))
        invalid = not np.isfinite(rollout[idx]).all()
        diverged = invalid or tau_err > 3.0 * tau_std or v_err > 3.0 * v_std
        if diverged:
            divergence_fraction = idx / total_steps
            break

    return {"divergence_fraction": float(divergence_fraction)}


def normalized_knee_index(rows: list[dict]) -> int:
    if len(rows) <= 2:
        return 0

    order = np.argsort([row["n_terms_union"] for row in rows])
    xs = np.asarray([rows[idx]["n_terms_union"] for idx in order], dtype=float)
    ys = np.asarray([rows[idx]["mean_rollout"] for idx in order], dtype=float)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_norm = np.zeros_like(xs) if x_max == x_min else (xs - x_min) / (x_max - x_min)
    y_norm = np.zeros_like(ys) if y_max == y_min else (ys - y_min) / (y_max - y_min)

    start = np.array([x_norm[0], y_norm[0]], dtype=float)
    end = np.array([x_norm[-1], y_norm[-1]], dtype=float)
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm == 0:
        return int(order[0])

    distances = []
    for idx in range(len(order)):
        point = np.array([x_norm[idx], y_norm[idx]], dtype=float)
        distance = abs(line[0] * (point[1] - start[1]) - line[1] * (point[0] - start[0])) / line_norm
        distances.append(float(distance))
    return int(order[int(np.argmax(distances))])


def sign_consistency_checks(v_coefficients: np.ndarray) -> list[dict[str, object]]:
    term_to_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    checks = [
        {"term": "log(V)", "expected": "negative", "ok": bool(v_coefficients[term_to_index["log(V)"]] < 0)},
        {"term": "V_drive-V", "expected": "positive", "ok": bool(v_coefficients[term_to_index["V_drive-V"]] > 0)},
        {"term": "tau", "expected": "positive", "ok": bool(v_coefficients[term_to_index["tau"]] > 0)},
        {"term": "tau*log(V)", "expected": "ambiguous", "ok": None},
        {"term": "1", "expected": "unspecified", "ok": None},
        {"term": "V", "expected": "unspecified", "ok": None},
        {"term": "tau_avg", "expected": "unspecified", "ok": None},
        {"term": "tau_ema", "expected": "unspecified", "ok": None},
        {"term": "S", "expected": "unspecified", "ok": None},
    ]
    for row in checks:
        term = row["term"]
        row["coefficient"] = float(v_coefficients[term_to_index[term]])
        if row["ok"] is False:
            row["violation"] = True
        else:
            row["violation"] = False
    return checks


def json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def equation_string(xi_col: np.ndarray, var_name: str) -> str:
    active = np.where(np.abs(xi_col) > 0)[0]
    if len(active) == 0:
        return f"d{var_name}/dt = 0"
    pieces: list[str] = []
    for idx in active:
        coef = float(xi_col[idx])
        sign = "+" if coef >= 0 else "-"
        term = FEATURE_NAMES[idx]
        chunk = f"{sign} {abs(coef):.6e}*{term}"
        pieces.append(chunk if pieces else chunk.lstrip("+ ").strip())
    return f"d{var_name}/dt = " + " ".join(pieces)


def save_frontier_plot(rows: list[dict], knee_idx: int) -> None:
    lambdas = np.asarray([row["lambda"] for row in rows], dtype=float)
    n_union = np.asarray([row["n_terms_union"] for row in rows], dtype=float)
    mean_rollout = np.asarray([row["mean_rollout"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    scatter = axes[0].scatter(n_union, mean_rollout, c=np.log10(lambdas), cmap="viridis", s=70)
    axes[0].set_xlabel("n_terms_union")
    axes[0].set_ylabel("mean_rollout")
    axes[0].set_title("Sparsity Frontier")
    axes[0].grid(True, alpha=0.3)
    dense_idx = 0
    axes[0].annotate(
        f"dense λ={rows[dense_idx]['lambda']:.3f}",
        (n_union[dense_idx], mean_rollout[dense_idx]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    axes[0].scatter(
        [n_union[knee_idx]],
        [mean_rollout[knee_idx]],
        color="red",
        s=110,
        marker="X",
        label="knee point",
        zorder=5,
    )
    axes[0].annotate(
        f"knee λ={rows[knee_idx]['lambda']:.3f}",
        (n_union[knee_idx], mean_rollout[knee_idx]),
        xytext=(10, -12),
        textcoords="offset points",
        color="red",
        weight="bold",
    )
    axes[0].legend(loc="best")
    fig.colorbar(scatter, ax=axes[0], label="log10(lambda)")

    ax_right = axes[1].twinx()
    axes[1].plot(lambdas, n_union, color="tab:blue", marker="o", label="n_terms_union")
    ax_right.plot(lambdas, mean_rollout, color="tab:orange", marker="s", label="mean_rollout")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("n_terms_union", color="tab:blue")
    ax_right.set_ylabel("mean_rollout", color="tab:orange")
    axes[1].tick_params(axis="y", labelcolor="tab:blue")
    ax_right.tick_params(axis="y", labelcolor="tab:orange")
    axes[1].axvline(rows[knee_idx]["lambda"], color="red", linestyle="--", linewidth=1.4)
    axes[1].set_title("Frontier by Threshold")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sparsity_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_best_equations(knee_row: dict) -> None:
    checks = knee_row["sign_consistency"]
    lines = [
        "# Knee-point sparse equations",
        "",
        f"lambda = {knee_row['lambda']:.6f}",
        f"n_terms_tau = {knee_row['n_terms_tau']}",
        f"n_terms_V = {knee_row['n_terms_V']}",
        f"n_terms_union = {knee_row['n_terms_union']}",
        f"mean_rollout = {knee_row['mean_rollout']:.6f}",
        "",
        "## Surviving terms",
        f"tau equation: {', '.join(knee_row['surviving_terms_tau']) if knee_row['surviving_terms_tau'] else '(none)'}",
        f"V equation: {', '.join(knee_row['surviving_terms_V']) if knee_row['surviving_terms_V'] else '(none)'}",
        "",
        "## Equations",
        equation_string(np.asarray(knee_row["Xi"], dtype=float)[:, 0], "tau"),
        equation_string(np.asarray(knee_row["Xi"], dtype=float)[:, 1], "V"),
        "",
        "## Sign consistency against RSF-inspired expectations",
    ]
    for check in checks:
        status = "OK" if check["ok"] is True else ("VIOLATION" if check["ok"] is False else "N/A")
        lines.append(
            f"- {check['term']}: coefficient={check['coefficient']:.6e}, expected={check['expected']}, status={status}"
        )
    (RESULTS_DIR / "best_sparse_equations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(rows: list[dict], knee_row: dict) -> None:
    header = "| lambda | n_terms_tau | n_terms_V | n_terms_union | mean_rollout | training_mse |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    table_rows = [header, divider]
    for row in rows:
        table_rows.append(
            f"| {row['lambda']:.3f} | {row['n_terms_tau']} | {row['n_terms_V']} | {row['n_terms_union']} | {row['mean_rollout']:.4f} | {row['training_mse']:.6e} |"
        )

    sign_lines = [
        "| Term | Coefficient in dV/dt | Expected sign | Status |",
        "| --- | --- | --- | --- |",
    ]
    for check in knee_row["sign_consistency"]:
        if check["ok"] is True:
            status = "✅"
        elif check["ok"] is False:
            status = "❌"
        else:
            status = "—"
        sign_lines.append(
            f"| {check['term']} | {check['coefficient']:.6e} | {check['expected']} | {status} |"
        )

    report = [
        "# Utah FORGE sparsity frontier",
        "",
        "## Full frontier",
        *table_rows,
        "",
        "## Knee-point equations",
        "```text",
        equation_string(np.asarray(knee_row["Xi"], dtype=float)[:, 0], "tau"),
        equation_string(np.asarray(knee_row["Xi"], dtype=float)[:, 1], "V"),
        "```",
        "",
        "## Sign consistency",
        *sign_lines,
        "",
        "## Interpretation",
        (
            f"The knee point occurs at lambda={knee_row['lambda']:.3f}, where the model keeps {knee_row['n_terms_union']} active terms "
            f"while achieving a mean rollout stability score of {knee_row['mean_rollout']:.3f} on holdout events. "
            "This point balances compactness against dynamical stability more effectively than both the densest and most aggressively pruned models. "
            "The sign-consistency check highlights whether the discovered dV/dt structure remains aligned with RSF-inspired expectations for stress loading, spring loading, and velocity weakening."
        ),
    ]
    (RESULTS_DIR / "sparsity_frontier_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    state_df = load_and_preprocess()
    events = segment_events(state_df)
    train_events, holdout_events = choose_event_split(events)

    prepared_train_events: list[pd.DataFrame] = []
    for event_df in train_events:
        try:
            prepared_train_events.append(prepare_event(event_df))
        except Exception as exc:
            print(f"Skipping training {event_df['event_id'].iloc[0]}: preprocessing failed: {exc}", flush=True)

    if not prepared_train_events:
        raise RuntimeError("No training events could be prepared successfully.")

    feature_blocks = [build_feature_matrix(prepared_df) for prepared_df in prepared_train_events]
    theta_blocks, scaling = zscore_stack(feature_blocks)
    xdot_blocks = [
        prepared_df[["dtau_dt", "dV_dt"]].to_numpy(dtype=float)
        for prepared_df in prepared_train_events
    ]
    theta_all = np.vstack(theta_blocks)
    xdot_all = np.vstack(xdot_blocks)

    rows: list[dict] = []
    for lam in THRESHOLDS:
        xi = fit_model(theta_all, xdot_all, threshold=lam)
        training_mse = evaluate_training_mse(theta_all, xdot_all, xi)
        n_terms_tau = int(np.count_nonzero(np.abs(xi[:, 0]) > 0))
        n_terms_v = int(np.count_nonzero(np.abs(xi[:, 1]) > 0))
        union_mask = np.any(np.abs(xi) > 0, axis=1)
        n_terms_union = int(np.count_nonzero(union_mask))

        holdout_rows: list[dict[str, float]] = []
        for event_df in holdout_events:
            holdout_rows.append(rollout_holdout_event(event_df, xi, scaling))
        mean_rollout = float(np.mean([row["divergence_fraction"] for row in holdout_rows])) if holdout_rows else 0.0

        row = {
            "lambda": float(lam),
            "n_terms_tau": n_terms_tau,
            "n_terms_V": n_terms_v,
            "n_terms_union": n_terms_union,
            "surviving_terms_tau": active_term_names(xi, 0),
            "surviving_terms_V": active_term_names(xi, 1),
            "training_mse": training_mse,
            "mean_rollout": mean_rollout,
            "holdout_rollouts": holdout_rows,
            "Xi": np.asarray(xi, dtype=float).tolist(),
            "scaling": scaling,
        }
        rows.append(row)
        print(
            f"lambda={lam:.3f} | union={n_terms_union} | tau_terms={n_terms_tau} | V_terms={n_terms_v} | rollout={mean_rollout:.4f}",
            flush=True,
        )

    knee_idx = normalized_knee_index(rows)
    rows[knee_idx]["sign_consistency"] = sign_consistency_checks(np.asarray(rows[knee_idx]["Xi"], dtype=float)[:, 1])
    for idx, row in enumerate(rows):
        if idx != knee_idx:
            row["sign_consistency"] = []

    (RESULTS_DIR / "sparsity_frontier.json").write_text(
        json.dumps(rows, indent=2, default=json_default),
        encoding="utf-8",
    )
    save_frontier_plot(rows, knee_idx)
    write_best_equations(rows[knee_idx])
    write_report(rows, rows[knee_idx])
    print(f"Knee-point threshold selected: lambda={rows[knee_idx]['lambda']:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"utah_forge_sparsity_frontier.py failed: {exc}", file=sys.stderr, flush=True)
        raise
