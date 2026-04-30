from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

from scripts import improve_utah_forge_model as base
from scripts import refine_utah_forge_validation as delay_ref
from scripts import utah_forge_memory_refinement as memory_ref
from scripts import utah_forge_proposal_equation_recovery as recovery
from scripts import utah_forge_reviewer_ablation as reviewer_ablation
from src.derivatives import derivative_savgol
from src.preprocess.common import smooth_series
from src.utils.paths import ensure_directory


EPS = 1e-8


@dataclass
class ExactRSFSegment:
    step_name: str
    split: str
    time: np.ndarray
    tau: np.ndarray
    V: np.ndarray
    V_drive: np.ndarray
    sigmaN: np.ndarray
    theta_proxy: np.ndarray
    acoustic: np.ndarray
    acoustic_name: str
    acoustic_event_value: float
    dtau_dt: np.ndarray
    dV_dt: np.ndarray
    tau0: float
    V_init: float
    sigma0: float
    tau_scale: float
    v_scale: float
    V0_ref: float
    Dc_ref: float
    mu0_ref: float
    a_ref: float
    b_ref: float


def timing_start(context: str, stage: str) -> float:
    print(f"[timing:{context}] start {stage}", flush=True)
    return time.perf_counter()


def timing_end(context: str, stage: str, started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    print(f"[timing:{context}] end {stage} elapsed_s={elapsed:.3f}", flush=True)
    return elapsed


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_checkpoint(checkpoint_dir: Path, stage_name: str, payload, summary: dict | None = None) -> None:
    ensure_directory(checkpoint_dir)
    payload_path = checkpoint_dir / f"{stage_name}.pkl"
    meta_path = checkpoint_dir / f"{stage_name}.json"
    pd.to_pickle(payload, payload_path)
    meta_path.write_text(
        json.dumps(
            {
                "stage": stage_name,
                "updated_epoch_s": time.time(),
                "summary": json_ready(summary or {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(checkpoint_dir: Path, stage_name: str):
    payload_path = checkpoint_dir / f"{stage_name}.pkl"
    if not payload_path.exists():
        return None
    return pd.read_pickle(payload_path)


def load_workflow_context() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, delay_ref.RSFitStep], dict]:
    state_df, _ = recovery.load_enriched_utah_forge_state()
    inventory_df, segments, steps = memory_ref.segment_step_windows(state_df)
    rsfit_globals = reviewer_ablation.load_rsfit_globals()
    return inventory_df, segments, steps, rsfit_globals


def split_segments(
    inventory_df: pd.DataFrame,
    segments: dict[str, pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str], list[str]]:
    return memory_ref.split_train_holdout_segments(inventory_df, segments)


def choose_acoustic_name(state_columns: list[str]) -> str | None:
    for column in recovery.ACOUSTIC_CANDIDATES:
        if column in state_columns:
            return column
    return None


def _constrained_sparse(
    design: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    threshold: float,
    bounds: dict[str, tuple[float, float]],
    mandatory_terms: set[str],
) -> np.ndarray:
    active = np.ones(len(feature_names), dtype=bool)
    mandatory_mask = np.array([name in mandatory_terms for name in feature_names], dtype=bool)
    lower = np.array([bounds.get(name, (-np.inf, np.inf))[0] for name in feature_names], dtype=float)
    upper = np.array([bounds.get(name, (-np.inf, np.inf))[1] for name in feature_names], dtype=float)
    for _ in range(20):
        result = lsq_linear(design[:, active], target, bounds=(lower[active], upper[active]), method="trf", lsmr_tol="auto")
        coefficients = np.zeros(len(feature_names), dtype=float)
        coefficients[active] = result.x
        small = np.abs(coefficients) < threshold
        small[mandatory_mask] = False
        updated_active = ~small
        if np.array_equal(updated_active, active):
            active = updated_active
            break
        active = updated_active
    result = lsq_linear(design[:, active], target, bounds=(lower[active], upper[active]), method="trf", lsmr_tol="auto")
    coefficients = np.zeros(len(feature_names), dtype=float)
    coefficients[active] = result.x
    return coefficients


def prepare_exact_segment(
    segment_df: pd.DataFrame,
    step: delay_ref.RSFitStep,
    rsfit_globals: dict,
    acoustic_name: str | None,
    split_name: str,
    *,
    max_points: int = 900,
    smoothing_window: int = 61,
) -> ExactRSFSegment:
    requested = ["step_name", "time", "tau", "V", "V_drive"]
    if acoustic_name and acoustic_name in segment_df.columns:
        requested.append(acoustic_name)
    working = segment_df[requested].copy()
    working = base.downsample_frame(working, max_points=max_points)
    working = base.enforce_monotonic_time(working)

    time_values = working["time"].to_numpy(dtype=float)
    tau = smooth_series(working["tau"].to_numpy(dtype=float), window=smoothing_window, polyorder=3)
    velocity = smooth_series(working["V"].to_numpy(dtype=float), window=smoothing_window, polyorder=3)
    velocity = np.clip(velocity, EPS, None)
    dtau_dt = derivative_savgol(tau, t=time_values, window=15, polyorder=3)
    dV_dt = derivative_savgol(velocity, t=time_values, window=15, polyorder=3)
    sigma = np.interp(time_values, rsfit_globals["time"], rsfit_globals["sigmaN"])
    sigma = np.clip(sigma, EPS, None)
    theta_proxy = np.interp(time_values, step.time, step.theta_eff)
    theta_proxy = np.clip(theta_proxy, EPS, None)
    acoustic = np.full(len(time_values), np.nan, dtype=float)
    if acoustic_name and acoustic_name in working.columns:
        acoustic = recovery.fill_missing_1d(working[acoustic_name].to_numpy(dtype=float))
        acoustic = smooth_series(acoustic, window=smoothing_window, polyorder=3)
    acoustic_event_value = float(np.nanmean(acoustic)) if np.isfinite(acoustic).any() else float("nan")
    params = reviewer_ablation.effective_step_params(step)
    return ExactRSFSegment(
        step_name=str(segment_df["step_name"].iloc[0]),
        split=split_name,
        time=time_values,
        tau=tau,
        V=velocity,
        V_drive=working["V_drive"].to_numpy(dtype=float),
        sigmaN=sigma,
        theta_proxy=theta_proxy,
        acoustic=acoustic,
        acoustic_name=acoustic_name or "",
        acoustic_event_value=acoustic_event_value,
        dtau_dt=dtau_dt,
        dV_dt=dV_dt,
        tau0=float(tau[0]),
        V_init=float(velocity[0]),
        sigma0=float(sigma[0]),
        tau_scale=float(np.std(tau) + 1e-6),
        v_scale=float(np.std(velocity) + 1e-6),
        V0_ref=float(params["V0"]),
        Dc_ref=float(params["Dc"]),
        mu0_ref=float(params["mu0"]),
        a_ref=float(params["a"]),
        b_ref=float(params["b"]),
    )


def prepare_exact_segments(
    train_segments: list[pd.DataFrame],
    holdout_segments: list[pd.DataFrame],
    steps: dict[str, delay_ref.RSFitStep],
    rsfit_globals: dict,
) -> tuple[list[ExactRSFSegment], list[ExactRSFSegment], str | None]:
    all_columns = list(pd.concat(train_segments + holdout_segments, ignore_index=True).columns)
    acoustic_name = choose_acoustic_name(all_columns)
    prepared_train = [prepare_exact_segment(segment_df, steps[str(segment_df["step_name"].iloc[0])], rsfit_globals, acoustic_name, "train") for segment_df in train_segments]
    prepared_holdout = [prepare_exact_segment(segment_df, steps[str(segment_df["step_name"].iloc[0])], rsfit_globals, acoustic_name, "holdout") for segment_df in holdout_segments]
    return prepared_train, prepared_holdout, acoustic_name


def sparse_structure_confirmation(train_segments: list[ExactRSFSegment], acoustic_name: str | None) -> dict:
    context = "sparse_structure_confirmation"
    started = timing_start(context, "overall")
    tau_features = []
    tau_targets = []
    for segment in train_segments:
        tau_features.append(
            np.column_stack(
                [
                    np.ones(len(segment.time)),
                    segment.V,
                    segment.V_drive - segment.V,
                    segment.tau,
                ]
            )
        )
        tau_targets.append(segment.dtau_dt)
    tau_design = np.vstack(tau_features)
    tau_target = np.concatenate(tau_targets)
    tau_names = ["1", "V", "V_drive_minus_V", "tau"]
    tau_coeffs = _constrained_sparse(
        tau_design,
        tau_target,
        tau_names,
        threshold=1e-4,
        bounds={"V_drive_minus_V": (0.0, np.inf)},
        mandatory_terms={"V_drive_minus_V"},
    )

    velocity_features = []
    velocity_targets = []
    for segment in train_segments:
        columns = [
            np.ones(len(segment.time)),
            segment.tau,
            segment.sigmaN,
            segment.sigmaN * np.log(np.clip(segment.V / segment.V0_ref, EPS, None)),
            segment.sigmaN * np.log(np.clip(segment.theta_proxy * segment.V0_ref / segment.Dc_ref, EPS, None)),
        ]
        names = ["1", "tau", "sigmaN", "sigmaN_logV", "sigmaN_logTheta_proxy"]
        if acoustic_name and np.isfinite(segment.acoustic).any():
            columns.append(segment.acoustic)
            names.append("acoustic_feature")
        velocity_features.append(np.column_stack(columns))
        velocity_targets.append(segment.dV_dt)
    velocity_design = np.vstack(velocity_features)
    velocity_target = np.concatenate(velocity_targets)
    velocity_coeffs = _constrained_sparse(
        velocity_design,
        velocity_target,
        names,
        threshold=1e-4,
        bounds={
            "tau": (0.0, np.inf),
            "sigmaN": (-np.inf, 0.0),
            "sigmaN_logV": (-np.inf, 0.0),
            "sigmaN_logTheta_proxy": (-np.inf, 0.0),
        },
        mandatory_terms={"tau", "sigmaN_logV"},
    )
    tau_active = [name for name, value in zip(tau_names, tau_coeffs) if abs(value) > 1e-8]
    v_active = [name for name, value in zip(names, velocity_coeffs) if abs(value) > 1e-8]
    payload = {
        "tau_feature_names": tau_names,
        "tau_coefficients": tau_coeffs.tolist(),
        "tau_active_terms": tau_active,
        "tau_spring_loading_confirmed": bool("V_drive_minus_V" in tau_active and tau_coeffs[tau_names.index("V_drive_minus_V")] > 0),
        "velocity_feature_names": names,
        "velocity_coefficients": velocity_coeffs.tolist(),
        "velocity_active_terms": v_active,
        "velocity_tau_confirmed": bool("tau" in v_active),
        "velocity_logv_confirmed": bool("sigmaN_logV" in v_active),
        "velocity_hidden_state_evidence": bool("sigmaN_logTheta_proxy" in v_active),
        "acoustic_feature_considered": acoustic_name,
    }
    timing_end(context, "overall", started)
    return payload


def observed_consistent_theta0(params: dict[str, float], segment: ExactRSFSegment) -> float:
    mu_obs = segment.tau0 / max(segment.sigma0, EPS)
    base = (params["Dc"] / max(segment.V0_ref, EPS)) * math.exp(
        (mu_obs - params["mu0"] - params["a"] * math.log(max(segment.V_init, EPS) / max(segment.V0_ref, EPS))) / max(params["b"], EPS)
    )
    return float(np.clip(base, 1e-6, 1e8))


def unpack_parameters(
    vector: np.ndarray,
    train_segments: list[ExactRSFSegment],
    *,
    use_acoustic: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    globals_dict = {
        "k": float(vector[0]),
        "m": float(vector[1]),
        "mu0": float(vector[2]),
        "a": float(vector[3]),
        "b": float(vector[4]),
        "Dc": float(vector[5]),
    }
    offset_start = 6
    acoustic_gamma = 0.0
    if use_acoustic:
        acoustic_gamma = float(vector[offset_start])
        offset_start += 1
    globals_dict["acoustic_gamma"] = acoustic_gamma
    offsets = {segment.step_name: float(vector[offset_start + index]) for index, segment in enumerate(train_segments)}
    return globals_dict, offsets


def pack_initial_vector(
    train_segments: list[ExactRSFSegment],
    *,
    use_acoustic: bool,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[str]]:
    k0 = 8.5e-3
    mu0 = float(np.median([segment.mu0_ref for segment in train_segments]))
    a0 = float(np.median([segment.a_ref for segment in train_segments]))
    b0 = float(np.median([segment.b_ref for segment in train_segments]))
    dc0 = float(np.median([segment.Dc_ref for segment in train_segments]))
    m0 = 1.0
    names = ["k", "m", "mu0", "a", "b", "Dc"]
    initial = [k0, m0, mu0, a0, b0, dc0]
    lower = [1e-6, 1e-3, 0.0, 1e-5, 1e-5, 1e-4]
    upper = [1.0, 1e3, 2.0, 1.0, 1.0, 1e3]
    if use_acoustic:
        initial.append(0.0)
        lower.append(-3.0)
        upper.append(3.0)
        names.append("acoustic_gamma")
    for segment in train_segments:
        initial.append(0.0)
        lower.append(-1.5)
        upper.append(1.5)
        names.append(f"delta_log_theta0:{segment.step_name}")
    return np.array(initial, dtype=float), (np.array(lower, dtype=float), np.array(upper, dtype=float)), names


def simulate_exact_rsf_segment(
    segment: ExactRSFSegment,
    params: dict[str, float],
    *,
    delta_log_theta0: float = 0.0,
    acoustic_z: float = 0.0,
) -> dict:
    n = len(segment.time)
    tau_sim = np.zeros(n, dtype=float)
    v_sim = np.zeros(n, dtype=float)
    theta_sim = np.zeros(n, dtype=float)
    tau_sim[0] = segment.tau0
    v_sim[0] = max(segment.V_init, EPS)
    theta0 = observed_consistent_theta0(params, segment) * math.exp(delta_log_theta0 + params.get("acoustic_gamma", 0.0) * acoustic_z)
    theta_sim[0] = max(theta0, EPS)
    feasible = True

    for index in range(n - 1):
        dt = float(segment.time[index + 1] - segment.time[index])
        tau_now = float(tau_sim[index])
        v_now = max(float(v_sim[index]), EPS)
        theta_now = max(float(theta_sim[index]), EPS)
        sigma_now = float(segment.sigmaN[index])
        drive_now = float(segment.V_drive[index])
        friction = params["mu0"] + params["a"] * math.log(v_now / max(segment.V0_ref, EPS)) + params["b"] * math.log(theta_now * max(segment.V0_ref, EPS) / max(params["Dc"], EPS))
        tau_dot = params["k"] * (drive_now - v_now)
        v_dot = (tau_now - sigma_now * friction) / max(params["m"], EPS)
        theta_dot = 1.0 - v_now * theta_now / max(params["Dc"], EPS)
        tau_next = tau_now + dt * tau_dot
        v_next = max(v_now + dt * v_dot, EPS)
        theta_next = max(theta_now + dt * theta_dot, EPS)
        tau_sim[index + 1] = tau_next
        v_sim[index + 1] = v_next
        theta_sim[index + 1] = theta_next
        if not (np.isfinite(tau_next) and np.isfinite(v_next) and np.isfinite(theta_next)):
            feasible = False
            tau_sim[index + 1 :] = np.nan
            v_sim[index + 1 :] = np.nan
            theta_sim[index + 1 :] = np.nan
            break
    return {
        "tau": tau_sim,
        "V": v_sim,
        "theta": theta_sim,
        "theta0": float(theta_sim[0]),
        "feasible": feasible and np.isfinite(tau_sim).all() and np.isfinite(v_sim).all() and np.isfinite(theta_sim).all(),
    }


def event_acoustic_zscores(train_segments: list[ExactRSFSegment], all_segments: list[ExactRSFSegment]) -> dict[str, float]:
    train_values = np.array([segment.acoustic_event_value for segment in train_segments if np.isfinite(segment.acoustic_event_value)], dtype=float)
    if len(train_values) == 0:
        return {segment.step_name: 0.0 for segment in all_segments}
    mean = float(np.mean(train_values))
    std = float(np.std(train_values)) or 1.0
    return {
        segment.step_name: (float(segment.acoustic_event_value) - mean) / std if np.isfinite(segment.acoustic_event_value) else 0.0
        for segment in all_segments
    }


def build_train_residual_vector(
    vector: np.ndarray,
    train_segments: list[ExactRSFSegment],
    acoustic_z: dict[str, float],
    *,
    use_acoustic: bool,
    derivative_weight: float = 0.05,
    theta_offset_weight: float = 0.05,
) -> np.ndarray:
    params, offsets = unpack_parameters(vector, train_segments, use_acoustic=use_acoustic)
    residual_parts = []
    for segment in train_segments:
        sim = simulate_exact_rsf_segment(
            segment,
            params,
            delta_log_theta0=offsets.get(segment.step_name, 0.0),
            acoustic_z=acoustic_z.get(segment.step_name, 0.0),
        )
        if not sim["feasible"]:
            penalty = np.full(max(20, len(segment.time) // 4), 50.0, dtype=float)
            residual_parts.append(penalty)
            continue
        tau_resid = (sim["tau"] - segment.tau) / segment.tau_scale
        v_resid = (sim["V"] - segment.V) / segment.v_scale
        dv_sim = derivative_savgol(sim["V"], t=segment.time, window=15, polyorder=3)
        dv_resid = derivative_weight * (dv_sim - segment.dV_dt) / (np.std(segment.dV_dt) + 1e-6)
        offset_resid = np.array([theta_offset_weight * offsets.get(segment.step_name, 0.0)], dtype=float)
        residual_parts.extend([tau_resid, v_resid, dv_resid, offset_resid])
    return np.concatenate(residual_parts)


def fit_exact_rsf_inverse_model(
    train_segments: list[ExactRSFSegment],
    holdout_segments: list[ExactRSFSegment],
    *,
    use_acoustic: bool,
    checkpoint_dir: Path,
    stage_name: str,
    max_nfev: int = 80,
    initial_vector: np.ndarray | None = None,
) -> dict:
    context = f"fit_exact_rsf_inverse_model:{stage_name}"
    overall_started = timing_start(context, "overall")
    cached = load_checkpoint(checkpoint_dir, stage_name)
    if cached is not None:
        print(f"[resume] loaded {stage_name} checkpoint", flush=True)
        timing_end(context, "overall", overall_started)
        return cached

    acoustic_z = event_acoustic_zscores(train_segments, train_segments + holdout_segments)
    default_initial, bounds, parameter_names = pack_initial_vector(train_segments, use_acoustic=use_acoustic)
    initial = np.asarray(initial_vector, dtype=float) if initial_vector is not None else default_initial
    fit_started = timing_start(context, "least_squares")
    result = least_squares(
        build_train_residual_vector,
        initial,
        bounds=bounds,
        args=(train_segments, acoustic_z),
        kwargs={"use_acoustic": use_acoustic},
        method="trf",
        max_nfev=max_nfev,
        verbose=0,
    )
    timing_end(context, "least_squares", fit_started)
    params, offsets = unpack_parameters(result.x, train_segments, use_acoustic=use_acoustic)

    evaluation_started = timing_start(context, "evaluation")
    train_rows = []
    holdout_rows = []
    per_event_theta0 = {}
    for segment in train_segments:
        sim = simulate_exact_rsf_segment(segment, params, delta_log_theta0=offsets[segment.step_name], acoustic_z=acoustic_z.get(segment.step_name, 0.0))
        row = rollout_metrics(segment, sim)
        row["split"] = "train"
        row["step_name"] = segment.step_name
        train_rows.append(row)
        per_event_theta0[segment.step_name] = float(sim["theta0"])
    for segment in holdout_segments:
        sim = simulate_exact_rsf_segment(segment, params, delta_log_theta0=0.0, acoustic_z=acoustic_z.get(segment.step_name, 0.0))
        row = rollout_metrics(segment, sim)
        row["split"] = "holdout"
        row["step_name"] = segment.step_name
        holdout_rows.append(row)
        per_event_theta0[segment.step_name] = float(sim["theta0"])
    timing_end(context, "evaluation", evaluation_started)

    diagnostics_started = timing_start(context, "identifiability")
    diagnostics = identifiability_diagnostics(result, train_segments, params, parameter_names, acoustic_z, use_acoustic=use_acoustic)
    timing_end(context, "identifiability", diagnostics_started)

    tau_equation = f"dtau/dt = {params['k']:.6e}*(V_drive - V)"
    velocity_equation = (
        "dV/dt = (1/{m:.6e}) * [tau - sigmaN*({mu0:.6e} + {a:.6e}*log(V/V0) + {b:.6e}*log(theta*V0/{Dc:.6e}))]".format(
            m=params["m"],
            mu0=params["mu0"],
            a=params["a"],
            b=params["b"],
            Dc=params["Dc"],
        )
    )
    theta_equation = f"dtheta/dt = 1 - V*theta/{params['Dc']:.6e}"
    payload = {
        "use_acoustic": use_acoustic,
        "parameter_names": parameter_names,
        "parameters": params,
        "theta_offsets_train": offsets,
        "per_event_theta0": per_event_theta0,
        "optimization": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "optimality": float(result.optimality),
        },
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "tau_equation": tau_equation,
        "velocity_equation": velocity_equation,
        "theta_equation": theta_equation,
        "identifiability": diagnostics,
        "acoustic_zscores": acoustic_z,
    }
    save_checkpoint(
        checkpoint_dir,
        stage_name,
        payload,
        {
            "success": payload["optimization"]["success"],
            "holdout_mean_rollout_error": float(np.mean([row["combined_rollout_error"] for row in holdout_rows])),
        },
    )
    timing_end(context, "overall", overall_started)
    return payload


def onset_time(values: np.ndarray, time_axis: np.ndarray) -> float:
    threshold = float(np.min(values) + 0.15 * (np.max(values) - np.min(values)))
    indices = np.flatnonzero(values >= threshold)
    if len(indices) == 0:
        return float("nan")
    return float(time_axis[indices[0]] - time_axis[0])


def peak_time(values: np.ndarray, time_axis: np.ndarray) -> float:
    return float(time_axis[int(np.argmax(values))] - time_axis[0])


def rollout_metrics(segment: ExactRSFSegment, sim: dict) -> dict:
    tau_sim = sim["tau"]
    v_sim = sim["V"]
    finite = np.isfinite(tau_sim) & np.isfinite(v_sim)
    if not np.any(finite):
        return {
            "tau_rmse": float("inf"),
            "V_rmse": float("inf"),
            "combined_rollout_error": float("inf"),
            "tau_mae": float("inf"),
            "V_mae": float("inf"),
            "onset_timing_error_s": float("inf"),
            "peak_timing_error_s": float("inf"),
            "stable_fraction": 0.0,
        }
    tau_rmse = float(np.sqrt(np.mean((tau_sim[finite] - segment.tau[finite]) ** 2)))
    v_rmse = float(np.sqrt(np.mean((v_sim[finite] - segment.V[finite]) ** 2)))
    tau_mae = float(np.mean(np.abs(tau_sim[finite] - segment.tau[finite])))
    v_mae = float(np.mean(np.abs(v_sim[finite] - segment.V[finite])))
    combined = 0.5 * (tau_rmse / segment.tau_scale + v_rmse / segment.v_scale)
    divergence_idx = len(segment.time)
    threshold = 3.0 * max(float(np.std(segment.V)), 1e-6)
    for index, (pred, obs) in enumerate(zip(v_sim, segment.V)):
        if (not np.isfinite(pred)) or abs(pred - obs) > threshold:
            divergence_idx = index
            break
    stable_fraction = float(divergence_idx / len(segment.time))
    return {
        "tau_rmse": tau_rmse,
        "V_rmse": v_rmse,
        "combined_rollout_error": float(combined),
        "tau_mae": tau_mae,
        "V_mae": v_mae,
        "onset_timing_error_s": abs(onset_time(v_sim[finite], segment.time[finite]) - onset_time(segment.V[finite], segment.time[finite])),
        "peak_timing_error_s": abs(peak_time(v_sim[finite], segment.time[finite]) - peak_time(segment.V[finite], segment.time[finite])),
        "stable_fraction": stable_fraction,
    }


def identifiability_diagnostics(
    result,
    train_segments: list[ExactRSFSegment],
    params: dict[str, float],
    parameter_names: list[str],
    acoustic_z: dict[str, float],
    *,
    use_acoustic: bool,
) -> dict:
    jac = np.asarray(result.jac, dtype=float)
    jtj = jac.T @ jac
    singular_values = np.linalg.svd(jtj, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jtj))
    cond = float(np.linalg.cond(jtj)) if np.all(np.isfinite(jtj)) else float("inf")
    covariance = np.linalg.pinv(jtj)
    variances = np.diag(covariance)
    denom = np.sqrt(np.outer(np.maximum(variances, 0.0), np.maximum(variances, 0.0))) + 1e-12
    corr = covariance / denom

    train_sigma = np.concatenate([segment.sigmaN for segment in train_segments])
    sigma_cv = float(np.std(train_sigma) / (np.mean(train_sigma) + 1e-12))

    profile_rows = []
    base_cost = float(result.cost)
    base_vector = np.asarray(result.x, dtype=float)
    for index, name in enumerate(parameter_names[: min(6 + int(use_acoustic), len(parameter_names))]):
        for frac in (0.9, 1.1):
            probe = base_vector.copy()
            probe[index] = np.clip(probe[index] * frac, result.x[index] - abs(result.x[index]) - 1 if False else probe[index] * frac, probe[index] * frac)
            residual = build_train_residual_vector(probe, train_segments, acoustic_z, use_acoustic=use_acoustic)
            profile_rows.append(
                {
                    "parameter": name,
                    "probe_scale": frac,
                    "cost_ratio": float(0.5 * np.sum(residual ** 2) / (base_cost + 1e-12)),
                }
            )

    theta_offset_names = [name for name in parameter_names if name.startswith("delta_log_theta0:")]
    theta_sensitivity = {
        name: float(np.linalg.norm(jac[:, parameter_names.index(name)]))
        for name in theta_offset_names
    }
    diagnostics = {
        "jtj_condition_number": cond,
        "jtj_rank": rank,
        "jtj_singular_values": singular_values.tolist(),
        "parameter_names": parameter_names,
        "parameter_correlation_matrix": corr.tolist(),
        "parameter_sensitivity_norms": [float(np.linalg.norm(jac[:, index])) for index in range(jac.shape[1])],
        "sigma_cv": sigma_cv,
        "sigma_too_constant_for_mu_a_b_separation": bool(sigma_cv < 0.01),
        "theta_offset_sensitivity": theta_sensitivity,
        "weak_theta_observability": bool(np.mean(list(theta_sensitivity.values()) or [0.0]) < 1.0),
        "profile_checks": profile_rows,
        "parameter_confounding_flag": bool(cond > 1e8 or rank < jac.shape[1]),
    }
    return diagnostics
