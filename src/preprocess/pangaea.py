from __future__ import annotations

import pandas as pd

from .common import build_canonical_modeling_frame, derive_velocity_from_displacement, remove_invalid_rows


def build_pangaea_state(
    df: pd.DataFrame,
    mapping: dict,
    derive_velocity_window: int = 31,
    derive_velocity_polyorder: int = 3,
) -> tuple[pd.DataFrame, dict]:
    time_col = mapping.get("time")
    tau_col = mapping.get("tau")
    displacement_col = mapping.get("displacement")
    velocity_col = mapping.get("velocity")

    if time_col is None or tau_col is None:
        raise ValueError("PANGAEA preprocessing requires mapped time and tau columns.")
    if velocity_col is None and displacement_col is None:
        raise ValueError("PANGAEA preprocessing requires either a velocity column or a displacement column.")

    working = pd.DataFrame({"time": df[time_col], "tau": df[tau_col]})
    if displacement_col is not None:
        working["displacement"] = df[displacement_col]
    if velocity_col is not None:
        working["V"] = df[velocity_col]
        velocity_mode = "measured"
    else:
        working = derive_velocity_from_displacement(
            working,
            displacement_col="displacement",
            time_col="time",
            output_col="V",
            window=derive_velocity_window,
            polyorder=derive_velocity_polyorder,
        )
        velocity_mode = "derived_from_displacement"

    working = remove_invalid_rows(working)
    modeling_df, metadata = build_canonical_modeling_frame(working, "time", ["tau", "V"], ["tau", "V"])
    metadata["velocity_mode"] = velocity_mode
    metadata["modeling_frame"] = modeling_df.to_dict(orient="list")
    return working, metadata

