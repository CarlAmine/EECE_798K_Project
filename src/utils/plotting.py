from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_signal_panel(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    title: str,
    figsize: tuple[int, int] = (12, 8),
):
    fig, axes = plt.subplots(len(value_cols), 1, figsize=figsize, sharex=True)
    if len(value_cols) == 1:
        axes = [axes]

    for axis, column in zip(axes, value_cols):
        axis.plot(df[time_col], df[column], linewidth=0.8)
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel(time_col)
    plt.tight_layout()
    return fig, axes


def plot_segment_boundaries(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    segments: Iterable[tuple[int, int]],
    title: str,
):
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(df[time_col], df[value_col], linewidth=0.8)
    for start, end in segments:
        axis.axvline(df.iloc[start][time_col], color="tab:red", alpha=0.5, linestyle="--")
        axis.axvline(df.iloc[end - 1][time_col], color="tab:green", alpha=0.4, linestyle=":")
    axis.set_title(title)
    axis.set_xlabel(time_col)
    axis.set_ylabel(value_col)
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, axis

