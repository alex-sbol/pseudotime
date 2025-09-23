"""Visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_models import ComparisonReport, CorridorSignals
from .utils import ensure_dir


def plot_corridor_signals(signals: CorridorSignals, env_name: str, output_dir: Path, fmt: str = "png") -> Path:
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(signals.y_coordinates, signals.corridor_width, label="Corridor width")
    ax.plot(signals.y_coordinates, signals.cell_density, label="Cell density")
    ax.set_xlabel("Position (pixels)")
    ax.set_ylabel("Signal")
    ax.set_title(f"Corridor signals – {env_name}")
    ax.legend()
    output = output_dir / f"{env_name.replace(' ', '_').lower()}_signals.{fmt}"
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_feature_trends(stats: pd.DataFrame, env_name: str, output_dir: Path, fmt: str = "png") -> Dict[str, Path]:
    ensure_dir(output_dir)
    paths: Dict[str, Path] = {}
    for feature, group in stats.groupby("feature"):
        fig, ax = plt.subplots(figsize=(6, 4))
        pivot = group.pivot(index="bin", columns="env_id", values="median")
        env_series = pivot.iloc[:, 0]
        ax.plot(env_series.index, env_series.values, label="Median")
        variability = group.set_index("bin")[group.columns[-1]]
        ax.fill_between(variability.index, env_series.values - variability.values, env_series.values + variability.values, alpha=0.2, label="Variability")
        ax.set_xlabel("Pseudotime bin")
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} vs pseudotime – {env_name}")
        ax.legend()
        output = output_dir / f"{env_name.replace(' ', '_').lower()}_{feature}.{fmt}"
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)
        paths[feature] = output
    return paths


def plot_comparison(report: ComparisonReport, output_dir: Path, fmt: str = "png") -> Dict[str, Path]:
    ensure_dir(output_dir)
    paths: Dict[str, Path] = {}
    for feature, diff in report.feature_differences.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.linspace(0, 1, diff.size), diff)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("EnvA − EnvB")
        ax.set_title(f"Difference in {feature}")
        output = output_dir / f"comparison_{feature}.{fmt}"
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)
        paths[feature] = output
    return paths
