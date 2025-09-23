"""Pseudotime mapping utilities."""
from __future__ import annotations

import numpy as np

from .data_models import CellRecord, PeriodEstimate


def position_to_pseudotime(y: float, period: float, phase_offset: float) -> float:
    return ((y - phase_offset) % period) / period


def assign_pseudotime(records: list[CellRecord], period: PeriodEstimate) -> None:
    for record in records:
        record.pseudotime = position_to_pseudotime(record.centroid_y, period.period, period.phase_offset)


def bin_pseudotime(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist
