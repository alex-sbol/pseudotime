"""Confinement signal extraction."""
from __future__ import annotations

import numpy as np
from skimage import filters

from .data_models import CorridorSignals, SegmentationResult
from .utils import normalise_image, sliding_window


def corridor_mask(background: np.ndarray, threshold: float) -> np.ndarray:
    norm = normalise_image(background)
    # Either use adaptive thresholding or a fixed numeric cutoff.
    adaptive = filters.threshold_local(norm, block_size=51) if threshold == "adaptive" else None
    if adaptive is not None:
        mask = norm > adaptive
    else:
        mask = norm > threshold
    return mask


def compute_signals(background: np.ndarray, segmentation: SegmentationResult, threshold: float, density_window: int) -> CorridorSignals:
    mask = corridor_mask(background, threshold)
    height = mask.shape[0]
    # Corridor width is the number of foreground pixels per row.
    corridor_width = mask.sum(axis=1).astype(float)

    # Collapse labelled mask to binary occupancy then smooth with a sliding window.
    cell_presence = segmentation.cell_mask > 0
    cell_area = cell_presence.sum(axis=1).astype(float)
    smoothed_density = sliding_window(cell_area, density_window)

    y_coords = np.arange(height)
    return CorridorSignals(corridor_width=corridor_width, cell_density=smoothed_density, y_coordinates=y_coords)
