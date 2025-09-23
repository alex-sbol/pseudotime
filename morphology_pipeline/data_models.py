"""Common data structures used across pipeline modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ImageBundle:
    """Container for a single multi-channel image and associated file paths."""
    index: int
    channels: Dict[str, np.ndarray]
    paths: Dict[str, Path]


@dataclass
class AlignedBundle:
    """Result of rotating an image bundle to align corridor orientation."""
    bundle: ImageBundle
    rotation_deg: float
    channels: Dict[str, np.ndarray]


@dataclass
class SegmentationResult:
    """Segmentation products for one image, including optional nucleus mask."""
    cell_mask: np.ndarray
    nucleus_mask: Optional[np.ndarray] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class CellRecord:
    """Per-cell record storing environment context, location, and extracted features."""
    env_id: str
    image_index: int
    cell_id: int
    centroid_y: float
    pseudotime: float
    features: Dict[str, float]


@dataclass
class CorridorSignals:
    """One-dimensional corridor measurements sampled along the vertical axis."""
    corridor_width: np.ndarray
    cell_density: np.ndarray
    y_coordinates: np.ndarray


@dataclass
class PeriodEstimate:
    """Estimated repeating spatial period and phase offset for a corridor."""
    period: float
    phase_offset: float
    method: str


@dataclass
class PseudotimeStats:
    """Aggregated statistics computed in pseudotime bins."""
    bins: np.ndarray
    metrics: Dict[str, np.ndarray]


@dataclass
class ComparisonReport:
    """Summary of inter-environment differences across pseudotime."""
    feature_differences: Dict[str, np.ndarray]
    morphogenic_capacity: Dict[str, float]
