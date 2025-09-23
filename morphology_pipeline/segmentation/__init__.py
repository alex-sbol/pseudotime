"""Segmentation package."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import PipelineConfig
from ..data_models import SegmentationResult


class Segmenter(Protocol):
    def segment(self, channels: dict[str, np.ndarray]) -> SegmentationResult:
        ...


def get_segmenter(config: PipelineConfig) -> Segmenter:
    if config.segmentation.use_cellprofiler_masks:
        from .mask_segmenter import MaskSegmenter

        return MaskSegmenter()
    from .heuristic_segmenter import HeuristicSegmenter

    return HeuristicSegmenter(config.segmentation)
