"""Segmenter that ingests CellProfiler label masks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import tifffile

from ..data_models import SegmentationResult


class MaskSegmenter:
    """Wrapper around externally generated masks (e.g. CellProfiler)."""
    def __init__(self, mask_suffix: str = "_mask.tiff") -> None:
        self.mask_suffix = mask_suffix

    def segment(self, channels: Dict[str, object]) -> SegmentationResult:
        primary_path = channels.get("mask_path")
        if primary_path is None:
            raise ValueError("MaskSegmenter expects 'mask_path' entry in channels dict")
        # Simply load the provided label image and wrap it in the common result.
        cell_mask = tifffile.imread(Path(primary_path))
        return SegmentationResult(cell_mask=cell_mask)
