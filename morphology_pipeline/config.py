"""Configuration loading utilities for the morphology pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ChannelConfig:
    """Mapping from logical channel roles to filename suffixes."""
    actin: str
    nucleus: str
    background: Optional[str] = None
    auxiliary: Dict[str, str] = field(default_factory=dict)


@dataclass
class SegmentationConfig:
    """Parameters that influence nucleus/cell mask creation."""
    use_cellprofiler_masks: bool = False
    cell_min_area: int = 200
    nucleus_min_area: int = 80
    gaussian_sigma: float = 1.5
    closing_radius: int = 3
    watershed_compactness: float = 0.001


@dataclass
class PipelineConfig:
    """Top-level container aggregating all pipeline configuration sections."""
    channels: ChannelConfig
    output_dir: Path
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    file_pattern: str = "obj{index:04d}_stain{channel}_real.tiff"
    indices: Optional[List[int]] = None

    def env_paths(self) -> Dict[str, Path]:
        return {key: cfg.root for key, cfg in self.environments.items()}


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
