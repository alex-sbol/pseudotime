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
class SignalConfig:
    """Settings for corridor signal extraction and smoothing."""
    corridor_threshold: float = 0.5
    smooth_sigma: float = 2.0
    density_window: int = 15


@dataclass
class PeriodConfig:
    """Bounds and tuning for periodicity estimation."""
    min_period_pixels: int = 40
    max_period_pixels: int = 400
    prominence: float = 0.1


@dataclass
class PseudotimeConfig:
    """Controls binning of pseudotime assignments for downstream stats."""
    bins: int = 20
    reference_phase: Optional[float] = None


@dataclass
class AnalysisConfig:
    """Statistical aggregation options for morphology variation analysis."""
    variability_metric: str = "mad"
    detect_threshold_quantile: float = 0.85


@dataclass
class VisualizationConfig:
    """Flags that toggle figure creation and file formats."""
    enabled: bool = True
    format: str = "png"


@dataclass
class ModelConfig:
    """Configuration for the optional TopoChip generative model."""
    enabled: bool = False
    model_type: str = "topochip"
    weights_path: Optional[Path] = None
    framework: Optional[str] = None
    use_ema: bool = True
    image_size: int = 256
    base_channels: int = 32
    num_res_blocks: int = 2
    channel_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 2])
    num_heads: int = 4
    num_head_channels: int = -1
    attention_downsample: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    num_cond_channels: int = 2
    use_cfg: bool = True
    cfg_strength: float = 1.0
    num_integration_steps: int = 100
    normalize_background: str = "per-image"
    samples_per_env: int = 4
    max_backgrounds: int = 32
    device: Optional[str] = None
    output_format: str = "png"


@dataclass
class EnvironmentConfig:
    """Definition of a single corridor environment input."""
    name: str
    root: Path


@dataclass
class PipelineConfig:
    """Top-level container aggregating all pipeline configuration sections."""
    environments: Dict[str, EnvironmentConfig]
    channels: ChannelConfig
    output_dir: Path
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    period: PeriodConfig = field(default_factory=PeriodConfig)
    pseudotime: PseudotimeConfig = field(default_factory=PseudotimeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    file_pattern: str = "obj{index:04d}_stain{channel}_real.tiff"
    indices: Optional[List[int]] = None

    def env_paths(self) -> Dict[str, Path]:
        return {key: cfg.root for key, cfg in self.environments.items()}


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    envs = {
        key: EnvironmentConfig(name=value["name"], root=Path(value["root"]).expanduser())
        for key, value in data["environments"].items()
    }

    channels = ChannelConfig(**data["channels"])
    segmentation = SegmentationConfig(**data.get("segmentation", {}))
    signals = SignalConfig(**data.get("signals", {}))
    period = PeriodConfig(**data.get("period", {}))
    pseudotime = PseudotimeConfig(**data.get("pseudotime", {}))
    analysis = AnalysisConfig(**data.get("analysis", {}))
    visualization = VisualizationConfig(**data.get("visualization", {}))

    model_data = data.get("model", {})
    if "weights_path" in model_data and model_data["weights_path"]:
        model_data["weights_path"] = Path(model_data["weights_path"]).expanduser()
    if "channel_mult" in model_data:
        model_data["channel_mult"] = list(model_data["channel_mult"])
    if "attention_downsample" in model_data:
        model_data["attention_downsample"] = list(model_data["attention_downsample"])
    model = ModelConfig(**model_data)

    output_dir = Path(data["output_dir"]).expanduser()
    indices = data.get("indices")

    return PipelineConfig(
        environments=envs,
        channels=channels,
        output_dir=output_dir,
        segmentation=segmentation,
        signals=signals,
        period=period,
        pseudotime=pseudotime,
        analysis=analysis,
        visualization=visualization,
        model=model,
        file_pattern=data.get("file_pattern", "obj{index:04d}_stain{channel}_real.tiff"),
        indices=indices,
    )
