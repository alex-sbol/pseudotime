"""Integration with external generative models."""
from __future__ import annotations

from typing import Optional

from .config import ModelConfig

try:
    from .models.topochip import TopochipFlowGenerator
except ImportError:  # pragma: no cover
    TopochipFlowGenerator = None  # type: ignore


def create_shape_generator(config: ModelConfig) -> Optional[TopochipFlowGenerator]:
    if not config.enabled:
        return None

    model_type = (config.model_type or "topochip").lower()
    if model_type == "topochip":
        if TopochipFlowGenerator is None:
            raise RuntimeError("TopoChip generator requires torch and associated dependencies to be installed.")
        return TopochipFlowGenerator(config)

    raise ValueError(f"Unsupported model type '{config.model_type}'")
