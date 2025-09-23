"""Image loading and discovery utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Generator, Iterable

import numpy as np
import tifffile

from .config import ChannelConfig, PipelineConfig
from .data_models import ImageBundle


_INDEX_RE = re.compile(r"obj(\d+)")


def _pattern_to_glob(pattern: str, channel: str) -> str:
    # Replace the formatted placeholders with wildcards so we can glob.
    glob_pattern = pattern.replace("{index:04d}", "*")
    glob_pattern = glob_pattern.replace("{channel}", channel)
    return glob_pattern


def _extract_index(path: Path) -> int:
    # Expect filenames like obj0001_stainActin... and extract the numeric token.
    match = _INDEX_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not parse index from {path.name}")
    return int(match.group(1))


def _channel_map(channels: ChannelConfig) -> Dict[str, str]:
    mapping = {
        "actin": channels.actin,
        "nucleus": channels.nucleus,
    }
    if channels.background:
        mapping["background"] = channels.background
    mapping.update(channels.auxiliary)
    return mapping


def discover_indices(root: Path, config: PipelineConfig) -> Iterable[int]:
    # Build a deterministic list of available indices using the primary channel.
    channel_map = _channel_map(config.channels)
    primary_channel = channel_map.get("nucleus") or channel_map.get("actin")
    if primary_channel is None:
        raise ValueError("At least one primary channel (nucleus or actin) must be defined.")
    glob_pattern = _pattern_to_glob(config.file_pattern, primary_channel)
    return sorted(_extract_index(path) for path in root.glob(glob_pattern))


def load_bundle(root: Path, config: PipelineConfig, index: int) -> ImageBundle:
    # Build a deterministic list of available indices using the primary channel.
    channel_map = _channel_map(config.channels)
    channels: Dict[str, np.ndarray] = {}
    paths: Dict[str, Path] = {}
    for label, channel in channel_map.items():
        filename = config.file_pattern.format(index=index, channel=channel)
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing channel '{channel}' for image {index:04d} at {path}")
        data = tifffile.imread(path)
        channels[label] = data
        paths[label] = path
    return ImageBundle(index=index, channels=channels, paths=paths)


def iter_bundles(root: Path, config: PipelineConfig, indices: Iterable[int]) -> Generator[ImageBundle, None, None]:
    for idx in indices:
        # Lazily yield one bundle at a time to keep memory usage manageable.
        yield load_bundle(root, config, idx)
