"""Utility helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalise_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    rng = image.max() - image.min()
    if rng == 0:
        return np.zeros_like(image)
    return (image - image.min()) / rng


def sliding_window(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    pad = window // 2  # Mirror edges so averages remain centered.
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.array([
        padded[i : i + window].mean() for i in range(len(arr))
    ])


def iter_indices(indices: Iterable[int] | None, total: int) -> Iterable[int]:
    if indices is None:
        return range(total)
    return indices
