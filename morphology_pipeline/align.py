"""Corridor alignment utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
from skimage import feature, transform

from .data_models import AlignedBundle, ImageBundle
from .utils import normalise_image


@dataclass
class CorridorAligner:
    """Find and apply rotations that align corridor structures vertically."""
    low_threshold: float = 0.1
    high_threshold: float = 0.3
    angle_snap: float = 0.5  # degrees

    def _estimate_angle(self, image: np.ndarray) -> float:
        # Detect corridor edges and orientation using Canny + Hough transforms.
        edges = feature.canny(normalise_image(image), sigma=3.0,
                              low_threshold=self.low_threshold,
                              high_threshold=self.high_threshold)
        hspace, angles, _ = transform.hough_line(edges)
        accum, angle_list = transform.hough_line_peaks(hspace, angles, num_peaks=5)
        if len(accum) == 0:
            return 0.0
        degrees = [math.degrees(a) for a in angle_list]
        avg_angle = float(np.mean(degrees))
        # Express angle relative to vertical axis and snap for stability.
        deviation = avg_angle % 90
        if deviation > 45:
            deviation -= 90
        snapped = round(deviation / self.angle_snap) * self.angle_snap
        return snapped

    def align(self, bundle: ImageBundle) -> AlignedBundle:
        reference = bundle.channels.get("background") or bundle.channels.get("actin")
        if reference is None:
            raise ValueError("Alignment requires a background or actin channel")
        angle = self._estimate_angle(reference)
        aligned_channels: Dict[str, np.ndarray] = {}
        for label, image in bundle.channels.items():
            rotated = transform.rotate(image, angle=angle, resize=False, preserve_range=True, mode="edge")
            aligned_channels[label] = rotated.astype(image.dtype)
        return AlignedBundle(bundle=bundle, rotation_deg=angle, channels=aligned_channels)
