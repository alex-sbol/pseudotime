"""Shape descriptor extraction."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from skimage import measure

try:
    import mahotas
except ImportError:  # pragma: no cover - optional dependency
    mahotas = None

from .data_models import CellRecord


def _fourier_descriptors(mask: np.ndarray, order: int = 6) -> Dict[str, float]:
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return {}
    contour = max(contours, key=lambda c: c.shape[0])
    complex_points = contour[:, 0] + 1j * contour[:, 1]
    complex_points -= complex_points.mean()
    if complex_points.size == 0:
        return {}
    spectrum = np.fft.fft(complex_points)
    mags = np.abs(spectrum[1 : order + 1])
    if mags.size == 0:
        return {}
    first = mags[0] or 1.0
    norm = mags / first
    return {f"efd_{i}": float(val) for i, val in enumerate(norm, start=1)}


def _zernike_descriptors(mask: np.ndarray, radius: int = 10, degree: int = 8) -> Dict[str, float]:
    if mahotas is None:
        return {}
    padded = np.pad(mask.astype(float), radius)
    return {
        f"zernike_{i}": float(val)
        for i, val in enumerate(mahotas.features.zernike_moments(padded, radius=radius, degree=degree))
    }


def extract_cell_records(env_id: str, image_index: int, segmentation, channels: Dict[str, np.ndarray]) -> List[CellRecord]:
    labels = segmentation.cell_mask
    intensity = channels.get("actin")
    props = measure.regionprops(labels, intensity_image=intensity)
    records: List[CellRecord] = []
    for region in props:
        features: Dict[str, float] = {
            "area": float(region.area),
            "perimeter": float(region.perimeter),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "major_axis_length": float(region.major_axis_length),
            "minor_axis_length": float(region.minor_axis_length),
            "orientation": float(region.orientation),
            "mean_intensity": float(region.mean_intensity) if intensity is not None else 0.0,
        }
        cell_mask_local = region.image
        features.update(_fourier_descriptors(cell_mask_local))
        features.update(_zernike_descriptors(cell_mask_local))

        if segmentation.nucleus_mask is not None:
            coords = tuple(region.coords.T)
            nucleus_labels = segmentation.nucleus_mask[coords]
            nucleus_labels = nucleus_labels[nucleus_labels > 0]
            if nucleus_labels.size > 0:
                unique, counts = np.unique(nucleus_labels, return_counts=True)
                idx = int(np.argmax(counts))
                dominant = unique[idx]
                nucleus_area = int(counts[idx])
                features["nucleus_area"] = float(nucleus_area)
                features["nc_ratio"] = float(nucleus_area) / float(region.area)
                features["nucleus_label"] = float(dominant)
            else:
                features["nucleus_area"] = 0.0
                features["nc_ratio"] = 0.0
        else:
            features["nucleus_area"] = 0.0
            features["nc_ratio"] = 0.0

        record = CellRecord(
            env_id=env_id,
            image_index=image_index,
            cell_id=int(region.label),
            centroid_y=float(region.centroid[0]),
            pseudotime=0.0,
            features=features,
        )
        records.append(record)
    return records
