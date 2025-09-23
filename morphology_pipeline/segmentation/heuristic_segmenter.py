"""Heuristic segmentation for cell and nucleus masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from ..config import SegmentationConfig
from ..data_models import SegmentationResult
from ..utils import normalise_image


class HeuristicSegmenter:
    """Simple rule-based segmentation using Gaussian+Otsu and watershed."""
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config

    def _threshold(self, image: np.ndarray, minimum_area: int) -> np.ndarray:
        # Lightly smooth to suppress noise before thresholding.
        blurred = filters.gaussian(image, sigma=self.config.gaussian_sigma)
        thresh = filters.threshold_otsu(blurred)
        mask = blurred > thresh
        # Remove speckles below the configured minimum object area.
        mask = morphology.remove_small_objects(mask, minimum_area)
        if self.config.closing_radius > 0:
            selem = morphology.disk(self.config.closing_radius)
            mask = morphology.closing(mask, selem)
        return mask

    def segment(self, channels: dict[str, np.ndarray]) -> SegmentationResult:
        actin = channels.get("actin")
        nucleus = channels.get("nucleus")
        if actin is None or nucleus is None:
            raise ValueError("Heuristic segmentation requires actin and nucleus channels")

        actin_norm = normalise_image(actin)
        nucleus_norm = normalise_image(nucleus)

        # Create a coarse cell mask from actin signal and compute distance transform.
        cell_mask = self._threshold(actin_norm, self.config.cell_min_area)
        distance = ndi.distance_transform_edt(cell_mask)

        if nucleus_norm is not None:
            nuc_mask = self._threshold(nucleus_norm, self.config.nucleus_min_area)
            nuc_mask = morphology.remove_small_objects(nuc_mask, self.config.nucleus_min_area)
        else:
            nuc_mask = distance > np.percentile(distance[cell_mask], 80)

        # Seed watershed with nuclei markers to split touching cells.
        markers, _ = ndi.label(nuc_mask)
        labels = segmentation.watershed(-distance, markers, mask=cell_mask, compactness=self.config.watershed_compactness)
        # Refine nucleus labels using the original intensity image.
        nuc_labels = segmentation.watershed(-normalise_image(nucleus), markers, mask=nuc_mask)

        return SegmentationResult(cell_mask=labels, nucleus_mask=nuc_labels)
