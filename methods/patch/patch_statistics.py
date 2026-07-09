from __future__ import annotations

import numpy as np

from data.patch.patch import Patch
from data.patch.patch_statistics import PatchStatistics

from methods.patch.patch_label_generator import (
    PatchLabelGenerator,
)


class PatchStatistics:
    """
    Computes descriptive statistics for an extracted patch.
    """

    @classmethod
    def compute(
        cls,
        patch: Patch,
        tissue_mask: np.ndarray,
    ) -> PatchStatistics:
        """
        Compute patch statistics.
        """

        image = patch.image
        mask = patch.mask

        total_pixels = mask.size

        tissue_pixels = int(
            np.count_nonzero(tissue_mask)
        )

        background_pixels = (
            total_pixels - tissue_pixels
        )

        class_pixel_counts = {}

        class_percentages = {}

        detected = (
            PatchLabelGenerator.detected_classes(
                mask
            )
        )

        for class_id in detected:

            pixels = int(
                np.sum(mask == class_id)
            )

            class_pixel_counts[class_id] = pixels

            class_percentages[class_id] = (
                pixels / total_pixels
            )

        return PatchStatistics(

            total_pixels=total_pixels,

            tissue_pixels=tissue_pixels,

            background_pixels=background_pixels,

            tissue_percentage=(
                tissue_pixels / total_pixels
            ),

            background_percentage=(
                background_pixels / total_pixels
            ),

            mean_rgb=tuple(
                image.reshape(-1, 3)
                .mean(axis=0)
                .tolist()
            ),

            std_rgb=tuple(
                image.reshape(-1, 3)
                .std(axis=0)
                .tolist()
            ),

            class_pixel_counts=class_pixel_counts,

            class_percentages=class_percentages,

            detected_classes=detected,
        )