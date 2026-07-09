from __future__ import annotations

import numpy as np

from data.patch.patch_coordinate import PatchCoordinate


class PatchExtractor:
    """
    Extracts image and annotation patches.

    This class is responsible only for extracting image
    and mask pixels. It does not create Patch objects or
    compute metadata, labels, or statistics.
    """

    @staticmethod
    def extract(
        image: np.ndarray,
        mask: np.ndarray,
        coordinate: PatchCoordinate,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract an image patch and its corresponding mask.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (image_patch, mask_patch)
        """

        x = coordinate.x
        y = coordinate.y
        w = coordinate.width
        h = coordinate.height

        image_patch = image[
            y:y + h,
            x:x + w,
        ].copy()

        mask_patch = mask[
            y:y + h,
            x:x + w,
        ].copy()

        return image_patch, mask_patch

    @staticmethod
    def is_complete(
        image: np.ndarray,
        coordinate: PatchCoordinate,
    ) -> bool:

        return (
            coordinate.x >= 0
            and coordinate.y >= 0
            and coordinate.x + coordinate.width <= image.shape[1]
            and coordinate.y + coordinate.height <= image.shape[0]
        )