"""
patch_extractor.py

Extracts image patches from the original image and annotation mask.

Responsibilities
----------------
1. Receive PatchCoordinate objects.
2. Crop image patch.
3. Crop annotation patch.
4. Pad border patches when necessary.
5. Construct Patch objects.
6. Return processed patches.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from data.patch.patch import Patch
from data.patch.patch_coordinate import PatchCoordinate


class PatchExtractor:
    """
    Extract image and annotation patches.

    This is the only class responsible for constructing
    Patch objects.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------

    def extract(
        self,
        coordinates: List[PatchCoordinate],
        image: np.ndarray,
        annotation: np.ndarray,
    ) -> List[Patch]:
        """
        Extract all patches.

        Parameters
        ----------
        coordinates

        image

        annotation

        Returns
        -------
        List[Patch]
        """

        patches = []

        for patch_id, coordinate in enumerate(coordinates):

            patch = self._extract_single_patch(
                patch_id,
                coordinate,
                image,
                annotation,
            )

            patches.append(patch)

        return patches

    # ---------------------------------------------------------

    def _extract_single_patch(
        self,
        patch_id: int,
        coordinate: PatchCoordinate,
        image: np.ndarray,
        annotation: np.ndarray,
    ) -> Patch:
        """
        Extract one patch.
        """

        x = coordinate.x
        y = coordinate.y
        w = coordinate.width
        h = coordinate.height

        image_patch = image[
            y:y+h,
            x:x+w,
        ].copy()

        annotation_patch = annotation[
            y:y+h,
            x:x+w,
        ].copy()

        image_patch = self._pad_image_patch(
            image_patch,
            h,
            w,
        )

        annotation_patch = self._pad_mask_patch(
            annotation_patch,
            h,
            w,
        )

        patch = Patch(
            patch_id=patch_id,
            coordinate=coordinate,
        )

        patch.image_patch = image_patch
        patch.annotation_patch = annotation_patch

        return patch

    # ---------------------------------------------------------

    def _pad_image_patch(
        self,
        image_patch: np.ndarray,
        expected_height: int,
        expected_width: int,
    ) -> np.ndarray:
        """
        Pad RGB image patch if required.
        """

        height, width = image_patch.shape[:2]

        pad_bottom = max(
            0,
            expected_height - height,
        )

        pad_right = max(
            0,
            expected_width - width,
        )

        if pad_bottom == 0 and pad_right == 0:
            return image_patch

        return cv2.copyMakeBorder(
            image_patch,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    # ---------------------------------------------------------

    def _pad_mask_patch(
        self,
        mask_patch: np.ndarray,
        expected_height: int,
        expected_width: int,
    ) -> np.ndarray:
        """
        Pad annotation patch if required.
        """

        height, width = mask_patch.shape[:2]

        pad_bottom = max(
            0,
            expected_height - height,
        )

        pad_right = max(
            0,
            expected_width - width,
        )

        if pad_bottom == 0 and pad_right == 0:
            return mask_patch

        return cv2.copyMakeBorder(
            mask_patch,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )