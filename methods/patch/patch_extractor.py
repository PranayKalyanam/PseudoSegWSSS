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
6. Validate extracted patch dimensions.
7. Return processed patches.
"""

from __future__ import annotations

from typing import List

import numpy as np
from pathlib import Path

from data.patch.patch import Patch
from data.patch.patch_coordinate import PatchCoordinate


class PatchExtractor:
    """
    Extracts image and annotation patches.

    This class is solely responsible for constructing
    Patch objects from image coordinates.
    """

    def __init__(self):
        pass

    # ==========================================================
    # Public API
    # ==========================================================

    def extract(
        self,
        coordinates: List[PatchCoordinate],
        image: np.ndarray,
        annotation: np.ndarray,
        source_filename: str,
    ) -> List[Patch]:
        """
        Extract all patches.

        Parameters
        ----------
        coordinates
            Candidate patch coordinates.

        image
            Original RGB image.

        annotation
            Original annotation mask.

        Returns
        -------
        List[Patch]
        """

        if image is None:
            raise ValueError("Image cannot be None.")

        if annotation is None:
            raise ValueError("Annotation cannot be None.")
        
        filename = Path(source_filename).stem

        patches: List[Patch] = []

        for patch_id, coordinate in enumerate(coordinates):

            patch = self._extract_single_patch(
                patch_id=patch_id,
                coordinate=coordinate,
                image=image,
                annotation=annotation,
                source_filename=filename,
            )

            patches.append(patch)

        return patches

    # ==========================================================
    # Internal Extraction
    # ==========================================================

    def _extract_single_patch(
        self,
        patch_id: int,
        coordinate: PatchCoordinate,
        image: np.ndarray,
        annotation: np.ndarray,
        source_filename: str,
    ) -> Patch:
        """
        Extract one image patch and annotation patch.
        """

        x = coordinate.x
        y = coordinate.y
        w = coordinate.width
        h = coordinate.height

        image_height, image_width = image.shape[:2]

        # ---------------------------------------------
        # Coordinate validation
        # ---------------------------------------------

        if x < 0 or y < 0:
            raise ValueError(
                f"Negative coordinate encountered "
                f"(x={x}, y={y})."
            )

        if x >= image_width or y >= image_height:
            raise ValueError(
                f"Coordinate outside image boundary "
                f"(x={x}, y={y})."
            )

        # ---------------------------------------------
        # Crop available region
        # ---------------------------------------------

        image_patch = image[
            y:min(y + h, image_height),
            x:min(x + w, image_width),
        ].copy()

        annotation_patch = annotation[
            y:min(y + h, image_height),
            x:min(x + w, image_width),
        ].copy()

        # ---------------------------------------------
        # Pad if border patch
        # ---------------------------------------------

        image_patch = self._pad_array(
            array=image_patch,
            expected_height=h,
            expected_width=w,
            pad_value=255,
        )

        annotation_patch = self._pad_array(
            array=annotation_patch,
            expected_height=h,
            expected_width=w,
            pad_value=0,
        )

        # ---------------------------------------------
        # Final validation
        # ---------------------------------------------

        self._validate_patch_shape(
            image_patch,
            annotation_patch,
            h,
            w,
        )

        # ---------------------------------------------
        # Construct Patch object
        # ---------------------------------------------

        patch = Patch(
            patch_id=patch_id,
            source_filename=source_filename,
            coordinate=coordinate,
        )

        patch.image_patch = image_patch
        patch.annotation_patch = annotation_patch

        return patch

    # ==========================================================
    # Padding
    # ==========================================================

    def _pad_array(
        self,
        array: np.ndarray,
        expected_height: int,
        expected_width: int,
        pad_value: int,
    ) -> np.ndarray:
        """
        Pad image or annotation array to the required size.
        """

        height, width = array.shape[:2]

        pad_bottom = max(
            0,
            expected_height - height,
        )

        pad_right = max(
            0,
            expected_width - width,
        )

        if pad_bottom == 0 and pad_right == 0:
            return array

        if array.ndim == 3:

            return np.pad(
                array,
                (
                    (0, pad_bottom),
                    (0, pad_right),
                    (0, 0),
                ),
                mode="constant",
                constant_values=pad_value,
            )

        return np.pad(
            array,
            (
                (0, pad_bottom),
                (0, pad_right),
            ),
            mode="constant",
            constant_values=pad_value,
        )

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_patch_shape(
        self,
        image_patch: np.ndarray,
        annotation_patch: np.ndarray,
        expected_height: int,
        expected_width: int,
    ) -> None:
        """
        Ensure extracted patches have the expected dimensions.
        """

        if image_patch.shape[:2] != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                "Image patch has invalid dimensions. "
                f"Expected {(expected_height, expected_width)}, "
                f"got {image_patch.shape[:2]}."
            )

        if annotation_patch.shape[:2] != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                "Annotation patch has invalid dimensions. "
                f"Expected {(expected_height, expected_width)}, "
                f"got {annotation_patch.shape[:2]}."
            )