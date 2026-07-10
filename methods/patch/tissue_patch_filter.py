"""
tissue_patch_filter.py

Filters extracted patches according to tissue coverage.

Responsibilities
----------------
1. Compute tissue percentage from annotation patch.
2. Store tissue statistics in Patch.
3. Reject patches below threshold.
4. Return valid patches.
"""

from __future__ import annotations

from typing import List

import numpy as np

from data.patch.patch import Patch


class TissuePatchFilter:
    """
    Filters extracted patches based on tissue coverage.

    This class operates only on extracted Patch objects.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        background_label: int = 0,
    ):
        self.threshold = threshold
        self.background_label = background_label

    # ==========================================================
    # Public API
    # ==========================================================

    def filter(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Filter patches using tissue percentage.

        Parameters
        ----------
        patches : List[Patch]

        Returns
        -------
        List[Patch]
        """

        valid_patches: List[Patch] = []

        for patch in patches:

            self._validate_patch(patch)

            (
                tissue_pixels,
                background_pixels,
                tissue_percentage,
                background_percentage,
            ) = self._compute_statistics(
                patch.annotation_patch,
            )

            # ---------------------------------------------
            # Store statistics
            # ---------------------------------------------

            patch.tissue_pixels = tissue_pixels
            patch.background_pixels = background_pixels

            patch.tissue_percentage = tissue_percentage
            patch.background_percentage = background_percentage

            # ---------------------------------------------
            # Keep / Reject
            # ---------------------------------------------

            if tissue_percentage >= self.threshold:
                valid_patches.append(patch)

        return valid_patches

    # ==========================================================
    # Internal Methods
    # ==========================================================

    def _compute_statistics(
        self,
        annotation_patch: np.ndarray,
    ):
        """
        Compute tissue statistics from one annotation patch.
        """

        total_pixels = annotation_patch.size

        if total_pixels == 0:

            return (
                0,
                0,
                0.0,
                0.0,
            )

        tissue_pixels = int(
            np.count_nonzero(
                annotation_patch != self.background_label
            )
        )

        background_pixels = (
            total_pixels - tissue_pixels
        )

        tissue_percentage = (
            tissue_pixels / total_pixels
        )

        background_percentage = (
            background_pixels / total_pixels
        )

        return (
            tissue_pixels,
            background_pixels,
            tissue_percentage,
            background_percentage,
        )

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_patch(
        self,
        patch: Patch,
    ) -> None:
        """
        Validate extracted patch before computing
        tissue statistics.
        """

        if patch.image_patch is None:
            raise ValueError(
                f"Patch {patch.patch_id} has no image patch."
            )

        if patch.annotation_patch is None:
            raise ValueError(
                f"Patch {patch.patch_id} has no annotation patch."
            )

        if (
            patch.image_patch.shape[:2]
            != patch.annotation_patch.shape[:2]
        ):
            raise ValueError(
                f"Image and annotation dimensions differ "
                f"for Patch {patch.patch_id}."
            )