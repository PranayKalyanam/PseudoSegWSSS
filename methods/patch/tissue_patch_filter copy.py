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

    # ---------------------------------------------------------

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

        valid_patches = []

        for patch in patches:

            tissue_percentage = self._compute_tissue_percentage(
                patch.annotation_patch,
            )

            patch.tissue_percentage = tissue_percentage

            if tissue_percentage >= self.threshold:
                valid_patches.append(patch)

        return valid_patches

    # ---------------------------------------------------------

    def _compute_tissue_percentage(
        self,
        annotation_patch: np.ndarray,
    ) -> float:
        """
        Compute tissue percentage from an annotation patch.

        Any pixel whose label is not the background label
        is considered tissue.
        """

        total_pixels = annotation_patch.size

        if total_pixels == 0:
            return 0.0

        tissue_pixels = np.count_nonzero(
            annotation_patch != self.background_label
        )

        return tissue_pixels / total_pixels