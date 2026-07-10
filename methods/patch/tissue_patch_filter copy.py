"""
tissue_patch_filter.py

Filters candidate patches based on tissue coverage.

The TissuePatchFilter determines whether each candidate patch
contains sufficient tissue to be retained for later processing.
Image patches are not extracted at this stage. Instead, tissue
coverage is computed directly from the binary tissue mask using
the patch coordinates.

Responsibilities
----------------
1. Compute tissue coverage for every candidate patch.
2. Remove patches with insufficient tissue.
3. Store tissue statistics inside the Patch object.
4. Return valid candidate patches.
"""

from typing import List

import numpy as np

from data.patch.patch import Patch


class TissuePatchFilter:
    """
    Filters candidate patches according to tissue coverage.
    """

    def __init__(
        self,
        threshold: float = 0.50,
    ):
        self.threshold = threshold

    # ---------------------------------------------------------

    def filter(
        self,
        patches: List[Patch],
        tissue_mask: np.ndarray,
    ) -> List[Patch]:
        """
        Filter patches according to tissue coverage.

        Parameters
        ----------
        patches : List[Patch]

        tissue_mask : np.ndarray

        Returns
        -------
        List[Patch]
        """

        valid_patches = []

        for patch in patches:

            coverage = self._compute_tissue_coverage(
                patch,
                tissue_mask,
            )

            patch.tissue_percentage = coverage

            if coverage >= self.threshold:
                valid_patches.append(patch)

        return valid_patches

    # ---------------------------------------------------------

    def _compute_tissue_coverage(
        self,
        patch: Patch,
        tissue_mask: np.ndarray,
    ) -> float:
        """
        Compute the percentage of tissue inside one patch.
        """

        coordinate = patch.coordinate

        x = coordinate.x
        y = coordinate.y
        w = coordinate.width
        h = coordinate.height

        patch_mask = tissue_mask[
            y:y+h,
            x:x+w,
        ]

        total_pixels = patch_mask.size

        if total_pixels == 0:
            return 0.0

        tissue_pixels = np.count_nonzero(patch_mask)

        return tissue_pixels / total_pixels