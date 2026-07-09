from __future__ import annotations

import numpy as np

from data.patch.patch import Patch


class PatchFilter:
    """
    Filters extracted patches.

    The filter removes patches that contain insufficient
    tissue or invalid content.

    Additional filtering rules can easily be added without
    modifying the PatchLoader.
    """

    def __init__(
        self,
        minimum_tissue_percentage: float = 0.75,
        minimum_variance: float = 5.0,
    ) -> None:

        self.minimum_tissue_percentage = minimum_tissue_percentage
        self.minimum_variance = minimum_variance

    def keep(
        self,
        patch: Patch,
        tissue_mask: np.ndarray,
    ) -> bool:
        """
        Determine whether a patch should be retained.

        Parameters
        ----------
        patch
            Extracted patch.

        tissue_mask
            Binary tissue mask corresponding to the patch.

        Returns
        -------
        bool
        """

        if patch.image.size == 0:
            return False

        tissue_ratio = self.tissue_percentage(
            tissue_mask
        )

        if tissue_ratio < self.minimum_tissue_percentage:
            return False

        if self.image_variance(
            patch.image
        ) < self.minimum_variance:
            return False

        return True

    @staticmethod
    def tissue_percentage(
        tissue_mask: np.ndarray,
    ) -> float:
        """
        Compute the percentage of tissue pixels.
        """

        if tissue_mask.size == 0:
            return 0.0

        return float(
            np.count_nonzero(tissue_mask)
        ) / tissue_mask.size

    @staticmethod
    def image_variance(
        image: np.ndarray,
    ) -> float:
        """
        Compute image intensity variance.

        Low variance patches are usually empty or nearly blank.
        """

        return float(
            np.var(image.astype(np.float32))
        )

    @staticmethod
    def contains_annotation(
        annotation_mask: np.ndarray,
    ) -> bool:
        """
        Check whether the annotation mask contains
        any labeled pixels.
        """

        return bool(
            np.any(annotation_mask > 0)
        )