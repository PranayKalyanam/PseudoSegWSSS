"""
tissue_statistics.py

Algorithms for computing quantitative statistics of
detected tissue.

This module contains no data structures. It computes
statistics and returns a TissueStatistics object.
"""

from __future__ import annotations

import numpy as np

from data.tissue.tissue_region import TissueRegion
from data.tissue.tissue_statistics import TissueStatistics


class TissueStatisticsGenerator:
    """
    Computes quantitative statistics describing
    detected tissue.
    """

    @staticmethod
    def compute(
        mask: np.ndarray,
        regions: list[TissueRegion],
    ) -> TissueStatistics:
        """
        Compute tissue statistics.

        Parameters
        ----------
        mask
            Binary tissue mask.

        regions
            Connected tissue regions.

        Returns
        -------
        TissueStatistics
        """

        # ------------------------------------------
        # Pixel Statistics
        # ------------------------------------------

        tissue_pixels = int(np.count_nonzero(mask))

        total_pixels = int(mask.size)

        background_pixels = (
            total_pixels - tissue_pixels
        )

        tissue_fraction = (
            tissue_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        background_fraction = (
            background_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        # ------------------------------------------
        # Region Statistics
        # ------------------------------------------

        number_of_regions = len(regions)

        if number_of_regions > 0:

            areas = [
                region.area
                for region in regions
            ]

            largest_region = max(areas)

            smallest_region = min(areas)

            mean_region = (
                sum(areas) / number_of_regions
            )

        else:

            largest_region = 0

            smallest_region = 0

            mean_region = 0.0

        # ------------------------------------------
        # Create Data Object
        # ------------------------------------------

        return TissueStatistics(

            tissue_pixels=tissue_pixels,

            background_pixels=background_pixels,

            total_pixels=total_pixels,

            tissue_fraction=tissue_fraction,

            background_fraction=background_fraction,

            number_of_regions=number_of_regions,

            largest_region_area=largest_region,

            smallest_region_area=smallest_region,

            mean_region_area=mean_region,
        )