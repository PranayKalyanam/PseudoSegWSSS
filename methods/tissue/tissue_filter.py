from __future__ import annotations

from data.tissue.tissue_region import TissueRegion


class TissueFilter:
    """
    Filters tissue regions.
    """

    @staticmethod
    def by_area(
        regions: list[TissueRegion],
        minimum_area: int,
    ) -> list[TissueRegion]:
        """
        Remove small regions.
        """

        return [

            region

            for region in regions

            if region.area >= minimum_area

        ]

    @staticmethod
    def by_size(
        regions: list[TissueRegion],
        minimum_width: int,
        minimum_height: int,
    ) -> list[TissueRegion]:

        return [

            region

            for region in regions

            if (
                region.width >= minimum_width
                and
                region.height >= minimum_height
            )

        ]

    @staticmethod
    def filter(
        regions: list[TissueRegion],
        minimum_area: int = 5000,
        minimum_width: int = 50,
        minimum_height: int = 50,
    ) -> list[TissueRegion]:
        """
        Apply all filtering operations.
        """

        regions = TissueFilter.by_area(
            regions,
            minimum_area,
        )

        regions = TissueFilter.by_size(
            regions,
            minimum_width,
            minimum_height,
        )

        return regions