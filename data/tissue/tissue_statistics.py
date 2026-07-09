from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TissueStatistics:
    """
    Quantitative statistics describing tissue.

    Statistics may describe either the complete
    tissue mask or an individual tissue region.
    """

    # --------------------------------------------------
    # Pixel Statistics
    # --------------------------------------------------

    tissue_pixels: int

    background_pixels: int

    total_pixels: int

    tissue_fraction: float

    background_fraction: float

    # --------------------------------------------------
    # Region Statistics
    # --------------------------------------------------

    number_of_regions: int = 0

    largest_region_area: int = 0

    smallest_region_area: int = 0

    mean_region_area: float = 0.0

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    @property
    def tissue_percentage(self) -> float:
        return self.tissue_fraction * 100.0

    @property
    def background_percentage(self) -> float:
        return self.background_fraction * 100.0