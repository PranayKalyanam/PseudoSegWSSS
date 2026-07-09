from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PatchStatistics:
    """
    Pixel statistics of an extracted patch.

    These statistics are independent of the labels used
    during training.
    """

    total_pixels: int

    tissue_pixels: int

    background_pixels: int

    tumor_pixels: int = 0

    stroma_pixels: int = 0

    lymphocyte_pixels: int = 0

    necrosis_pixels: int = 0

    other_pixels: int = 0

    @property
    def tissue_ratio(self) -> float:
        if self.total_pixels == 0:
            return 0.0
        return self.tissue_pixels / self.total_pixels

    @property
    def background_ratio(self) -> float:
        if self.total_pixels == 0:
            return 0.0
        return self.background_pixels / self.total_pixels