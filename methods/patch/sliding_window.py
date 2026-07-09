from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from data.patch.patch_coordinate import PatchCoordinate
from data.tissue.tissue_region import TissueRegion


@dataclass(slots=True)
class SlidingWindow:
    """
    Generates sliding-window patch coordinates over tissue regions.

    The generator operates in the working-image coordinate system.
    """

    patch_size: int
    overlap: float = 0.5

    @property
    def stride(self) -> int:
        """
        Sliding window stride.
        """
        return max(1, int(self.patch_size * (1.0 - self.overlap)))

    def generate(
        self,
        tissue_regions: list[TissueRegion],
        image_width: int,
        image_height: int,
    ) -> Iterator[PatchCoordinate]:
        """
        Generate candidate patch coordinates.

        Parameters
        ----------
        tissue_regions
            Connected tissue regions.

        image_width
            Working image width.

        image_height
            Working image height.

        Yields
        ------
        PatchCoordinate
        """

        for region in tissue_regions:

            x0 = region.bounding_box.x
            y0 = region.bounding_box.y
            w = region.bounding_box.width
            h = region.bounding_box.height

            xmax = min(x0 + w, image_width)
            ymax = min(y0 + h, image_height)

            y = y0

            while y + self.patch_size <= ymax:

                x = x0

                while x + self.patch_size <= xmax:

                    yield PatchCoordinate(
                        x=x,
                        y=y,
                        width=self.patch_size,
                        height=self.patch_size,
                    )

                    x += self.stride

                y += self.stride