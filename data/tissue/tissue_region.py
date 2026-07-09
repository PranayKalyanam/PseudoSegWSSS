from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.geometry.bounding_box import BoundingBox
from data.image.image_size import ImageSize


@dataclass(slots=True)
class TissueRegion:
    """
    Represents one connected tissue region.
    """

    # --------------------------------------------------
    # Identity
    # --------------------------------------------------

    region_id: int

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    bounding_box: BoundingBox

    area: int

    contour: np.ndarray | None = None

    # --------------------------------------------------
    # Binary Mask
    # --------------------------------------------------

    mask: np.ndarray | None = None

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    @property
    def x(self) -> int:
        return self.bounding_box.x

    @property
    def y(self) -> int:
        return self.bounding_box.y

    @property
    def width(self) -> int:
        return self.bounding_box.width

    @property
    def height(self) -> int:
        return self.bounding_box.height

    @property
    def right(self) -> int:
        return self.bounding_box.right

    @property
    def bottom(self) -> int:
        return self.bounding_box.bottom

    @property
    def center(self) -> tuple[float, float]:
        return self.bounding_box.center

    @property
    def size(self) -> ImageSize:
        return ImageSize(
            width=self.width,
            height=self.height,
        )