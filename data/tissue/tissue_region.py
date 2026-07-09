"""
tissue_region.py

Dataclass representing one connected tissue region detected
during tissue segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np


@dataclass(slots=True)
class TissueRegion:
    """
    Represents one connected tissue region.

    Attributes
    ----------
    region_id : int
        Unique region identifier.

    bbox : tuple
        Bounding box (x, y, width, height).

    contour : np.ndarray
        Contour points.

    mask : np.ndarray
        Binary region mask.

    image : np.ndarray
        Cropped RGB image.

    area : float
        Region area.

    tissue_percentage : float
        Tissue percentage.

    metadata : dict
        Additional dataset-specific information.
    """

    region_id: int

    bbox: tuple[int, int, int, int]

    contour: np.ndarray

    mask: np.ndarray

    image: Optional[np.ndarray] = None

    area: float = 0.0

    tissue_percentage: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]