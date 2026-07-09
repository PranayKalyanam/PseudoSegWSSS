from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.tissue.tissue_region import TissueRegion
from data.tissue.tissue_statistics import TissueStatistics


@dataclass(slots=True)
class TissueMask:
    """
    Result produced by a tissue detector.

    This object encapsulates everything generated
    during the tissue detection stage.
    """

    # --------------------------------------------------
    # Binary Tissue Mask
    # --------------------------------------------------

    mask: np.ndarray

    # --------------------------------------------------
    # Connected Regions
    # --------------------------------------------------

    regions: list[TissueRegion] = field(
        default_factory=list
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    statistics: TissueStatistics | None = None

    # --------------------------------------------------
    # Detector Information
    # --------------------------------------------------

    detector_name: str = ""

    threshold: float | None = None

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    @property
    def number_of_regions(self) -> int:

        return len(self.regions)

    @property
    def height(self) -> int:

        return self.mask.shape[0]

    @property
    def width(self) -> int:

        return self.mask.shape[1]

    @property
    def shape(self):

        return self.mask.shape

    @property
    def has_regions(self) -> bool:

        return len(self.regions) > 0