"""
Common data structures used throughout the preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
from typing import List
from typing import Optional

import numpy as np


# =============================================================================
# Patch Record
# =============================================================================

@dataclass(slots=True)
class PatchRecord:
    """
    Represents one extracted image patch and its metadata.
    """

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    patch_id: str

    patient_id: str

    region_id: int

    filename: str

    # -------------------------------------------------------------------------
    # Image Data
    # -------------------------------------------------------------------------

    image: np.ndarray

    mask: np.ndarray

    # -------------------------------------------------------------------------
    # Spatial Information
    # -------------------------------------------------------------------------

    x: int

    y: int

    width: int

    height: int

    # -------------------------------------------------------------------------
    # Tissue Information
    # -------------------------------------------------------------------------

    tissue_percentage: float

    # -------------------------------------------------------------------------
    # Weak Labels
    # -------------------------------------------------------------------------

    labels: List[int]

    # -------------------------------------------------------------------------
    # Optional Metadata
    # -------------------------------------------------------------------------

    metadata: Dict = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------

    @property
    def label_string(
        self,
    ) -> str:
        """
        Returns

        Example
        -------
        [1,0,1,0]

        →

        "1010"
        """

        return "".join(
            map(
                str,
                self.labels,
            )
        )

    # -------------------------------------------------------------------------

    @property
    def has_positive_label(
        self,
    ) -> bool:

        return any(
            self.labels
        )

    # -------------------------------------------------------------------------

    @property
    def coordinate(
        self,
    ):

        return (
            self.x,
            self.y,
        )