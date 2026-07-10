"""
patch_metadata.py

Defines the PatchMetadata dataclass.

PatchMetadata stores descriptive information about one extracted
patch. It contains no image data or processing logic and is
primarily intended for dataset serialization and experiment
tracking.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from data.patch.patch_coordinate import PatchCoordinate


@dataclass
class PatchMetadata:
    """
    Metadata describing one extracted patch.
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    patch_id: int

    # ---------------------------------------------------------
    # File Information
    # ---------------------------------------------------------

    image_filename: str

    annotation_filename: str

    # ---------------------------------------------------------
    # Spatial Information
    # ---------------------------------------------------------

    coordinate: PatchCoordinate

    # ---------------------------------------------------------
    # Image Information
    # ---------------------------------------------------------

    image_width: int

    image_height: int

    annotation_width: int

    annotation_height: int

    patch_area: int

    # ---------------------------------------------------------
    # Tissue Information
    # ---------------------------------------------------------

    tissue_percentage: float

    # ---------------------------------------------------------
    # Weak Label Information
    # ---------------------------------------------------------

    weak_label: np.ndarray

    detected_classes: List[int]

    class_pixel_counts: Dict[int, int]

    class_percentages: Dict[int, float]

    # ---------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------

    @property
    def image_shape(self):
        """
        Returns image shape as (height, width).
        """
        return (
            self.image_height,
            self.image_width,
        )

    @property
    def annotation_shape(self):
        """
        Returns annotation shape as (height, width).
        """
        return (
            self.annotation_height,
            self.annotation_width,
        )

    @property
    def num_classes(self) -> int:
        """
        Number of semantic classes present.
        """
        return len(self.detected_classes)

    @property
    def has_multiple_classes(self) -> bool:
        """
        Returns True if multiple semantic classes are present.
        """
        return self.num_classes > 1

    @property
    def tissue_ratio(self) -> float:
        """
        Alias for tissue_percentage.
        """
        return self.tissue_percentage

    @property
    def coordinate_tuple(self):
        """
        Returns (x, y, width, height).
        """
        return self.coordinate.to_tuple()