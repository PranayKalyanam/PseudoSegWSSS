"""
patch.py

Defines the Patch dataclass.

A Patch represents one extracted image region throughout the
preprocessing pipeline. The Patch object is progressively enriched
by successive preprocessing modules while remaining a lightweight
data container.

Pipeline Evolution
------------------
PatchGenerator
    -> patch_id
    -> coordinate

TissuePatchFilter
    -> tissue_percentage

PatchExtractor
    -> image_patch
    -> annotation_patch

WeakLabelGenerator
    -> weak_label
    -> detected_classes
    -> class_pixel_counts
    -> class_percentages

PatchMetadataGenerator
    -> metadata
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from data.patch.patch_coordinate import PatchCoordinate
from data.patch.patch_metadata import PatchMetadata


@dataclass
class Patch:
    """
    Represents one extracted image patch.
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    patch_id: int

    # ---------------------------------------------------------
    # Spatial Information
    # ---------------------------------------------------------

    coordinate: PatchCoordinate

    # ---------------------------------------------------------
    # Tissue Information
    # ---------------------------------------------------------

    tissue_percentage: float = 0.0

    # ---------------------------------------------------------
    # Image Data
    # ---------------------------------------------------------

    image_patch: Optional[np.ndarray] = None

    annotation_patch: Optional[np.ndarray] = None

    # ---------------------------------------------------------
    # Weak Labels
    # ---------------------------------------------------------

    weak_label: Optional[np.ndarray] = None

    detected_classes: Optional[List[int]] = None

    class_pixel_counts: Optional[Dict[int, int]] = None

    class_percentages: Optional[Dict[int, float]] = None

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata: Optional[PatchMetadata] = None

    # ---------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------

    @property
    def width(self) -> int:
        return self.coordinate.width

    @property
    def height(self) -> int:
        return self.coordinate.height

    @property
    def x(self) -> int:
        return self.coordinate.x

    @property
    def y(self) -> int:
        return self.coordinate.y

    @property
    def area(self) -> int:
        return self.coordinate.width * self.coordinate.height

    @property
    def has_image(self) -> bool:
        return self.image_patch is not None

    @property
    def has_annotation(self) -> bool:
        return self.annotation_patch is not None

    @property
    def has_labels(self) -> bool:
        return self.weak_label is not None

    @property
    def is_processed(self) -> bool:
        """
        Returns True if the patch has completed all
        preprocessing stages.
        """
        return (
            self.has_image
            and self.has_annotation
            and self.has_labels
            and self.metadata is not None
        )