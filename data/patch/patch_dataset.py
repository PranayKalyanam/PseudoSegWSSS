"""
patch_dataset.py

Defines the PatchDataset dataclass.

A PatchDataset represents all processed patches belonging to a
single patient after completion of the patch preprocessing stage.

The class acts purely as a data container and contains no
preprocessing logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from data.patch.patch import Patch
from data.patch.patch_dataset_statistics import PatchDatasetStatistics


@dataclass
class PatchDataset:
    """
    Collection of processed patches for one patient.
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    patient_id: str

    # ---------------------------------------------------------
    # Patch Collection
    # ---------------------------------------------------------

    patches: List[Patch] = field(default_factory=list)

    # ---------------------------------------------------------
    # Dataset Statistics
    # ---------------------------------------------------------

    statistics: Optional[PatchDatasetStatistics] = None

    # ---------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------

    @property
    def total_patches(self) -> int:
        """
        Total number of patches.
        """
        return len(self.patches)

    @property
    def is_empty(self) -> bool:
        """
        Returns True if no patches exist.
        """
        return len(self.patches) == 0
    
    @property
    def number_of_patches(self):
        return len(self.patches)

    @property
    def has_patches(self):
        return len(self.patches) > 0

    @property
    def patch_ids(self):
        """
        List of patch identifiers.
        """
        return [
            patch.patch_id
            for patch in self.patches
        ]

    @property
    def coordinates(self):
        """
        List of patch coordinates.
        """
        return [
            patch.coordinate
            for patch in self.patches
        ]

    @property
    def weak_labels(self):
        """
        Weak labels for every patch.
        """
        return [
            patch.weak_label
            for patch in self.patches
        ]

    @property
    def image_patches(self):
        """
        RGB image patches.
        """
        return [
            patch.image_patch
            for patch in self.patches
        ]

    @property
    def annotation_patches(self):
        """
        Annotation patches.
        """
        return [
            patch.annotation_patch
            for patch in self.patches
        ]

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------

    def add_patch(self, patch: Patch) -> None:
        """
        Add a patch to the collection.
        """
        self.patches.append(patch)

    def extend(self, patches: List[Patch]) -> None:
        """
        Add multiple patches.
        """
        self.patches.extend(patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self):
        return iter(self.patches)

    def __getitem__(self, index):
        return self.patches[index]