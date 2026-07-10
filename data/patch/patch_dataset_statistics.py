"""
patch_dataset_statistics.py

Defines the PatchDatasetStatistics dataclass.

This class stores aggregate statistics for all processed patches
belonging to a single patient. It contains only summary
information and no individual patch data.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PatchDatasetStatistics:
    """
    Aggregate statistics for one PatchDataset.
    """

    # ---------------------------------------------------------
    # Patch Statistics
    # ---------------------------------------------------------

    total_patches: int = 0

    average_tissue_percentage: float = 0.0

    average_patch_area: float = 0.0

    # ---------------------------------------------------------
    # Class Statistics
    # ---------------------------------------------------------

    total_classes: Dict[int, int] = None

    class_distribution: Dict[int, int] = None

    # ---------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------

    @property
    def average_tissue_ratio(self) -> float:
        """
        Alias for average tissue percentage.
        """
        return self.average_tissue_percentage

    @property
    def total_class_instances(self) -> int:
        """
        Total number of semantic class occurrences.
        """
        if not self.class_distribution:
            return 0

        return sum(
            self.class_distribution.values()
        )

    @property
    def number_of_classes(self) -> int:
        """
        Number of unique semantic classes.
        """
        if not self.class_distribution:
            return 0

        return len(
            self.class_distribution
        )

    @property
    def average_patch_size(self) -> float:
        """
        Alias for average patch area.
        """
        return self.average_patch_area

    @property
    def is_empty(self) -> bool:
        """
        Returns True if no patches were generated.
        """
        return self.total_patches == 0

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------

    def contains_class(
        self,
        class_id: int,
    ) -> bool:
        """
        Returns True if the specified class
        exists in the dataset.
        """

        if not self.class_distribution:
            return False

        return class_id in self.class_distribution

    def class_count(
        self,
        class_id: int,
    ) -> int:
        """
        Returns the number of patches containing
        the specified semantic class.
        """

        if not self.class_distribution:
            return 0

        return self.class_distribution.get(
            class_id,
            0,
        )