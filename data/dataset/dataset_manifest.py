"""
dataset_manifest.py

Data structure describing a generated dataset.

The DatasetManifest represents the final output of the
preprocessing pipeline. It stores dataset-level metadata
required for dataset serialization, reproducibility,
and subsequent model training.

This class contains no processing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class DatasetManifest:
    """
    Dataset-level metadata.

    Attributes
    ----------
    dataset_name
        Name of the dataset.

    number_of_patients
        Total number of patients.

    number_of_patches
        Total extracted patches.

    number_of_graphs
        Total generated graphs.

    patch_size
        Patch size in pixels.

    overlap
        Sliding-window overlap ratio.

    source_magnification
        Original WSI magnification.

    target_magnification
        Working magnification.

    class_names
        Ordered semantic class names.

    created_at
        Dataset creation timestamp.
    """

    dataset_name: str

    number_of_patients: int = 0

    number_of_patches: int = 0

    number_of_graphs: int = 0

    patch_size: int = 224

    overlap: float = 0.50

    source_magnification: float = 40.0

    target_magnification: float = 10.0

    class_names: list[str] = field(default_factory=list)

    created_at: str = field(
        default_factory=lambda: datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    @property
    def number_of_classes(self) -> int:
        """
        Returns
        -------
        int
            Total semantic classes.
        """
        return len(self.class_names)

    @property
    def patch_overlap_percentage(self) -> float:
        """
        Returns overlap percentage.
        """
        return self.overlap * 100

    def to_dict(self) -> dict:
        """
        Convert manifest into a dictionary suitable for
        JSON or YAML serialization.
        """

        return {
            "dataset_name": self.dataset_name,
            "number_of_patients": self.number_of_patients,
            "number_of_patches": self.number_of_patches,
            "number_of_graphs": self.number_of_graphs,
            "patch_size": self.patch_size,
            "overlap": self.overlap,
            "source_magnification": self.source_magnification,
            "target_magnification": self.target_magnification,
            "class_names": self.class_names,
            "number_of_classes": self.number_of_classes,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset='{self.dataset_name}', "
            f"patients={self.number_of_patients}, "
            f"patches={self.number_of_patches}, "
            f"graphs={self.number_of_graphs})"
        )