from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.patch.patch_labels import PatchLabels
from data.patch.patch_statistics import PatchStatistics
from data.patch.patch_metadata import PatchMetadata


@dataclass(slots=True)
class Patch:
    """
    Represents one extracted image patch.
    """

    image: np.ndarray

    mask: np.ndarray

    metadata: PatchMetadata

    labels: PatchLabels

    statistics: PatchStatistics

    @property
    def filename(self) -> str:
        """
        Standard filename used when exporting patches.
        """

        return (
            f"{self.metadata.patient_id}"
            f"_x{self.metadata.global_x}"
            f"_y{self.metadata.global_y}"
            f"_l{self.labels.binary_string}.png"
        )