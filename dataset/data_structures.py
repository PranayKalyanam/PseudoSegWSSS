"""
Common data structures used throughout the preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(slots=True)
class PatchRecord:
    """
    Represents one extracted image patch and its metadata.
    """

    patch_id: str
    patient_id: str
    region_id: int
    filename: str

    image: np.ndarray
    mask: np.ndarray

    x: int
    y: int
    width: int
    height: int

    tissue_percentage: float

    labels: List[int]

    metadata: Dict = field(default_factory=dict)

    @property
    def label_string(self) -> str:
        return "".join(map(str, self.labels))

    @property
    def has_positive_label(self) -> bool:
        return any(self.labels)

    @property
    def coordinate(self):
        return (self.x, self.y)
