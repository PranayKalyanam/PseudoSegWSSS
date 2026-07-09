from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
from typing import List

import numpy as np


@dataclass(slots=True)
class PatchRecord:
    """
    Serializable dataset record.
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

    metadata: Dict = field(
        default_factory=dict
    )