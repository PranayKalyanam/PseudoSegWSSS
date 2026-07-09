# data/dataset_record.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Dict


@dataclass(slots=True)
class DatasetRecord:
    """
    Serializable dataset record.

    This object contains only metadata required for
    CSV/JSON manifest generation.

    It does not store image or mask arrays.
    """

    patch_id: int

    patient_id: str

    region_id: int

    filename: str

    x: int

    y: int

    width: int

    height: int

    tissue_percentage: float

    tumor: int

    stroma: int

    lymphocyte: int

    necrosis: int

    label_string: str

    image_path: str

    mask_path: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )