from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from data.patch.patch_coordinate import PatchCoordinate
from data.patch.patch_label import PatchLabel


@dataclass(slots=True)
class PatchMetadata:
    """
    Metadata associated with one extracted patch.
    """

    patch_id: int

    patient_id: str

    patch_name: str

    image_path: Optional[Path]

    mask_path: Optional[Path]

    coordinate: PatchCoordinate

    tissue_region_id: int

    tissue_percentage: float

    magnification: float

    label: PatchLabel

    label_string: str = ""

    filename: str = ""

    generated: bool = False

    generator: str = ""

    detected_classes: List[int] = field(default_factory=list)

    class_statistics: Dict[int, Dict[str, float]] = field(
        default_factory=dict
    )

    additional_metadata: Dict[str, Any] = field(
        default_factory=dict
    )