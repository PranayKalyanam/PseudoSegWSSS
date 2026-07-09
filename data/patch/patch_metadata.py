from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.patch.patch_coordinate import PatchCoordinate


@dataclass(slots=True)
class PatchMetadata:
    """
    Metadata describing an extracted patch.

    This class stores only spatial and acquisition
    information. Labels and statistics are stored
    separately.
    """

    # --------------------------------------------------
    # Identity
    # --------------------------------------------------

    patch_id: int

    patient_id: str

    tissue_region_id: int

    # --------------------------------------------------
    # Source Files
    # --------------------------------------------------

    image_path: Path

    mask_path: Path

    # --------------------------------------------------
    # Spatial Information
    # --------------------------------------------------

    coordinate: PatchCoordinate

    width: int

    height: int

    # --------------------------------------------------
    # Acquisition Information
    # --------------------------------------------------

    magnification: float

    overlap: float

    patch_size: int

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    @property
    def global_x(self) -> int:
        return self.coordinate.global_x

    @property
    def global_y(self) -> int:
        return self.coordinate.global_y

    @property
    def filename_prefix(self) -> str:
        """
        Base filename before labels are appended.

        Example
        -------
        TCGA-001_x896_y448
        """

        return (
            f"{self.patient_id}"
            f"_x{self.global_x}"
            f"_y{self.global_y}"
        )