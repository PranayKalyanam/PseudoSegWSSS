"""
image_pair.py

Dataclass representing one image-mask pair belonging
to a single patient.

This object is created during dataset discovery and is
used by the ImageLoader to locate the corresponding
image and annotation mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ImagePair:
    """
    Represents one patient image and its annotation mask.

    Attributes
    ----------
    patient_id
        Unique patient identifier.

    image_path
        Absolute path of RGB image.

    mask_path
        Absolute path of annotation mask.
    """

    patient_id: str

    image_path: Path

    mask_path: Path

    @property
    def image_filename(self) -> str:
        """
        Returns image filename.
        """

        return self.image_path.name

    @property
    def mask_filename(self) -> str:
        """
        Returns mask filename.
        """

        return self.mask_path.name

    @property
    def image_extension(self) -> str:
        """
        Image file extension.
        """

        return self.image_path.suffix.lower()

    @property
    def mask_extension(self) -> str:
        """
        Mask file extension.
        """

        return self.mask_path.suffix.lower()