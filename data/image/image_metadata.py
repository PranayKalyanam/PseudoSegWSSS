"""
image_metadata.py

Stores metadata describing a loaded whole-slide image
and its corresponding annotation mask.

This class contains descriptive information only.
No image arrays are stored here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ImageMetadata:
    """
    Metadata describing the loaded image.

    Attributes
    ----------
    patient_id
        Unique patient identifier.

    width
        Original image width.

    height
        Original image height.

    channels
        Number of image channels.

    image_dtype
        NumPy image datatype.

    source_magnification
        Original magnification.

    target_magnification
        Working magnification.

    scale_factor
        Image scaling factor.

    working_width
        Width after resizing.

    working_height
        Height after resizing.
    """

    patient_id: str

    width: int

    height: int

    channels: int

    image_dtype: str

    source_magnification: float

    target_magnification: float

    scale_factor: float

    working_width: int

    working_height: int

    @property
    def original_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def working_shape(self) -> tuple[int, int]:
        return (self.working_height, self.working_width)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def is_resized(self) -> bool:
        return (
            self.width != self.working_width
            or
            self.height != self.working_height
        )