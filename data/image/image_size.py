"""
image_size.py

Defines a lightweight data structure representing the
dimensions of an image.

This class contains no processing logic and is used
throughout the preprocessing pipeline wherever image
dimensions are required.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageSize:
    """
    Represents image dimensions.

    Attributes
    ----------
    width
        Image width in pixels.

    height
        Image height in pixels.
    """

    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns
        -------
        tuple[int, int]
            (height, width)
        """
        return self.height, self.width

    @property
    def area(self) -> int:
        """
        Returns
        -------
        int
            Total number of pixels.
        """
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """
        Returns
        -------
        float
            Width divided by height.
        """
        return self.width / self.height

    def scale(self, factor: float) -> "ImageSize":
        """
        Returns a scaled copy of the image size.
        """

        return ImageSize(
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
        )

    def __str__(self) -> str:
        return f"{self.width} × {self.height}"