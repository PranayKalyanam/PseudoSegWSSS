"""
patch_coordinate.py

Defines the PatchCoordinate dataclass.

A PatchCoordinate represents the spatial location of one patch
within the working whole slide image.

The class contains only geometric information and no image data.
It is reused throughout the preprocessing pipeline by coordinate
generation, patch extraction, metadata generation, and graph
construction.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchCoordinate:
    """
    Spatial coordinate of one patch.
    """

    # ---------------------------------------------------------
    # Position
    # ---------------------------------------------------------

    x: int

    y: int

    # ---------------------------------------------------------
    # Size
    # ---------------------------------------------------------

    width: int

    height: int

    # ---------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + (self.width / 2)

    @property
    def center_y(self) -> float:
        return self.y + (self.height / 2)

    @property
    def center(self):
        return (
            self.center_x,
            self.center_y,
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def shape(self):
        return (
            self.height,
            self.width,
        )

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------

    def to_tuple(self):
        """
        Returns (x, y, width, height).
        """
        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )

    def to_slice(self):
        """
        Returns NumPy slicing indices.

        Example
        -------
        image[
            coordinate.to_slice()
        ]
        """
        return (
            slice(self.y, self.bottom),
            slice(self.x, self.right),
        )