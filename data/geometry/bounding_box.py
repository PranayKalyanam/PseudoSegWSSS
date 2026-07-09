from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box.

    Coordinates follow image convention:

        (x, y)
          ┌──────────────────┐
          │                  │
          │                  │
          └──────────────────┘
              width
              height
    """

    x: int
    y: int
    width: int
    height: int

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
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns (height, width).
        """
        return (
            self.height,
            self.width,
        )

    def contains(self, x: int, y: int) -> bool:
        """
        Returns True if the point lies inside the box.
        """

        return (
            self.left <= x < self.right
            and
            self.top <= y < self.bottom
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        """
        Returns (x, y, width, height).
        """

        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )

    @classmethod
    def from_tuple(
        cls,
        bbox: tuple[int, int, int, int],
    ) -> "BoundingBox":

        return cls(*bbox)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.width
        yield self.height