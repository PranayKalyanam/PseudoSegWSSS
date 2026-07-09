from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PatchCoordinate:
    """
    Spatial location of a patch inside the slide.
    """

    x: int

    y: int

    width: int

    height: int