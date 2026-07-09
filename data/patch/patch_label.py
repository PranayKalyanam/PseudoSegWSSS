from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PatchLabel:
    """
    Weak multi-label assigned to a patch.
    """

    tumor: int = 0

    stroma: int = 0

    lymphocyte: int = 0

    necrosis: int = 0