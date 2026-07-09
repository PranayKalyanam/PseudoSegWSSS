from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.patch.patch_metadata import PatchMetadata


@dataclass(slots=True)
class Patch:
    """
    Represents one extracted image patch.
    """

    image: np.ndarray

    mask: np.ndarray

    metadata: PatchMetadata

    status: str