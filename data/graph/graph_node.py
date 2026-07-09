"""
graph_node.py

Graph node dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Dict

import numpy as np


@dataclass(slots=True)
class GraphNode:
    """
    Represents one graph node.

    Usually corresponds to one image patch.
    """

    node_id: int

    patch_id: int

    x: float

    y: float

    feature: np.ndarray

    label: np.ndarray

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def position(self):
        return (self.x, self.y)