"""
graph_edge.py

Graph edge dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class GraphEdge:
    """
    Represents one graph edge.
    """

    source: int

    target: int

    weight: float = 1.0

    distance: Optional[float] = None