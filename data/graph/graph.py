"""
graph.py

Graph dataclass used throughout the project.

No graph construction algorithms should be implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Dict
from typing import List

from .graph_edge import GraphEdge
from .graph_node import GraphNode


@dataclass(slots=True)
class Graph:
    """
    Graph representation.

    Stores nodes and edges only.
    """

    graph_id: str

    patient_id: str

    region_id: int

    nodes: List[GraphNode] = field(
        default_factory=list
    )

    edges: List[GraphEdge] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def number_of_nodes(self) -> int:
        return len(self.nodes)

    @property
    def number_of_edges(self) -> int:
        return len(self.edges)

    def add_node(
        self,
        node: GraphNode,
    ) -> None:
        self.nodes.append(node)

    def add_edge(
        self,
        edge: GraphEdge,
    ) -> None:
        self.edges.append(edge)