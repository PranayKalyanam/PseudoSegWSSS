from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(slots=True, frozen=True)
class GraphEdge:
    """
    Represents an undirected spatial relationship between two graph nodes.

    Each edge connects two neighboring GraphNode objects in the spatial
    graph. The edge stores only graph-specific information and remains
    immutable after construction.

    Parameters
    ----------
    edge_id : int
        Unique edge identifier.

    source : GraphNode
        Source node.

    target : GraphNode
        Target node.

    direction : str
        Relative spatial direction from source to target.

    distance : float
        Euclidean distance between node centers.
    """

    edge_id: int

    source: object

    target: object

    direction: str

    distance: float

    # --------------------------------------------------
    # Convenience Properties
    # --------------------------------------------------

    @property
    def source_id(self) -> int:
        """Source node identifier."""
        return self.source.node_id

    @property
    def target_id(self) -> int:
        """Target node identifier."""
        return self.target.node_id

    @property
    def endpoints(self) -> tuple:
        """
        Return both endpoint nodes.
        """
        return (
            self.source,
            self.target,
        )

    @property
    def center_distance(self) -> float:
        """
        Compute the Euclidean distance between node centers.

        This value is computed dynamically to remain consistent even if
        node coordinates are updated in future implementations.
        """

        x1, y1 = self.source.center
        x2, y2 = self.target.center

        return sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    # --------------------------------------------------
    # Graph Utilities
    # --------------------------------------------------

    def contains(
        self,
        node_id: int,
    ) -> bool:
        """
        Check whether a node belongs to this edge.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        bool
        """

        return (
            node_id == self.source.node_id
            or
            node_id == self.target.node_id
        )

    def opposite(
        self,
        node,
    ):
        """
        Return the opposite endpoint.

        Parameters
        ----------
        node : GraphNode

        Returns
        -------
        GraphNode

        Raises
        ------
        ValueError
            If the supplied node does not belong to this edge.
        """

        if node.node_id == self.source.node_id:
            return self.target

        if node.node_id == self.target.node_id:
            return self.source

        raise ValueError(
            f"Node {node.node_id} does not belong to edge {self.edge_id}."
        )

    def __repr__(self) -> str:
        """
        Compact representation for debugging.
        """

        return (
            "GraphEdge("
            f"id={self.edge_id}, "
            f"{self.source.node_id} <-> {self.target.node_id}, "
            f"direction='{self.direction}', "
            f"distance={self.distance:.2f}"
            ")"
        )
    
    def to_dict(self) -> dict:
        """
        Convert the graph edge into a JSON-serializable dictionary.
        """

        return {

            "edge_id": self.edge_id,

            "source": self.source_id,

            "target": self.target_id,

            "direction": self.direction,

            "distance": self.distance,

        }