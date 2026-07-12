from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SpatialGraph:
    """
    Represents the complete spatial graph of a single Whole Slide Image (WSI).

    A SpatialGraph consists of graph nodes, graph edges, an adjacency list,
    and graph-level statistics. Each graph corresponds to exactly one patient
    (or one WSI).

    Parameters
    ----------
    patient
        Patient object associated with this graph.

    nodes
        Dictionary mapping node_id -> GraphNode.

    edges
        List of GraphEdge objects.

    adjacency
        Graph adjacency list.

    statistics
        Graph statistics generated during graph construction.
    """

    patient: object

    nodes: dict[int, object] = field(
        default_factory=dict,
    )

    edges: list[object] = field(
        default_factory=list,
    )

    adjacency: dict[int, list[int]] = field(
        default_factory=dict,
    )

    statistics: object | None = None

    # --------------------------------------------------
    # Convenience Properties
    # --------------------------------------------------

    @property
    def number_of_nodes(self) -> int:
        """Total number of graph nodes."""
        return len(self.nodes)

    @property
    def number_of_edges(self) -> int:
        """Total number of graph edges."""
        return len(self.edges)

    @property
    def patient_id(self) -> str:
        """Patient identifier."""
        return self.patient.patient_id

    # --------------------------------------------------
    # Node Operations
    # --------------------------------------------------

    def get_node(
        self,
        node_id: int,
    ):
        """
        Retrieve a graph node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        GraphNode
        """

        return self.nodes[node_id]

    def has_node(
        self,
        node_id: int,
    ) -> bool:
        """
        Check whether a node exists.
        """

        return node_id in self.nodes

    # --------------------------------------------------
    # Edge Operations
    # --------------------------------------------------

    def neighbors(
        self,
        node_id: int,
    ) -> list:
        """
        Return neighboring GraphNode objects.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        list[GraphNode]
        """

        neighbor_ids = self.adjacency.get(
            node_id,
            [],
        )

        return [
            self.nodes[n]
            for n in neighbor_ids
        ]

    def degree(
        self,
        node_id: int,
    ) -> int:
        """
        Degree of a node.
        """

        return len(
            self.adjacency.get(
                node_id,
                [],
            )
        )

    def has_edge(
        self,
        source_id: int,
        target_id: int,
    ) -> bool:
        """
        Check whether two nodes are connected.
        """

        return (
            target_id in
            self.adjacency.get(
                source_id,
                [],
            )
        )

    # --------------------------------------------------
    # Graph Utilities
    # --------------------------------------------------

    def isolated_nodes(self) -> list:
        """
        Return all isolated graph nodes.
        """

        return [

            node

            for node_id, node in self.nodes.items()

            if self.degree(node_id) == 0

        ]

    def __len__(self) -> int:
        """
        Number of graph nodes.
        """

        return self.number_of_nodes

    def __contains__(
        self,
        node_id: int,
    ) -> bool:
        """
        Support:

        if node_id in graph
        """

        return self.has_node(
            node_id,
        )

    def __iter__(self):
        """
        Iterate over graph nodes.

        Example
        -------
        for node in graph:
            ...
        """

        return iter(
            self.nodes.values(),
        )

    def __repr__(self) -> str:
        """
        Compact representation for debugging.
        """

        return (
            "SpatialGraph("
            f"patient='{self.patient_id}', "
            f"nodes={self.number_of_nodes}, "
            f"edges={self.number_of_edges}"
            ")"
        )