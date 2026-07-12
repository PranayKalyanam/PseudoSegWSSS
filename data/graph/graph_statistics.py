from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GraphStatistics:
    """
    Stores descriptive statistics of a SpatialGraph.

    These statistics are computed immediately after graph construction
    and are intended for debugging, validation, visualization, and
    dataset analysis.

    Parameters
    ----------
    number_of_nodes : int
        Total number of graph nodes.

    number_of_edges : int
        Total number of graph edges.

    average_degree : float
        Average node degree.

    isolated_nodes : int
        Number of isolated graph nodes.

    connected_components : int
        Number of connected components.
    """

    number_of_nodes: int = 0

    number_of_edges: int = 0

    average_degree: float = 0.0

    isolated_nodes: int = 0

    connected_components: int = 0

    def __repr__(self) -> str:
        """
        Compact representation for debugging.
        """

        return (
            "GraphStatistics("
            f"nodes={self.number_of_nodes}, "
            f"edges={self.number_of_edges}, "
            f"avg_degree={self.average_degree:.2f}, "
            f"isolated={self.isolated_nodes}, "
            f"components={self.connected_components}"
            ")"
        )