from __future__ import annotations

from collections import deque

from utils.logger import get_logger


class NeighborhoodFinder:
    """
    Finds graph neighborhoods centered around a given node.

    This class is responsible only for graph traversal. It does not
    construct PatchGroup objects or modify the graph.

    Current Implementation
    ----------------------
    - k-hop neighborhood (default: 1-hop)

    Future Extensions
    -----------------
    - Radius-based neighborhood
    - Adaptive neighborhood
    - Feature similarity neighborhood
    - Tissue-aware neighborhood
    """

    def __init__(
        self,
        config,
        logger=None,
    ) -> None:

        self.config = config
        self.logger = logger or get_logger(
            self.__class__.__name__,
        )

        #
        # Default neighborhood depth.
        # If not present in the config, use 1-hop.
        #
        self._k_hops = getattr(
            config,
            "graph_k_hops",
            1,
        )

    def find(
        self,
        center_node,
        graph,
    ) -> list:
        """
        Find the neighborhood around a center node.

        Parameters
        ----------
        center_node : GraphNode
            Node around which the neighborhood is generated.

        graph : SpatialGraph
            Spatial graph.

        Returns
        -------
        list[GraphNode]
            Nodes belonging to the neighborhood.
        """

        self.logger.debug(
            "Finding neighborhood for node %d",
            center_node.node_id,
        )

        #
        # Breadth-First Search
        #
        visited = set()

        queue = deque()

        queue.append(
            (
                center_node.node_id,
                0,
            )
        )

        visited.add(
            center_node.node_id,
        )

        neighborhood = []

        while queue:

            node_id, depth = queue.popleft()

            node = graph.nodes[node_id]

            neighborhood.append(
                node,
            )

            #
            # Stop expanding after k hops.
            #
            if depth >= self._k_hops:
                continue

            neighbors = graph.adjacency.get(
                node_id,
                [],
            )

            for neighbor_id in neighbors:

                if neighbor_id in visited:
                    continue

                visited.add(
                    neighbor_id,
                )

                queue.append(
                    (
                        neighbor_id,
                        depth + 1,
                    )
                )

        self.logger.debug(
            "Neighborhood size for node %d : %d",
            center_node.node_id,
            len(neighborhood),
        )

        return neighborhood