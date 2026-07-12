from __future__ import annotations

from utils.logger import get_logger

from data.graph.graph_edge import GraphEdge


class NeighborFinder:
    """
    Identifies neighboring graph nodes using their spatial coordinates.

    Neighbor relationships are determined directly from the patch
    extraction grid. Since patches are extracted with a fixed stride,
    neighbor lookup is performed through a coordinate hash table rather
    than pairwise distance computation.

    Notes
    -----
    Current implementation:
        • 4-neighborhood

    Future implementations:
        • 8-neighborhood
        • Radius graph
        • k-NN graph
        • Adaptive graph
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

        self._stride = int(
            config.patch_size * (1.0 - config.overlap)
        )

    def find(
        self,
        nodes: dict[int, object],
        coordinate_lookup: dict[tuple[int, int], object],
    ):
        """
        Find neighboring nodes.

        Parameters
        ----------
        nodes
            Dictionary of GraphNode objects.

        coordinate_lookup
            Mapping:
                (x, y) -> GraphNode

        Returns
        -------
        tuple

            edges

            adjacency
        """

        self.logger.info(
            "Finding graph neighbors..."
        )

        edges = []

        adjacency = {
            node.node_id: []
            for node in nodes.values()
        }

        edge_id = 0

        for node in nodes.values():

            x = node.patch.coordinate.x
            y = node.patch.coordinate.y

            neighbors = [

                ("LEFT", (x - self._stride, y)),

                ("RIGHT", (x + self._stride, y)),

                ("TOP", (x, y - self._stride)),

                ("BOTTOM", (x, y + self._stride)),
            ]

            for direction, coordinate in neighbors:

                neighbor = coordinate_lookup.get(
                    coordinate
                )

                if neighbor is None:
                    continue

                #
                # Avoid duplicate edges
                #

                if neighbor.node_id < node.node_id:
                    continue

                edge = GraphEdge(

                    edge_id=edge_id,

                    source=node,

                    target=neighbor,

                    direction=direction,

                    distance=self._stride,
                )

                edge_id += 1

                edges.append(
                    edge,
                )

                adjacency[node.node_id].append(
                    neighbor.node_id,
                )

                adjacency[neighbor.node_id].append(
                    node.node_id,
                )

        self.logger.info(
            "Neighbor search completed."
        )

        self.logger.info(
            "Edges created : %d",
            len(edges),
        )

        return (
            edges,
            adjacency,
        )