from __future__ import annotations

from typing import Dict

from data.graph.graph_statistics import GraphStatistics
from utils.logger import get_logger

from methods.graph.neighbor_finder import NeighborFinder

from data.graph.graph_node import GraphNode
from data.graph.graph_edge import GraphEdge
from data.graph.spatial_graph import SpatialGraph


class GraphBuilder:
    """
    Builds a spatial graph from a PatchDataset.

    This class coordinates graph construction by creating graph nodes,
    identifying neighboring patches, creating graph edges, and assembling
    the final SpatialGraph object.

    Notes
    -----
    This class intentionally delegates neighborhood computation to
    NeighborFinder to maintain separation of responsibilities.
    """

    def __init__(
        self,
        config,
        logger=None,
    ) -> None:

        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)

        self._neighbor_finder = NeighborFinder(
            config=config,
            logger=self.logger,
        )

    def build(
        self,
        dataset,
    ) -> SpatialGraph:
        """
        Construct a spatial graph from a PatchDataset.

        Parameters
        ----------
        dataset : PatchDataset

        Returns
        -------
        SpatialGraph
        """

        self.logger.info("Building spatial graph...")

        # -------------------------------------------------
        # Step 1
        # Create graph nodes
        # -------------------------------------------------

        nodes = self._create_nodes(dataset)

        # -------------------------------------------------
        # Step 2
        # Coordinate lookup
        # -------------------------------------------------

        coordinate_lookup = self._build_coordinate_lookup(nodes)

        # -------------------------------------------------
        # Step 3
        # Find neighbors
        # -------------------------------------------------

        edges, adjacency = self._neighbor_finder.find(
            nodes=nodes,
            coordinate_lookup=coordinate_lookup,
        )

        # -------------------------------------------------
        # Step 4
        # Assemble graph
        # -------------------------------------------------

        graph = SpatialGraph(
            patient=dataset.patient_id,
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
        )
        graph.statistics = self._compute_statistics(
            graph,
        )

        self.logger.info(
            "Graph constructed successfully "
            "(nodes=%d, edges=%d)",
            len(nodes),
            len(edges),
        )

        return graph

    def _create_nodes(
        self,
        dataset,
    ) -> dict[int, GraphNode]:
        """
        Create one GraphNode for every patch in the dataset.

        Parameters
        ----------
        dataset : PatchDataset

        Returns
        -------
        dict[int, GraphNode]
            Mapping:
                node_id -> GraphNode
        """

        self.logger.info("Creating graph nodes...")

        nodes: dict[int, GraphNode] = {}

        for node_id, patch in enumerate(dataset.patches):

            node = GraphNode(
                node_id=node_id,
                patch=patch,
            )

            nodes[node_id] = node

        self.logger.info(
            "Graph nodes created : %d",
            len(nodes),
        )

        return nodes

    def _build_coordinate_lookup(
        self,
        nodes: dict[int, GraphNode],
    ) -> dict[tuple[int, int], GraphNode]:
        """
        Build a coordinate lookup table.

        Parameters
        ----------
        nodes : dict[int, GraphNode]

        Returns
        -------
        dict[(x, y), GraphNode]
        """

        self.logger.info(
            "Building coordinate lookup table..."
        )

        coordinate_lookup = {}

        for node in nodes.values():

            coordinate = (
                node.patch.coordinate.x,
                node.patch.coordinate.y,
            )

            coordinate_lookup[coordinate] = node

        self.logger.info(
            "Coordinate lookup entries : %d",
            len(coordinate_lookup),
        )

        return coordinate_lookup
    
    def _compute_statistics(
        self,
        graph: SpatialGraph,
    ) -> GraphStatistics:
        """
        Compute graph-level statistics.

        Parameters
        ----------
        graph : SpatialGraph

        Returns
        -------
        GraphStatistics
        """

        self.logger.info(
            "Computing graph statistics..."
        )

        number_of_nodes = graph.number_of_nodes

        number_of_edges = graph.number_of_edges

        degrees = [
            len(neighbors)
            for neighbors in graph.adjacency.values()
        ]

        average_degree = (
            sum(degrees) / number_of_nodes
            if number_of_nodes > 0
            else 0.0
        )

        isolated_nodes = sum(
            degree == 0
            for degree in degrees
        )

        connected_components = self._count_connected_components(
            graph,
        )

        statistics = GraphStatistics(
            number_of_nodes=number_of_nodes,
            number_of_edges=number_of_edges,
            average_degree=average_degree,
            isolated_nodes=isolated_nodes,
            connected_components=connected_components,
        )

        self.logger.info(
            "%s",
            statistics,
        )

        return statistics
    
    def _count_connected_components(
        self,
        graph: SpatialGraph,
    ) -> int:
        """
        Count connected components using Depth-First Search.

        Parameters
        ----------
        graph : SpatialGraph

        Returns
        -------
        int
        """

        visited = set()

        components = 0

        for node_id in graph.nodes:

            if node_id in visited:
                continue

            components += 1

            stack = [node_id]

            while stack:

                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)

                stack.extend(
                    graph.adjacency.get(
                        current,
                        [],
                    )
                )

        return components