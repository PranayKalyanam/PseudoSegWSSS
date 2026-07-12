"""
{
    "graph_information": {
        "number_of_nodes": 40,
        "number_of_edges": 67,
        "average_degree": 3.35,
        "connected_components": 1,
        "isolated_nodes": 0
    },

    "nodes": [
        {
            "node_id": 0,
            "patch_id": 0,
            "image_filename": "TCGA_x0_y0_label13_img.png",
            "mask_filename": "TCGA_x0_y0_label13_mask.png",
            "x": 0,
            "y": 0,
            "width": 224,
            "height": 224,
            "neighbors": [1, 8]
        },
        {
            "node_id": 1,
            "patch_id": 1,
            "image_filename": "...",
            "mask_filename": "...",
            "x": 112,
            "y": 0,
            "width": 224,
            "height": 224,
            "neighbors": [0, 2, 9]
        }
    ],

    "edges": [
        {
            "source": 0,
            "target": 1
        },
        {
            "source": 0,
            "target": 8
        }
    ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

from data.graph.spatial_graph import SpatialGraph
from utils.logger import get_logger


class GraphSerializer:
    """
    Serializes a SpatialGraph into a JSON file.

    Output
    ------
    patient_folder/
        graph/
            graph.json

    The serializer stores only graph metadata and topology.
    Image patches are never duplicated.
    """

    def __init__(
        self,
        logger=None,
        indent: int = 4,
    ) -> None:

        self.logger = logger or get_logger(
            self.__class__.__name__,
        )

        self.indent = indent

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def save(
        self,
        graph: SpatialGraph,
        output_directory: str | Path,
    ) -> Path:
        """
        Save SpatialGraph as graph.json.

        Parameters
        ----------
        graph:
            Spatial graph to serialize.

        output_directory:
            Patient output directory.

        Returns
        -------
        Path
            Path to graph.json
        """

        output_directory = Path(output_directory)

        graph_directory = output_directory / "graph"
        graph_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = graph_directory / "graph.json"

        self.logger.info(
            "Saving graph : %s",
            output_file,
        )

        graph_dict = self._graph_to_dict(
            graph,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                graph_dict,
                fp,
                indent=self.indent,
            )

        self.logger.info(
            "Graph saved successfully."
        )

        return output_file

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _graph_to_dict(
        self,
        graph: SpatialGraph,
    ) -> dict:
        
        graph_dict = {

            "patient_id": graph.patient,

            "number_of_nodes": graph.number_of_nodes,

            "number_of_edges": graph.number_of_edges,

            "statistics": {

                "number_of_nodes": graph.statistics.number_of_nodes,
                "number_of_edges": graph.statistics.number_of_edges,
                "average_degree": graph.statistics.average_degree,
                "isolated_nodes": graph.statistics.isolated_nodes,
                "connected_components": graph.statistics.connected_components,

            },

            "nodes": [
                node.to_dict()
                for node in graph
            ],

            "edges": [
                edge.to_dict()
                for edge in graph.edges
            ],
        }

        return graph_dict