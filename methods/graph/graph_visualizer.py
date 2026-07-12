from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger


class GraphVisualizer:
    """
    Utility class for visualizing spatial graphs.

    The visualizer overlays graph nodes and edges on the original
    whole-slide image to verify graph construction.

    Future Extensions
    -----------------
    - Graph-only visualization
    - Node degree heatmap
    - Patch group visualization
    - Interactive visualization
    """

    NODE_RADIUS = 4
    NODE_COLOR = (0, 0, 255)      # Red
    EDGE_COLOR = (0, 255, 0)      # Green
    EDGE_THICKNESS = 1

    @classmethod
    def save(
        cls,
        patient,
        graph,
        output_dir: str | Path,
        logger=None,
    ) -> None:
        """
        Save graph visualization.

        Parameters
        ----------
        patient
            Patient object containing the original WSI.

        graph
            SpatialGraph object.

        output_dir
            Output directory.

        logger
            Optional logger.
        """

        logger = logger or get_logger(cls.__name__)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating graph visualization...")

        canvas = patient.image.copy()

        cls._draw_edges(
            canvas=canvas,
            graph=graph,
        )

        cls._draw_nodes(
            canvas=canvas,
            graph=graph,
        )

        output_path = output_dir / "graph_overlay.png"

        cv2.imwrite(
            str(output_path),
            canvas,
        )

        logger.info(
            "Graph visualization saved : %s",
            output_path,
        )

    # --------------------------------------------------
    # Private Methods
    # --------------------------------------------------

    @classmethod
    def _draw_nodes(
        cls,
        canvas: np.ndarray,
        graph,
    ) -> None:
        """
        Draw graph nodes.
        """

        for node in graph.nodes.values():

            coordinate = node.patch.coordinate

            center = (
                coordinate.x + coordinate.width // 2,
                coordinate.y + coordinate.height // 2,
            )

            cv2.circle(
                canvas,
                center,
                cls.NODE_RADIUS,
                cls.NODE_COLOR,
                thickness=-1,
            )

    @classmethod
    def _draw_edges(
        cls,
        canvas: np.ndarray,
        graph,
    ) -> None:
        """
        Draw graph edges.
        """

        for edge in graph.edges:

            source_coordinate = edge.source.patch.coordinate
            target_coordinate = edge.target.patch.coordinate

            p1 = (
                source_coordinate.x + source_coordinate.width // 2,
                source_coordinate.y + source_coordinate.height // 2,
            )

            p2 = (
                target_coordinate.x + target_coordinate.width // 2,
                target_coordinate.y + target_coordinate.height // 2,
            )

            cv2.line(
                canvas,
                p1,
                p2,
                cls.EDGE_COLOR,
                cls.EDGE_THICKNESS,
            )