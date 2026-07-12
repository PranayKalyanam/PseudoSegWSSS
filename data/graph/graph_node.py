from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphNode:
    """
    Represents a single node in the spatial graph.

    Each node corresponds to exactly one retained image patch.
    The node stores only graph-specific information while the
    associated Patch object remains the single source of truth
    for image, annotation, metadata and weak labels.

    Parameters
    ----------
    node_id : int
        Unique graph node identifier.

    patch : Patch
        Reference to the corresponding Patch object.

    neighbor_ids : list[int], optional
        Adjacent node identifiers.
    """

    node_id: int

    patch: object

    neighbor_ids: list[int] = field(
        default_factory=list,
    )

    # --------------------------------------------------
    # Convenience Properties
    # --------------------------------------------------

    @property
    def x(self) -> int:
        """Top-left x-coordinate."""
        return self.patch.coordinate.x

    @property
    def y(self) -> int:
        """Top-left y-coordinate."""
        return self.patch.coordinate.y

    @property
    def width(self) -> int:
        """Patch width."""
        return self.patch.coordinate.width

    @property
    def height(self) -> int:
        """Patch height."""
        return self.patch.coordinate.height

    @property
    def center(self) -> tuple[int, int]:
        """
        Center point of the patch.
        """
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )

    @property
    def weak_label(self):
        """Weak multi-label vector."""
        return self.patch.weak_label

    @property
    def metadata(self):
        """Patch metadata."""
        return self.patch.metadata

    @property
    def tissue_percentage(self) -> float:
        """Percentage of tissue within the patch."""
        return self.patch.tissue_percentage

    @property
    def detected_classes(self):
        """Detected semantic classes."""
        return self.patch.detected_classes

    # --------------------------------------------------
    # Graph Utilities
    # --------------------------------------------------

    @property
    def degree(self) -> int:
        """
        Number of neighboring nodes.
        """
        return len(self.neighbor_ids)

    def add_neighbor(
        self,
        neighbor_id: int,
    ) -> None:
        """
        Add a neighboring node.

        Parameters
        ----------
        neighbor_id : int
            Neighbor node identifier.
        """

        if neighbor_id not in self.neighbor_ids:
            self.neighbor_ids.append(
                neighbor_id,
            )

    def remove_neighbor(
        self,
        neighbor_id: int,
    ) -> None:
        """
        Remove a neighboring node.

        Parameters
        ----------
        neighbor_id : int
            Neighbor node identifier.
        """

        if neighbor_id in self.neighbor_ids:
            self.neighbor_ids.remove(
                neighbor_id,
            )

    def is_neighbor(
        self,
        neighbor_id: int,
    ) -> bool:
        """
        Check whether a node is adjacent.

        Parameters
        ----------
        neighbor_id : int

        Returns
        -------
        bool
        """

        return neighbor_id in self.neighbor_ids

    def __hash__(self) -> int:
        """
        Enable GraphNode usage in sets and dictionaries.
        """
        return hash(self.node_id)

    def __repr__(self) -> str:
        """
        Compact representation for debugging.
        """
        return (
            f"GraphNode("
            f"id={self.node_id}, "
            f"center={self.center}, "
            f"degree={self.degree}"
            f")"
        )
    

    def to_dict(self) -> dict:

        return {

            "node_id": self.node_id,

            "patch_id": self.patch.patch_id,

            "image_filename": self.metadata.image_filename,

            "mask_filename": self.metadata.annotation_filename,

            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,

            "center": {
                "x": self.center[0],
                "y": self.center[1],
            },

            "degree": self.degree,

            "neighbors": sorted(self.neighbor_ids),

            "weak_label": (
                self.weak_label.tolist()
                if hasattr(self.weak_label, "tolist")
                else self.weak_label
            ),

            "detected_classes": [
                int(c)
                for c in self.detected_classes],

            "tissue_percentage": self.tissue_percentage,
        }