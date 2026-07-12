from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PatchGroup:
    """
    Represents a graph-defined neighborhood of image patches.

    A PatchGroup is centered around a single GraphNode and contains
    all neighboring GraphNodes returned by the NeighborhoodFinder.
    During training, each PatchGroup serves as one training sample.

    Parameters
    ----------
    group_id : int
        Unique patch group identifier.

    center_node : GraphNode
        Central node of the neighborhood.

    member_nodes : list[GraphNode]
        Graph nodes belonging to this neighborhood.
    """

    group_id: int

    center_node: object

    member_nodes: list[object] = field(
        default_factory=list,
    )

    # --------------------------------------------------
    # Convenience Properties
    # --------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of nodes in the group.
        """
        return len(self.member_nodes)

    @property
    def center_patch(self):
        """
        Return the center patch.
        """
        return self.center_node.patch

    @property
    def patches(self) -> list:
        """
        Return all Patch objects in this group.
        """
        return [
            node.patch
            for node in self.member_nodes
        ]

    @property
    def node_ids(self) -> list[int]:
        """
        Return node identifiers.
        """
        return [
            node.node_id
            for node in self.member_nodes
        ]

    @property
    def image_patches(self) -> list:
        """
        Return image patches.
        """
        return [
            node.patch.image_patch
            for node in self.member_nodes
        ]

    @property
    def annotation_patches(self) -> list:
        """
        Return annotation patches.
        """
        return [
            node.patch.annotation_patch
            for node in self.member_nodes
        ]

    @property
    def weak_labels(self) -> list:
        """
        Return weak labels for all member patches.
        """
        return [
            node.patch.weak_label
            for node in self.member_nodes
        ]

    @property
    def detected_classes(self) -> list:
        """
        Return detected semantic classes.
        """
        return [
            node.patch.detected_classes
            for node in self.member_nodes
        ]

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def contains(
        self,
        node_id: int,
    ) -> bool:
        """
        Check whether a node belongs to this group.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        bool
        """

        return any(
            node.node_id == node_id
            for node in self.member_nodes
        )

    def get_node(
        self,
        node_id: int,
    ):
        """
        Retrieve a GraphNode by its identifier.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        GraphNode

        Raises
        ------
        KeyError
            If the node is not present.
        """

        for node in self.member_nodes:

            if node.node_id == node_id:
                return node

        raise KeyError(
            f"Node {node_id} not found in PatchGroup {self.group_id}."
        )

    def __len__(self) -> int:
        """
        Number of nodes in the group.
        """
        return self.size

    def __iter__(self):
        """
        Iterate over GraphNode objects.
        """
        return iter(self.member_nodes)

    def __contains__(
        self,
        node_id: int,
    ) -> bool:
        """
        Support:

        if node_id in patch_group
        """
        return self.contains(node_id)

    def __repr__(self) -> str:
        """
        Compact representation for debugging.
        """

        return (
            "PatchGroup("
            f"id={self.group_id}, "
            f"center={self.center_node.node_id}, "
            f"size={self.size}"
            ")"
        )