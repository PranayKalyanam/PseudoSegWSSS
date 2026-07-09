"""
label_utils.py

Utility functions for working with weak labels.
"""

from __future__ import annotations

from data.patch_label import PatchLabel


class LabelUtils:
    """
    Helper methods for PatchLabel.
    """

    @staticmethod
    def label_string(
        label: PatchLabel,
    ) -> str:
        """
        Convert a PatchLabel into a binary string.

        Example
        -------
        PatchLabel(1,0,1,0)

        returns

        "1010"
        """

        return (
            f"{label.tumor}"
            f"{label.stroma}"
            f"{label.lymphocyte}"
            f"{label.necrosis}"
        )

    @staticmethod
    def to_list(
        label: PatchLabel,
    ) -> list[int]:
        """
        Convert PatchLabel to list.
        """

        return [
            label.tumor,
            label.stroma,
            label.lymphocyte,
            label.necrosis,
        ]

    @staticmethod
    def has_positive_label(
        label: PatchLabel,
    ) -> bool:
        """
        Check whether any class is present.
        """

        return any(
            LabelUtils.to_list(label)
        )

    @staticmethod
    def number_of_positive_classes(
        label: PatchLabel,
    ) -> int:
        """
        Count positive classes.
        """

        return sum(
            LabelUtils.to_list(label)
        )