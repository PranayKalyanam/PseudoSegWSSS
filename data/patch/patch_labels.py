from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PatchLabels:
    """
    Weak multi-label assigned to a patch.

    Only clinically relevant semantic classes are used
    for training.
    """

    tumor: bool = False
    stroma: bool = False
    lymphocyte: bool = False
    necrosis: bool = False

    @property
    def binary_vector(self) -> tuple[int, int, int, int]:
        """
        Returns the labels as a binary vector.
        """
        return (
            int(self.tumor),
            int(self.stroma),
            int(self.lymphocyte),
            int(self.necrosis),
        )

    @property
    def binary_string(self) -> str:
        """
        Returns the labels as a filename-friendly string.

        Example
        -------
        1010
        """
        return "".join(map(str, self.binary_vector))

    @property
    def number_of_positive_labels(self) -> int:
        return sum(self.binary_vector)

    @property
    def is_empty(self) -> bool:
        return self.number_of_positive_labels == 0