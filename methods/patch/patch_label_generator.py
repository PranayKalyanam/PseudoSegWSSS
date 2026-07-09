from __future__ import annotations

import numpy as np

from data.patch.patch_labels import PatchLabels


class PatchLabelGenerator:
    """
    Generates weak multi-label annotations from a
    segmentation mask.

    BCSS classes
    ------------
    1 : Tumor
    2 : Stroma
    3 : Lymphocyte
    4 : Necrosis

    Other classes are ignored.
    """

    # --------------------------------------------------
    # BCSS Semantic Labels
    # --------------------------------------------------

    TUMOR = 1
    STROMA = 2
    LYMPHOCYTE = 3
    NECROSIS = 4

    VALID_CLASSES = (
        TUMOR,
        STROMA,
        LYMPHOCYTE,
        NECROSIS,
    )

    @classmethod
    def generate(
        cls,
        mask: np.ndarray,
    ) -> PatchLabels:
        """
        Generate a PatchLabel from the segmentation mask.
        """

        unique_classes = set(
            np.unique(mask).tolist()
        )

        return PatchLabels(
            tumor=cls.TUMOR in unique_classes,
            stroma=cls.STROMA in unique_classes,
            lymphocyte=cls.LYMPHOCYTE in unique_classes,
            necrosis=cls.NECROSIS in unique_classes,
        )

    @classmethod
    def detected_classes(
        cls,
        mask: np.ndarray,
    ) -> list[int]:
        """
        Return detected semantic classes.
        """

        return sorted(
            list(
                set(np.unique(mask))
                &
                set(cls.VALID_CLASSES)
            )
        )

    @classmethod
    def binary_string(
        cls,
        label: PatchLabels,
    ) -> str:
        """
        Convert PatchLabel into a binary string.

        Order
        -----
        Tumor
        Stroma
        Lymphocyte
        Necrosis
        """

        return (
            f"{int(label.tumor)}"
            f"{int(label.stroma)}"
            f"{int(label.lymphocyte)}"
            f"{int(label.necrosis)}"
        )