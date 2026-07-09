from __future__ import annotations

from pathlib import Path

from data.patch.patch_labels import PatchLabels


class PatchNaming:
    """
    Generates standardized filenames for extracted patches.

    Naming convention
    -----------------
    OriginalFilename_xGlobal_yGlobal_BinaryLabels.png

    Example
    -------
    TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_x14562_y27891_1010.png
    """

    @staticmethod
    def binary_label_string(
        label: PatchLabels,
    ) -> str:
        """
        Convert PatchLabel into a binary label string.

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

    @classmethod
    def generate(
        cls,
        original_filename: str,
        global_x: int,
        global_y: int,
        label: PatchLabels,
    ) -> str:
        """
        Generate standardized patch filename.
        """

        stem = Path(original_filename).stem

        label_string = cls.binary_label_string(label)

        return (
            f"{stem}"
            f"_x{global_x}"
            f"_y{global_y}"
            f"_{label_string}.png"
        )

    @classmethod
    def generate_without_label(
        cls,
        original_filename: str,
        global_x: int,
        global_y: int,
    ) -> str:
        """
        Generate filename before labels are available.
        """

        stem = Path(original_filename).stem

        return (
            f"{stem}"
            f"_x{global_x}"
            f"_y{global_y}.png"
        )