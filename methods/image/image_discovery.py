"""
image_discovery.py

Dataset image discovery utilities.

Responsible only for discovering valid image-mask pairs.
No image reading is performed here.
"""

from __future__ import annotations

from pathlib import Path

from data.image.image_pair import ImagePair


class ImageDiscovery:
    """
    Image discovery utilities.
    """

    VALID_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    }

    @staticmethod
    def discover_image_pairs(
        image_directory: Path,
        mask_directory: Path,
    ) -> list[ImagePair]:
        """
        Discovers all valid image-mask pairs.
        """

        image_files = {
            file.stem: file
            for file in image_directory.iterdir()
            if file.suffix.lower() in ImageDiscovery.VALID_EXTENSIONS
        }

        mask_files = {
            file.stem: file
            for file in mask_directory.iterdir()
            if file.suffix.lower() in ImageDiscovery.VALID_EXTENSIONS
        }

        common_ids = sorted(
            image_files.keys() & mask_files.keys()
        )

        pairs = []

        for patient_id in common_ids:

            pairs.append(
                ImagePair(
                    patient_id=patient_id,
                    image_path=image_files[patient_id],
                    mask_path=mask_files[patient_id],
                )
            )

        return pairs

    @staticmethod
    def count_images(directory: Path) -> int:
        """
        Returns the number of supported images.
        """

        return sum(
            file.suffix.lower() in ImageDiscovery.VALID_EXTENSIONS
            for file in directory.iterdir()
        )