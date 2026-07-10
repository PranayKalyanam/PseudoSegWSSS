"""
patch_metadata_generator.py

Generates metadata describing each extracted patch.

The PatchMetadataGenerator does not modify image data or labels.
Instead, it creates descriptive metadata that can later be used
by dataset serialization modules.

Responsibilities
----------------
1. Generate filenames.
2. Compute patch dimensions.
3. Compute patch area.
4. Record coordinate information.
5. Store metadata inside Patch objects.
"""

from typing import List

from data.patch.patch import Patch
from data.patch.patch_metadata import PatchMetadata


class PatchMetadataGenerator:
    """
    Generates metadata for extracted patches.
    """

    def __init__(
        self,
        image_extension: str = ".png",
        annotation_extension: str = ".png",
    ):

        self.image_extension = image_extension
        self.annotation_extension = annotation_extension

    # ---------------------------------------------------------

    def generate(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Generate metadata for every patch.
        """

        for patch in patches:

            metadata = self._create_metadata(
                patch
            )

            patch.metadata = metadata

        return patches

    # ---------------------------------------------------------

    def _create_metadata(
        self,
        patch: Patch,
    ) -> PatchMetadata:
        """
        Create metadata for a single patch.
        """

        coordinate = patch.coordinate

        image_height = patch.image_patch.shape[0]
        image_width = patch.image_patch.shape[1]

        annotation_height = patch.annotation_patch.shape[0]
        annotation_width = patch.annotation_patch.shape[1]

        metadata = PatchMetadata(

            # Identity
            patch_id=patch.patch_id,

            # Filenames
            image_filename=self._image_filename(
                patch.patch_id
            ),

            annotation_filename=self._annotation_filename(
                patch.patch_id
            ),

            # Coordinate
            coordinate=coordinate,

            # Image Information
            image_width=image_width,
            image_height=image_height,

            annotation_width=annotation_width,
            annotation_height=annotation_height,

            # Geometry
            patch_area=image_width * image_height,

            # Tissue
            tissue_percentage=patch.tissue_percentage,

            # Weak Labels
            weak_label=patch.weak_label,
            detected_classes=patch.detected_classes,

            # Statistics
            class_pixel_counts=patch.class_pixel_counts,
            class_percentages=patch.class_percentages,
        )

        return metadata

    # ---------------------------------------------------------

    def _image_filename(
        self,
        patch_id: int,
    ) -> str:

        return f"patch_{patch_id:06d}{self.image_extension}"

    # ---------------------------------------------------------

    def _annotation_filename(
        self,
        patch_id: int,
    ) -> str:

        return f"mask_{patch_id:06d}{self.annotation_extension}"