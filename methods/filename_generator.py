"""
==============================================================================
File        : filename_generator.py

Description :
    Utility class responsible for generating standardized filenames used
    throughout the preprocessing pipeline.

    This module centralizes all filename generation logic so that naming
    conventions remain consistent across patch extraction, weak label
    generation, graph generation, and dataset writing.

Responsibilities
----------------
1. Generate image patch filenames.
2. Generate mask filenames.
3. Generate graph filenames.
4. Generate metadata filenames.
5. Generate visualization filenames.
6. Provide helper methods for file extensions.

Author : Your Name
==============================================================================
"""

from __future__ import annotations

from pathlib import Path

from data.patch_metadata import PatchMetadata
from methods.patch_label_methods import PatchLabelMethods


class FilenameGenerator:
    """
    Generates standardized filenames for every preprocessing output.

    Naming Convention
    -----------------
    PatientID_Rxxx_Pxxxxxx_Xxxxx_Yyyyy_10X_L1100.png

    Example
    -------
    TCGA-A1-A0SK_R001_P000123_X012345_Y056789_10X_L1010.png
    """

    # ==========================================================================
    # Patch Image
    # ==========================================================================

    @staticmethod
    def generate_patch_filename(
        metadata: PatchMetadata,
        extension: str = ".png",
    ) -> str:
        """
        Generate patch image filename.

        Parameters
        ----------
        metadata : PatchMetadata

        extension : str

        Returns
        -------
        str
        """

        label_string = PatchLabelMethods.to_string(
            metadata.label
        )

        filename = (
            f"{metadata.patient_id}"
            f"_R{metadata.tissue_region_id:03d}"
            f"_P{metadata.patch_id:06d}"
            f"_X{metadata.coordinate.x:06d}"
            f"_Y{metadata.coordinate.y:06d}"
            f"_{int(metadata.magnification)}X"
            f"_L{label_string}"
            f"{extension}"
        )

        return filename

    # ==========================================================================
    # Patch Mask
    # ==========================================================================

    @staticmethod
    def generate_mask_filename(
        metadata: PatchMetadata,
        extension: str = ".png",
    ) -> str:
        """
        Generate patch mask filename.

        Parameters
        ----------
        metadata : PatchMetadata

        extension : str

        Returns
        -------
        str
        """

        stem = Path(
            FilenameGenerator.generate_patch_filename(
                metadata,
                extension=""
            )
        ).stem

        return f"{stem}_mask{extension}"

    # ==========================================================================
    # Graph
    # ==========================================================================

    @staticmethod
    def generate_graph_filename(
        metadata: PatchMetadata,
        extension: str = ".pt",
    ) -> str:
        """
        Generate graph filename.

        Parameters
        ----------
        metadata : PatchMetadata

        extension : str

        Returns
        -------
        str
        """

        stem = Path(
            FilenameGenerator.generate_patch_filename(
                metadata,
                extension=""
            )
        ).stem

        return f"{stem}_graph{extension}"

    # ==========================================================================
    # Metadata
    # ==========================================================================

    @staticmethod
    def generate_metadata_filename(
        metadata: PatchMetadata,
        extension: str = ".json",
    ) -> str:
        """
        Generate metadata filename.

        Parameters
        ----------
        metadata : PatchMetadata

        extension : str

        Returns
        -------
        str
        """

        stem = Path(
            FilenameGenerator.generate_patch_filename(
                metadata,
                extension=""
            )
        ).stem

        return f"{stem}_metadata{extension}"

    # ==========================================================================
    # Visualization
    # ==========================================================================

    @staticmethod
    def generate_visualization_filename(
        metadata: PatchMetadata,
        extension: str = ".png",
    ) -> str:
        """
        Generate visualization filename.

        Parameters
        ----------
        metadata : PatchMetadata

        extension : str

        Returns
        -------
        str
        """

        stem = Path(
            FilenameGenerator.generate_patch_filename(
                metadata,
                extension=""
            )
        ).stem

        return f"{stem}_visualization{extension}"

    # ==========================================================================
    # Label Text File
    # ==========================================================================

    @staticmethod
    def generate_label_filename(
        metadata: PatchMetadata,
        extension: str = ".txt",
    ) -> str:
        """
        Generate label filename.

        Parameters
        ----------
        metadata : PatchMetadata

        extension : str

        Returns
        -------
        str
        """

        stem = Path(
            FilenameGenerator.generate_patch_filename(
                metadata,
                extension=""
            )
        ).stem

        return f"{stem}_label{extension}"

    # ==========================================================================
    # Generic Helper
    # ==========================================================================

    @staticmethod
    def change_extension(
        filename: str,
        extension: str,
    ) -> str:
        """
        Change filename extension.

        Parameters
        ----------
        filename : str

        extension : str

        Returns
        -------
        str
        """

        return str(
            Path(filename).with_suffix(extension)
        )

    # ==========================================================================
    # Stem
    # ==========================================================================

    @staticmethod
    def get_stem(
        filename: str,
    ) -> str:
        """
        Return filename without extension.

        Parameters
        ----------
        filename : str

        Returns
        -------
        str
        """

        return Path(filename).stem