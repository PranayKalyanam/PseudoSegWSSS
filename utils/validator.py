"""
validator.py

Central validation module for the HistoGraphWSL project.

This validator is designed to be the single source of validation logic for the project. Instead of scattering checks across multiple preprocessing modules, every component calls the same validation routines. This provides consistent behavior, centralized maintenance, and clearer error reporting.

For example, the future slide_loader.py will simply do:

Validator.validate_wsi_path(slide_path)
Validator.validate_mask_path(mask_path)
Validator.validate_magnification(args.magnification)

Responsibilities
----------------
1. Validate files and directories.
2. Validate WSI and mask paths.
3. Validate preprocessing parameters.
4. Validate patch coordinates.
5. Validate graph inputs.
6. Provide reusable validation routines across the project.

Every preprocessing module should use this validator
instead of implementing its own validation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from utils.exceptions import (
    ConfigurationError,
    ValidationError,
    WSINotFoundError,
    MaskNotFoundError,
    OutputDirectoryError,
    UnsupportedSlideFormatError,
    InvalidMagnificationError,
    InvalidPatchCoordinateError,
    EmptyGraphError,
)


SUPPORTED_WSI_EXTENSIONS = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".png",
    ".jpg"
}

class Validator:
    """
    Central validation class.

    All methods are static because validation
    maintains no internal state.
    """

    # ==========================================================
    # Directories
    # ==========================================================

    @staticmethod
    def validate_directory(directory: Path) -> None:
        """
        Validate that a directory exists.

        Parameters
        ----------
        directory : Path

        Raises
        ------
        FileNotFoundError
        """

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist:\n{directory}"
            )

        if not directory.is_dir():
            raise ValidationError(
                f"Expected directory but found file:\n{directory}"
            )

    @staticmethod
    def validate_output_directory(directory: Path) -> None:
        """
        Create output directory if necessary.

        Raises
        ------
        OutputDirectoryError
        """

        try:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        except Exception as exc:
            raise OutputDirectoryError(
                f"Cannot create output directory:\n{directory}"
            ) from exc

    # ==========================================================
    # WSI
    # ==========================================================

    @staticmethod
    def validate_wsi_path(slide_path: Path) -> None:
        """
        Validate WSI path.

        Raises
        ------
        WSINotFoundError
        """

        if not slide_path.exists():
            raise WSINotFoundError(
                f"WSI not found:\n{slide_path}"
            )

        extension = slide_path.suffix.lower()

        if extension not in SUPPORTED_WSI_EXTENSIONS:
            raise UnsupportedSlideFormatError(
                f"Unsupported WSI format: {extension}"
            )

    # ==========================================================
    # Mask
    # ==========================================================

    @staticmethod
    def validate_mask_path(mask_path: Path) -> None:
        """
        Validate mask path.
        """

        if not mask_path.exists():
            raise MaskNotFoundError(
                f"Mask not found:\n{mask_path}"
            )

    # ==========================================================
    # Magnification
    # ==========================================================

    @staticmethod
    def validate_magnification(level: int) -> None:
        """
        Validate requested magnification.
        """

        allowed = {
            2,
            5,
            10,
            20,
            40,
        }

        if level not in allowed:
            raise InvalidMagnificationError(
                f"Unsupported magnification: {level}x"
            )

    # ==========================================================
    # Patch
    # ==========================================================

    @staticmethod
    def validate_patch_size(size: int) -> None:
        """
        Validate patch size.
        """

        if size <= 0:
            raise ConfigurationError(
                "Patch size must be greater than zero."
            )

    @staticmethod
    def validate_overlap(overlap: float) -> None:
        """
        Validate overlap ratio.
        """

        if overlap < 0 or overlap >= 1:
            raise ConfigurationError(
                "Overlap must satisfy 0 <= overlap < 1."
            )

    @staticmethod
    def validate_patch_coordinate(
        x: int,
        y: int,
        width: int,
        height: int,
        patch_size: int,
    ) -> None:
        """
        Validate patch coordinate.

        Raises
        ------
        InvalidPatchCoordinateError
        """

        if x < 0 or y < 0:
            raise InvalidPatchCoordinateError(
                "Negative patch coordinate."
            )

        if x + patch_size > width:
            raise InvalidPatchCoordinateError(
                "Patch exceeds slide width."
            )

        if y + patch_size > height:
            raise InvalidPatchCoordinateError(
                "Patch exceeds slide height."
            )

    # ==========================================================
    # Graph
    # ==========================================================

    @staticmethod
    def validate_graph_nodes(
        coordinates: Sequence,
    ) -> None:
        """
        Ensure graph has nodes.
        """

        if len(coordinates) == 0:
            raise EmptyGraphError(
                "Graph contains no nodes."
            )

    # ==========================================================
    # Labels
    # ==========================================================

    @staticmethod
    def validate_label_vector(
        label: Iterable[int],
        expected_length: int = 4,
    ) -> None:
        """
        Validate binary label vector.
        """

        label = list(label)

        if len(label) != expected_length:
            raise ValidationError(
                f"Expected {expected_length} labels "
                f"but received {len(label)}."
            )

        for value in label:

            if value not in (0, 1):
                raise ValidationError(
                    "Labels must be binary."
                )

    # ==========================================================
    # Generic
    # ==========================================================

    @staticmethod
    def validate_probability(value: float) -> None:
        """
        Validate probability.
        """

        if value < 0 or value > 1:
            raise ValidationError(
                "Probability must lie in [0,1]."
            )
            
            
    
        # ==========================================================
    # Generic Type Validation
    # ==========================================================

    @staticmethod
    def validate_type(
        value,
        expected_type,
        name: str = "value",
    ) -> None:
        """
        Validate that a value is of the expected type.

        Parameters
        ----------
        value
            Object to validate.

        expected_type
            Expected Python type or tuple of types.

        name
            Parameter name used in the error message.

        Raises
        ------
        ValidationError
        """

        if not isinstance(value, expected_type):

            if isinstance(expected_type, tuple):
                expected = ", ".join(
                    t.__name__ for t in expected_type
                )
            else:
                expected = expected_type.__name__

            raise ValidationError(
                f"{name} must be of type "
                f"{expected}, "
                f"received {type(value).__name__}."
            )

    # ==========================================================
    # None Validation
    # ==========================================================

    @staticmethod
    def validate_not_none(
        value,
        name: str = "value",
    ) -> None:
        """
        Ensure value is not None.
        """

        if value is None:
            raise ValidationError(
                f"{name} cannot be None."
            )

    # ==========================================================
    # Numeric Validation
    # ==========================================================

    @staticmethod
    def validate_positive_int(
        value: int,
        name: str = "value",
    ) -> None:
        """
        Validate positive integer.
        """

        Validator.validate_type(
            value,
            int,
            name,
        )

        if value <= 0:
            raise ValidationError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def validate_non_negative_int(
        value: int,
        name: str = "value",
    ) -> None:
        """
        Validate non-negative integer.
        """

        Validator.validate_type(
            value,
            int,
            name,
        )

        if value < 0:
            raise ValidationError(
                f"{name} must be non-negative."
            )

    # ==========================================================
    # Range Validation
    # ==========================================================

    @staticmethod
    def validate_range(
        value: float,
        minimum: float,
        maximum: float,
        name: str = "value",
    ) -> None:
        """
        Validate numeric range.
        """

        if value < minimum or value > maximum:
            raise ValidationError(
                f"{name} must be in "
                f"[{minimum}, {maximum}]."
            )
            
            """
            TO Do: 
            ==========================
            
            As the project grows, additional validation methods (such as validating WSI–mask alignment, graph connectivity, or metadata consistency) can be added to this module without modifying the rest of the codebase. This keeps the preprocessing pipeline clean, consistent, and easy to maintain.
            """