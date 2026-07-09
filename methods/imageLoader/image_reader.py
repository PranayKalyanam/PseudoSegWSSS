"""
image_reader.py

Image reading utilities for the HistoGraphWSL preprocessing pipeline.

This module is responsible for

    • reading RGB images
    • reading annotation masks
    • validating loaded images

Only image I/O is implemented here.

No preprocessing, resizing, alignment or magnification
logic belongs in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils.validator import Validator
from utils.exceptions import ValidationError


# =============================================================================
# RGB Image Reader
# =============================================================================

def read_rgb(
    image_path: Path,
) -> np.ndarray:
    """
    Read an RGB image.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    np.ndarray
        RGB image.
    """

    Validator.validate_wsi_path(
        image_path
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise ValidationError(
            f"Unable to read image:\n{image_path}"
        )

    #
    # OpenCV loads BGR.
    #

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return image


# =============================================================================
# Mask Reader
# =============================================================================

def read_mask(
    mask_path: Path,
) -> np.ndarray:
    """
    Read annotation mask.

    Parameters
    ----------
    mask_path : Path

    Returns
    -------
    np.ndarray
        Annotation mask.
    """

    Validator.validate_mask_path(
        mask_path
    )

    #
    # Preserve original values.
    #

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED,
    )

    if mask is None:

        raise ValidationError(
            f"Unable to read mask:\n{mask_path}"
        )

    return mask


# =============================================================================
# Generic Image Reader
# =============================================================================

def read_image(
    file_path: Path,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> np.ndarray:
    """
    Generic OpenCV image reader.

    Parameters
    ----------
    file_path : Path

    flags : int

    Returns
    -------
    np.ndarray
    """

    Validator.validate_not_none(
        file_path,
        "file_path",
    )

    image = cv2.imread(
        str(file_path),
        flags,
    )

    if image is None:

        raise ValidationError(
            f"Unable to read image:\n{file_path}"
        )

    return image


# =============================================================================
# Image Information
# =============================================================================

def image_shape(
    image: np.ndarray,
) -> Tuple[int, ...]:
    """
    Return image shape.

    Parameters
    ----------
    image

    Returns
    -------
    tuple
    """

    validate_image(
        image
    )

    return image.shape


def image_height(
    image: np.ndarray,
) -> int:
    """
    Return image height.
    """

    validate_image(
        image
    )

    return image.shape[0]


def image_width(
    image: np.ndarray,
) -> int:
    """
    Return image width.
    """

    validate_image(
        image
    )

    return image.shape[1]


def number_of_channels(
    image: np.ndarray,
) -> int:
    """
    Return number of image channels.

    Returns
    -------
    int
    """

    validate_image(
        image
    )

    if image.ndim == 2:
        return 1

    return image.shape[2]


# =============================================================================
# Validation
# =============================================================================

def validate_image(
    image: np.ndarray,
) -> None:
    """
    Validate loaded image.

    Parameters
    ----------
    image

    Raises
    ------
    ValidationError
    """

    Validator.validate_not_none(
        image,
        "image",
    )

    if not isinstance(
        image,
        np.ndarray,
    ):

        raise ValidationError(
            "Image must be a NumPy array."
        )

    if image.size == 0:

        raise ValidationError(
            "Image is empty."
        )


def validate_mask(
    mask: np.ndarray,
) -> None:
    """
    Validate loaded annotation mask.

    Parameters
    ----------
    mask

    Raises
    ------
    ValidationError
    """

    Validator.validate_not_none(
        mask,
        "mask",
    )

    if not isinstance(
        mask,
        np.ndarray,
    ):

        raise ValidationError(
            "Mask must be a NumPy array."
        )

    if mask.size == 0:

        raise ValidationError(
            "Mask is empty."
        )


# =============================================================================
# Shape Utilities
# =============================================================================

def same_shape(
    image: np.ndarray,
    mask: np.ndarray,
) -> bool:
    """
    Check whether image and mask have the
    same spatial dimensions.

    Parameters
    ----------
    image

    mask

    Returns
    -------
    bool
    """

    validate_image(
        image
    )

    validate_mask(
        mask
    )

    return (
        image.shape[:2] ==
        mask.shape[:2]
    )


def image_size(
    image: np.ndarray,
) -> Tuple[int, int]:
    """
    Return image size.

    Returns
    -------
    (width, height)
    """

    validate_image(
        image
    )

    height, width = image.shape[:2]

    return (
        width,
        height,
    )