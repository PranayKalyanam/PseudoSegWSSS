"""
image_discovery.py

Image discovery utilities for the HistoGraphWSL preprocessing pipeline.

This module is responsible for

    • discovering images
    • discovering masks
    • pairing images with masks
    • extracting patient identifiers

This module performs only dataset discovery.
It does not load any image into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List

from data.image.image_pair import ImagePair

from utils.validator import Validator
from utils.exceptions import MaskNotFoundError


# =============================================================================
# Supported Image Extensions
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# =============================================================================
# Patient ID
# =============================================================================

def get_patient_id(
    file_path: Path,
) -> str:
    """
    Extract patient identifier from filename.

    Example
    -------
    TCGA-XX-1111.png

    →

    TCGA-XX-1111

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    str
    """

    Validator.validate_not_none(
        file_path,
        "file_path",
    )

    return file_path.stem


# =============================================================================
# Image Discovery
# =============================================================================

def discover_images(
    image_directory: Path,
) -> Dict[str, Path]:
    """
    Discover all valid images inside a directory.

    Parameters
    ----------
    image_directory : Path

    Returns
    -------
    Dict[str, Path]

        key
            Patient ID

        value
            Image path
    """

    Validator.validate_directory(
        image_directory
    )

    image_files: Dict[str, Path] = {}

    for file in sorted(image_directory.iterdir()):

        #
        # Skip directories
        #

        if not file.is_file():
            continue

        #
        # Skip unsupported formats
        #

        if file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        patient_id = get_patient_id(
            file
        )

        image_files[patient_id] = file

    return image_files


# =============================================================================
# Image–Mask Pairing
# =============================================================================

def pair_images_and_masks(
    image_directory: Path,
    mask_directory: Path,
) -> List[ImagePair]:
    """
    Pair every RGB image with its corresponding
    segmentation mask.

    Pairing is performed using patient identifiers.

    Parameters
    ----------
    image_directory : Path

    mask_directory : Path

    Returns
    -------
    List[ImagePair]
    """

    Validator.validate_directory(
        image_directory
    )

    Validator.validate_directory(
        mask_directory
    )

    images = discover_images(
        image_directory
    )

    masks = discover_images(
        mask_directory
    )

    pairs: List[ImagePair] = []

    missing_masks: List[str] = []

    for patient_id, image_path in images.items():

        if patient_id not in masks:

            missing_masks.append(
                patient_id
            )

            continue

        pairs.append(

            ImagePair(

                patient_id=patient_id,

                image_path=image_path,

                mask_path=masks[patient_id],

            )

        )

    #
    # Report missing masks
    #

    if len(missing_masks) > 0:

        raise MaskNotFoundError(

            f"{len(missing_masks)} mask(s) are missing.\n"
            f"Example: {missing_masks[:5]}"

        )

    return pairs


# =============================================================================
# Dataset Summary
# =============================================================================

def number_of_patients(
    image_pairs: List[ImagePair],
) -> int:
    """
    Return number of paired patients.

    Parameters
    ----------
    image_pairs

    Returns
    -------
    int
    """

    return len(image_pairs)


def patient_ids(
    image_pairs: List[ImagePair],
) -> List[str]:
    """
    Return sorted patient identifiers.

    Parameters
    ----------
    image_pairs

    Returns
    -------
    List[str]
    """

    return sorted(

        pair.patient_id

        for pair in image_pairs

    )


# =============================================================================
# Patient Lookup
# =============================================================================

def build_patient_lookup(
    image_pairs: List[ImagePair],
) -> Dict[str, ImagePair]:
    """
    Build a dictionary for constant-time patient lookup.

    Parameters
    ----------
    image_pairs

    Returns
    -------
    Dict[str, ImagePair]
    """

    lookup: Dict[str, ImagePair] = {}

    for pair in image_pairs:

        lookup[pair.patient_id] = pair

    return lookup