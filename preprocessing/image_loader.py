"""
image_loader.py

Image loading module for the HistoGraphWSL preprocessing pipeline.

This module is responsible for

    • discovering image-mask pairs
    • loading RGB images
    • loading segmentation masks
    • validating image-mask consistency
    • resizing to the desired working magnification
    • providing a unified interface to downstream preprocessing modules

The loader is intentionally designed as the only component
that directly accesses image files. All subsequent preprocessing
modules should use ImageLoader instead of reading files directly.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from utils.logger import get_logger
from utils.validator import Validator
from utils.exceptions import (
    MaskNotFoundError,
    ValidationError,
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# ============================================================
# Data Classes
# ============================================================

@dataclass(frozen=True)
class ImagePair:
    """
    Represents one patient image and its corresponding mask.

    Attributes
    ----------
    patient_id : str

    image_path : Path

    mask_path : Path
    """

    patient_id: str

    image_path: Path

    mask_path: Path


@dataclass
class ImageMetadata:
    """
    Metadata associated with one patient image.

    This information is later propagated to
    patch extraction and graph construction.

    Attributes
    ----------
    patient_id

    width

    height

    channels

    scale_factor

    working_width

    working_height
    """

    patient_id: str

    width: int

    height: int

    channels: int

    scale_factor: float

    working_width: int

    working_height: int


# ============================================================
# Helper Functions
# ============================================================

def get_patient_id(file_path: Path) -> str:
    """
    Extract patient identifier from filename.

    Examples
    --------
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

    return file_path.stem


def discover_images(
    image_directory: Path,
) -> Dict[str, Path]:
    """
    Discover all valid images inside a directory.

    Parameters
    ----------
    image_directory

    Returns
    -------
    Dictionary

        key   → Patient ID

        value → Image path
    """

    Validator.validate_directory(image_directory)

    image_files: Dict[str, Path] = {}

    for file in sorted(image_directory.iterdir()):

        if not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        patient_id = get_patient_id(file)

        image_files[patient_id] = file

    return image_files


def pair_images_and_masks(
    image_directory: Path,
    mask_directory: Path,
) -> List[ImagePair]:
    """
    Pair every RGB image with its corresponding mask.

    Pairing is performed using Patient ID.

    Parameters
    ----------
    image_directory

    mask_directory

    Returns
    -------
    List[ImagePair]
    """

    Validator.validate_directory(image_directory)

    Validator.validate_directory(mask_directory)

    images = discover_images(image_directory)

    masks = discover_images(mask_directory)

    pairs: List[ImagePair] = []

    missing_masks: List[str] = []

    for patient_id, image_path in images.items():

        if patient_id not in masks:

            missing_masks.append(patient_id)

            continue

        pairs.append(
            ImagePair(
                patient_id=patient_id,
                image_path=image_path,
                mask_path=masks[patient_id],
            )
        )

    if len(missing_masks) > 0:

        raise MaskNotFoundError(
            f"{len(missing_masks)} masks are missing.\n"
            f"Example: {missing_masks[:5]}"
        )

    return pairs


def verify_image_alignment(
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    Verify image and mask have identical spatial dimensions.

    Parameters
    ----------
    image

    mask

    Raises
    ------
    ValidationError
    """

    if image.shape[:2] != mask.shape[:2]:

        raise ValidationError(
            "Image and mask dimensions do not match.\n"
            f"Image : {image.shape[:2]}\n"
            f"Mask  : {mask.shape[:2]}"
        )


def compute_scale_factor(
    source_magnification: float,
    target_magnification: float,
) -> float:
    """
    Compute scaling factor required to convert
    from source magnification to target magnification.

    Example
    -------
    40x

    →

    10x

    scale = 0.25
    """

    if source_magnification <= 0:

        raise ValidationError(
            "Source magnification must be positive."
        )

    if target_magnification <= 0:

        raise ValidationError(
            "Target magnification must be positive."
        )

    return target_magnification / source_magnification




# ============================================================
# Image Loader
# ============================================================

class ImageLoader:
    """
    Central image loading class.

    The ImageLoader is responsible for managing one
    patient image and its corresponding segmentation mask.

    Unlike a conventional image reader, this class stores
    both the original-resolution data and the working
    magnification data used during preprocessing.

    The original image is never modified.
    """

    def __init__(
        self,
        config,
        logger=None,
    ) -> None:
        """
        Parameters
        ----------
        config
            Configuration object exposing dataset paths and magnification.

        logger
            Project logger.
        """

        self.image_dir = Path(config.image_dir)
        self.mask_dir = Path(config.mask_dir)

        self.image_directory = self.image_dir
        self.mask_directory = self.mask_dir

        Validator.validate_directory(self.image_directory)
        Validator.validate_directory(self.mask_directory)

        self.logger = (
            logger
            if logger is not None
            else get_logger("ImageLoader")
        )

        self.source_magnification = float(
            getattr(config, "source_magnification", 40.0)
        )
        self.target_magnification = float(
            getattr(config, "target_magnification", getattr(config, "magnification", 10.0))
        )

        self.scale_factor = compute_scale_factor(
            self.source_magnification,
            self.target_magnification,
        )

        #
        # Discover all patient image-mask pairs
        #

        self.image_pairs = pair_images_and_masks(
            self.image_directory,
            self.mask_directory,
        )

        self.patient_lookup = {
            pair.patient_id: pair
            for pair in self.image_pairs
        }

        self.logger.info(
            "Discovered %d patient image-mask pairs.",
            len(self.image_pairs),
        )
        
        self._reset_cache()

    # --------------------------------------------------------
    # Properties
    # --------------------------------------------------------

    @property
    def number_of_patients(self) -> int:
        """
        Number of valid image-mask pairs.
        """

        return len(self.image_pairs)

    @property
    def patient_ids(self) -> List[str]:
        """
        Return all patient identifiers.
        """

        return sorted(self.patient_lookup.keys())

    # --------------------------------------------------------
    # Internal validation
    # --------------------------------------------------------

    def _validate_patient(
        self,
        patient_id: str,
    ) -> None:
        """
        Validate patient identifier.

        Raises
        ------
        ValidationError
        """

        if patient_id not in self.patient_lookup:

            raise ValidationError(
                f"Unknown patient ID: {patient_id}"
            )

    # --------------------------------------------------------
    # Accessors
    # --------------------------------------------------------

    def get_pair(
        self,
        patient_id: str,
    ) -> ImagePair:
        """
        Return ImagePair corresponding to one patient.
        """

        self._validate_patient(patient_id)

        return self.patient_lookup[patient_id]

    def get_image_path(
        self,
        patient_id: str,
    ) -> Path:
        """
        Return image path.
        """

        return self.get_pair(patient_id).image_path

    def get_mask_path(
        self,
        patient_id: str,
    ) -> Path:
        """
        Return mask path.
        """

        return self.get_pair(patient_id).mask_path

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """
        Return loader summary.

        Returns
        -------
        dict
        """

        return {
            "patients": self.number_of_patients,
            "image_directory": str(self.image_directory),
            "mask_directory": str(self.mask_directory),
            "source_magnification": self.source_magnification,
            "target_magnification": self.target_magnification,
            "scale_factor": self.scale_factor,
        }

    def __len__(self) -> int:
        """
        Number of patient pairs.
        """

        return self.number_of_patients

    def __repr__(self) -> str:
        """
        String representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"patients={self.number_of_patients}, "
            f"target_mag={self.target_magnification}x)"
        )
        
        
    # --------------------------------------------------------
    # Internal Cache
    # --------------------------------------------------------

    def _reset_cache(self) -> None:
        """
        Reset the internal cache.

        Only one patient is kept in memory at any time.
        """

        self._current_patient: Optional[str] = None

        self._image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

        self._working_image: Optional[np.ndarray] = None
        self._working_mask: Optional[np.ndarray] = None

        self._metadata: Optional[ImageMetadata] = None

    # --------------------------------------------------------
    # Image Reading
    # --------------------------------------------------------

    def _read_rgb(
        self,
        image_path: Path,
    ) -> np.ndarray:
        """
        Read an RGB image from disk.

        Parameters
        ----------
        image_path

        Returns
        -------
        numpy.ndarray
        """

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise ValidationError(
                f"Unable to read image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return image

    def _read_mask(
        self,
        mask_path: Path,
    ) -> np.ndarray:
        """
        Read segmentation mask.

        Original mask values are preserved.

        Parameters
        ----------
        mask_path

        Returns
        -------
        numpy.ndarray
        """

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:

            raise MaskNotFoundError(
                f"Unable to read mask:\n{mask_path}"
            )

        return mask

    # --------------------------------------------------------
    # Patient Loading
    # --------------------------------------------------------

    def load_patient(
        self,
        patient_id: str,
    ) -> None:
        """
        Load one patient into memory.

        The loaded patient becomes the active patient
        until another patient is loaded.

        Parameters
        ----------
        patient_id
        """

        #
        # Already loaded
        #

        if self._current_patient == patient_id:
            return

        self._validate_patient(patient_id)

        pair = self.get_pair(patient_id)

        self.logger.info(
            "Loading patient %s",
            patient_id,
        )

        image = self._read_rgb(pair.image_path)

        mask = self._read_mask(pair.mask_path)

        verify_image_alignment(
            image,
            mask,
        )

        height, width = image.shape[:2]

        channels = image.shape[2]

        working_width = int(
            round(width * self.scale_factor)
        )

        working_height = int(
            round(height * self.scale_factor)
        )

        metadata = ImageMetadata(
            patient_id=patient_id,
            width=width,
            height=height,
            channels=channels,
            scale_factor=self.scale_factor,
            working_width=working_width,
            working_height=working_height,
        )

        self._reset_cache()

        self._current_patient = patient_id

        self._image = image

        self._mask = mask

        self._metadata = metadata

    # --------------------------------------------------------
    # Getters
    # --------------------------------------------------------

    def get_image(self) -> np.ndarray:
        """
        Return original RGB image.

        Returns
        -------
        numpy.ndarray
        """

        if self._image is None:

            raise ValidationError(
                "No patient loaded."
            )

        return self._image

    def get_mask(self) -> np.ndarray:
        """
        Return original mask.

        Returns
        -------
        numpy.ndarray
        """

        if self._mask is None:

            raise ValidationError(
                "No patient loaded."
            )

        return self._mask

    def get_metadata(self) -> ImageMetadata:
        """
        Return metadata associated with
        the currently loaded patient.

        Returns
        -------
        ImageMetadata
        """

        if self._metadata is None:

            raise ValidationError(
                "No patient loaded."
            )

        return self._metadata

    def current_patient(self) -> str:
        """
        Return currently loaded patient ID.
        """

        if self._current_patient is None:

            raise ValidationError(
                "No patient loaded."
            )

        return self._current_patient