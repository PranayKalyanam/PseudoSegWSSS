"""
==============================================================================
File        : image_alignment.py

Description :
    Image and mask alignment utilities.

    This module contains the alignment-related functionality that was
    previously embedded inside image_loader.py. It verifies that the loaded
    image and annotation mask are spatially compatible before further
    preprocessing.

Responsibilities
----------------
1. Validate image-mask dimensions.
2. Validate channel compatibility.
3. Resize mask when explicitly requested.
4. Return aligned image/mask pair.

Author : Your Name
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from data.image.image_data import ImageData
from utils.exceptions import ValidationError
from utils.logger import get_logger


@dataclass(slots=True)
class ImageAlignmentResult:
    """
    Result returned after alignment.
    """

    image: np.ndarray

    mask: np.ndarray

    aligned: bool

    resized_mask: bool = False


class ImageAlignment:
    """
    Image-mask alignment utility.

    This class performs only spatial validation and optional mask resizing.
    It does not perform image registration.
    """

    def __init__(self) -> None:

        self.logger = get_logger(self.__class__.__name__)

    # -------------------------------------------------------------------------
    # Shape Validation
    # -------------------------------------------------------------------------

    def validate_dimensions(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> bool:
        """
        Verify that image and mask have identical width and height.
        """

        if image is None:
            raise ValidationError(
                "Image is None."
            )

        if mask is None:
            raise ValidationError(
                "Mask is None."
            )

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]

        if (
            image_height != mask_height
            or
            image_width != mask_width
        ):

            raise ValidationError(

                "Image and mask dimensions do not match.\n"
                f"Image : ({image_width}, {image_height})\n"
                f"Mask  : ({mask_width}, {mask_height})"

            )

        return True

    # -------------------------------------------------------------------------
    # Channel Validation
    # -------------------------------------------------------------------------

    def validate_channels(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> bool:
        """
        Validate image and mask channel configuration.
        """

        #
        # Image
        #

        if image.ndim not in (2, 3):

            raise ValidationError(
                "Unsupported image dimensions."
            )

        #
        # Mask
        #

        if mask.ndim not in (2, 3):

            raise ValidationError(
                "Unsupported mask dimensions."
            )

        return True

    # -------------------------------------------------------------------------
    # Resize Mask
    # -------------------------------------------------------------------------

    def resize_mask(
        self,
        mask: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize mask using nearest-neighbor interpolation.
        """

        target_height, target_width = target_shape

        resized_mask = cv2.resize(

            mask,

            (
                target_width,
                target_height,
            ),

            interpolation=cv2.INTER_NEAREST,

        )

        return resized_mask

    # -------------------------------------------------------------------------
    # Align Arrays
    # -------------------------------------------------------------------------

    def align_arrays(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        allow_resize: bool = False,
    ) -> ImageAlignmentResult:
        """
        Align image and mask arrays.
        """

        self.validate_channels(
            image,
            mask,
        )

        image_shape = image.shape[:2]
        mask_shape = mask.shape[:2]

        if image_shape == mask_shape:

            return ImageAlignmentResult(

                image=image,

                mask=mask,

                aligned=True,

                resized_mask=False,

            )

        if not allow_resize:

            raise ValidationError(

                "Image and mask have different sizes.\n"
                f"Image : {image_shape}\n"
                f"Mask  : {mask_shape}"

            )

        resized_mask = self.resize_mask(

            mask,

            image_shape,

        )

        self.validate_dimensions(

            image,

            resized_mask,

        )

        self.logger.warning(

            "Mask resized from %s to %s.",

            mask_shape,

            image_shape,

        )

        return ImageAlignmentResult(

            image=image,

            mask=resized_mask,

            aligned=True,

            resized_mask=True,

        )

    # -------------------------------------------------------------------------
    # Align ImageData
    # -------------------------------------------------------------------------

    def align(
        self,
        image_data: ImageData,
        allow_resize: bool = False,
    ) -> ImageData:
        """
        Align an ImageData object.
        """

        result = self.align_arrays(

            image=image_data.image,

            mask=image_data.mask,

            allow_resize=allow_resize,

        )

        image_data.image = result.image
        image_data.mask = result.mask

        return image_data