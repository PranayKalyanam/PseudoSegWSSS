"""
==============================================================================
File        : image_resizer.py
Location    : methods/imageLoader/

Description :
    Image resizing utilities extracted from image_loader.py.

Responsibilities
----------------
1. Resize image by scale factor.
2. Resize image to target dimensions.
3. Preserve interpolation strategy.
4. Validate resize parameters.

This module contains only resizing logic. Higher-level workflow remains inside
ImageLoader.
==============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.exceptions import ValidationError
from utils.logger import get_logger


class ImageResizer:
    """
    Utility class for resizing images.

    This class centralizes every resize operation used during
    preprocessing.
    """

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_image(
        image: np.ndarray,
    ) -> None:
        """
        Validate input image.
        """

        if image is None:
            raise ValidationError(
                "Image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            raise ValidationError(
                "Image must be a NumPy array."
            )

        if image.size == 0:
            raise ValidationError(
                "Image is empty."
            )

    @staticmethod
    def validate_scale(
        scale: float,
    ) -> None:
        """
        Validate resize scale.
        """

        if scale <= 0:
            raise ValidationError(
                "Scale factor must be greater than zero."
            )

    @staticmethod
    def validate_size(
        width: int,
        height: int,
    ) -> None:
        """
        Validate resize dimensions.
        """

        if width <= 0:
            raise ValidationError(
                "Width must be greater than zero."
            )

        if height <= 0:
            raise ValidationError(
                "Height must be greater than zero."
            )

    # -------------------------------------------------------------------------
    # Resize by Scale
    # -------------------------------------------------------------------------

    def resize_by_scale(
        self,
        image: np.ndarray,
        scale: float,
        interpolation: Optional[int] = None,
    ) -> np.ndarray:
        """
        Resize image using a scale factor.

        Parameters
        ----------
        image
            Input image.

        scale
            Scaling factor.

        interpolation
            OpenCV interpolation method.

        Returns
        -------
        np.ndarray
        """

        self.validate_image(image)
        self.validate_scale(scale)

        if interpolation is None:

            if scale >= 1.0:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA

        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=interpolation,
        )

        self.logger.debug(
            "Image resized using scale %.4f (%d x %d → %d x %d).",
            scale,
            image.shape[1],
            image.shape[0],
            resized.shape[1],
            resized.shape[0],
        )

        return resized

    # -------------------------------------------------------------------------
    # Resize to Fixed Size
    # -------------------------------------------------------------------------

    def resize_to_size(
        self,
        image: np.ndarray,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Resize image to target dimensions.

        Parameters
        ----------
        image
            Input image.

        width
            Target width.

        height
            Target height.

        interpolation
            OpenCV interpolation method.

        Returns
        -------
        np.ndarray
        """

        self.validate_image(image)
        self.validate_size(width, height)

        resized = cv2.resize(
            image,
            (width, height),
            interpolation=interpolation,
        )

        self.logger.debug(
            "Image resized (%d x %d → %d x %d).",
            image.shape[1],
            image.shape[0],
            width,
            height,
        )

        return resized

    # -------------------------------------------------------------------------
    # Resize Using Shape
    # -------------------------------------------------------------------------

    def resize_like(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        interpolation: int = cv2.INTER_NEAREST,
    ) -> np.ndarray:
        """
        Resize one image so it matches another image.

        Parameters
        ----------
        source_image
            Image to resize.

        reference_image
            Target image.

        interpolation
            OpenCV interpolation.

        Returns
        -------
        np.ndarray
        """

        self.validate_image(source_image)
        self.validate_image(reference_image)

        height, width = reference_image.shape[:2]

        resized = cv2.resize(
            source_image,
            (width, height),
            interpolation=interpolation,
        )

        self.logger.debug(
            "Image resized to match reference (%d x %d).",
            width,
            height,
        )

        return resized

    # -------------------------------------------------------------------------
    # Resize Using Target Shape
    # -------------------------------------------------------------------------

    def resize_to_shape(
        self,
        image: np.ndarray,
        shape: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Resize image using (height, width).

        Parameters
        ----------
        image
            Input image.

        shape
            (height, width)

        interpolation
            OpenCV interpolation.

        Returns
        -------
        np.ndarray
        """

        self.validate_image(image)

        if len(shape) != 2:
            raise ValidationError(
                "Shape must be (height, width)."
            )

        height, width = shape

        return self.resize_to_size(
            image=image,
            width=width,
            height=height,
            interpolation=interpolation,
        )

    # -------------------------------------------------------------------------
    # Match Image and Mask Sizes
    # -------------------------------------------------------------------------

    def match_image_and_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure image and mask have identical dimensions.

        If dimensions differ, the mask is resized using nearest-neighbor
        interpolation.

        Parameters
        ----------
        image
            RGB image.

        mask
            Annotation mask.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
        """

        self.validate_image(image)
        self.validate_image(mask)

        image_h, image_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if (image_h == mask_h) and (image_w == mask_w):
            return image, mask

        resized_mask = cv2.resize(
            mask,
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

        self.logger.info(
            "Mask resized from (%d x %d) to (%d x %d).",
            mask_w,
            mask_h,
            image_w,
            image_h,
        )

        return image, resized_mask

    # -------------------------------------------------------------------------
    # Generic Resize
    # -------------------------------------------------------------------------

    def resize(
        self,
        image: np.ndarray,
        *,
        scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Generic resize interface.

        Examples
        --------
        resize(image, scale=0.5)

        resize(image, width=1024, height=768)
        """

        if scale is not None:

            return self.resize_by_scale(
                image=image,
                scale=scale,
                interpolation=interpolation,
            )

        if width is not None and height is not None:

            return self.resize_to_size(
                image=image,
                width=width,
                height=height,
                interpolation=interpolation,
            )

        raise ValidationError(
            "Either scale or (width, height) must be specified."
        )