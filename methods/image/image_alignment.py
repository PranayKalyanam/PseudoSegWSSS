"""
image_alignment.py

Utilities for validating image-mask alignment.

This module verifies that an image and its corresponding
annotation mask are spatially compatible before further
processing.

The class performs validation only and never modifies
the input arrays.
"""

from __future__ import annotations

import numpy as np


class ImageAlignment:
    """
    Image-mask alignment utilities.
    """

    @staticmethod
    def validate_dimensions(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """
        Validate that image and mask have identical
        spatial dimensions.
        """

        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]

        if (
            image_height != mask_height
            or
            image_width != mask_width
        ):
            raise ValueError(
                "Image and mask dimensions do not match.\n"
                f"Image : {image_width} × {image_height}\n"
                f"Mask  : {mask_width} × {mask_height}"
            )

    @staticmethod
    def validate_channels(
        image: np.ndarray,
    ) -> None:
        """
        Validate RGB image.
        """

        if image.ndim != 3:

            raise ValueError(
                "Input image must be a 3-channel RGB image."
            )

        if image.shape[2] != 3:

            raise ValueError(
                "Input image must contain exactly 3 channels."
            )

    @staticmethod
    def validate_mask(
        mask: np.ndarray,
    ) -> None:
        """
        Validate annotation mask.
        """

        if mask.ndim != 2:

            raise ValueError(
                "Annotation mask must be single-channel."
            )

    @staticmethod
    def validate(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """
        Perform complete alignment validation.

        This method validates

        * RGB image
        * single-channel mask
        * identical spatial dimensions
        """

        ImageAlignment.validate_channels(image)

        ImageAlignment.validate_mask(mask)

        ImageAlignment.validate_dimensions(
            image,
            mask,
        )

    @staticmethod
    def is_aligned(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> bool:
        """
        Returns
        -------
        bool
            True if image and mask are aligned.
        """

        try:

            ImageAlignment.validate(
                image,
                mask,
            )

            return True

        except ValueError:

            return False