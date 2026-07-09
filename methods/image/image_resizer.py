"""
image_resizer.py

Utilities for resizing histopathological images and
annotation masks.

This module contains reusable image resizing methods
used throughout the preprocessing pipeline.

The class is completely stateless and performs no
dataset-specific processing.
"""

from __future__ import annotations

import cv2
import numpy as np

from data.image.image_size import ImageSize


class ImageResizer:
    """
    Image resizing utilities.
    """

    @staticmethod
    def resize_image(
        image: np.ndarray,
        size: ImageSize,
    ) -> np.ndarray:
        """
        Resize an RGB image.

        Parameters
        ----------
        image
            RGB image.

        size
            Target image size.

        Returns
        -------
        np.ndarray
            Resized RGB image.
        """

        return cv2.resize(
            image,
            (size.width, size.height),
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def resize_mask(
        mask: np.ndarray,
        size: ImageSize,
    ) -> np.ndarray:
        """
        Resize an annotation mask.

        Nearest-neighbor interpolation preserves
        integer class labels.

        Parameters
        ----------
        mask
            Annotation mask.

        size
            Target image size.

        Returns
        -------
        np.ndarray
            Resized annotation mask.
        """

        return cv2.resize(
            mask,
            (size.width, size.height),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def resize_pair(
        image: np.ndarray,
        mask: np.ndarray,
        size: ImageSize,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resize an image-mask pair.

        Parameters
        ----------
        image
            RGB image.

        mask
            Annotation mask.

        size
            Target image size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Resized image and mask.
        """

        resized_image = ImageResizer.resize_image(
            image,
            size,
        )

        resized_mask = ImageResizer.resize_mask(
            mask,
            size,
        )

        return resized_image, resized_mask

    @staticmethod
    def resize_by_scale(
        image: np.ndarray,
        mask: np.ndarray,
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, ImageSize]:
        """
        Resize an image-mask pair using a scale factor.

        Parameters
        ----------
        image
            RGB image.

        mask
            Annotation mask.

        scale_factor
            Image scaling factor.

        Returns
        -------
        tuple
            (resized_image,
             resized_mask,
             working_size)
        """

        if scale_factor <= 0:
            raise ValueError(
                "Scale factor must be greater than zero."
            )

        height, width = image.shape[:2]

        working_size = ImageSize(
            width=max(1, int(round(width * scale_factor))),
            height=max(1, int(round(height * scale_factor))),
        )

        resized_image, resized_mask = ImageResizer.resize_pair(
            image,
            mask,
            working_size,
        )

        return (
            resized_image,
            resized_mask,
            working_size,
        )

    @staticmethod
    def resize_like(
        image: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Resize an image to match the dimensions of a
        reference image.

        Parameters
        ----------
        image
            Input image.

        reference
            Reference image.

        Returns
        -------
        np.ndarray
            Resized image.
        """

        height, width = reference.shape[:2]

        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def resize_mask_like(
        mask: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Resize a mask to match a reference image.

        Parameters
        ----------
        mask
            Annotation mask.

        reference
            Reference image.

        Returns
        -------
        np.ndarray
            Resized annotation mask.
        """

        height, width = reference.shape[:2]

        return cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def requires_resizing(
        current_size: ImageSize,
        target_size: ImageSize,
    ) -> bool:
        """
        Determine whether resizing is necessary.

        Returns
        -------
        bool
        """

        return current_size != target_size