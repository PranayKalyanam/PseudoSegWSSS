"""
image_reader.py

Image reading utilities.

Responsible only for loading images from disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageReader:
    """
    Image loading utilities.
    """

    @staticmethod
    def read_rgb(path: Path) -> np.ndarray:
        """
        Reads an RGB image.

        Parameters
        ----------
        path
            Image file.

        Returns
        -------
        np.ndarray
            RGB image.
        """

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {path}"
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    @staticmethod
    def read_mask(path: Path) -> np.ndarray:
        """
        Reads an annotation mask.

        Parameters
        ----------
        path
            Mask file.

        Returns
        -------
        np.ndarray
            Grayscale mask.
        """

        mask = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            raise FileNotFoundError(
                f"Unable to read mask: {path}"
            )

        if mask.ndim == 3:

            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_BGR2GRAY,
            )

        return mask

    @staticmethod
    def image_size(image: np.ndarray):
        """
        Returns image dimensions.
        """

        from data.image.image_size import ImageSize

        height, width = image.shape[:2]

        return ImageSize(
            width=width,
            height=height,
        )

    @staticmethod
    def number_of_channels(image: np.ndarray) -> int:
        """
        Returns the number of image channels.
        """

        if image.ndim == 2:
            return 1

        return image.shape[2]

    @staticmethod
    def dtype(image: np.ndarray) -> str:
        """
        Returns the NumPy datatype.
        """

        return str(image.dtype)