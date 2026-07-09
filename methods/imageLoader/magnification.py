"""
==============================================================================
File: magnification.py

Description
-----------
Magnification utilities for the HistoGraphWSL preprocessing pipeline.

Responsibilities
----------------
1. Read image resolution (MPP).
2. Estimate slide magnification.
3. Compute scale factors.
4. Resize images to requested magnification.

This module contains only magnification-related functionality.
==============================================================================
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class Magnification:
    """
    Utility class for magnification handling.
    """

    #
    # Common MPP values
    #

    MPP_40X = 0.25
    MPP_20X = 0.50
    MPP_10X = 1.00
    MPP_5X = 2.00
    MPP_2X = 5.00

    MAGNIFICATION_TO_MPP = {
        40: MPP_40X,
        20: MPP_20X,
        10: MPP_10X,
        5: MPP_5X,
        2: MPP_2X,
    }

    # ------------------------------------------------------------------

    @staticmethod
    def estimate_magnification(
        mpp: Optional[float],
    ) -> int:
        """
        Estimate objective magnification from MPP.

        Parameters
        ----------
        mpp : float

        Returns
        -------
        int
        """

        if mpp is None:
            return 40

        if mpp <= 0.30:
            return 40

        if mpp <= 0.60:
            return 20

        if mpp <= 1.20:
            return 10

        if mpp <= 2.50:
            return 5

        return 2

    # ------------------------------------------------------------------

    @staticmethod
    def magnification_to_mpp(
        magnification: int,
    ) -> float:
        """
        Convert objective magnification to MPP.

        Parameters
        ----------
        magnification : int

        Returns
        -------
        float
        """

        if magnification not in Magnification.MAGNIFICATION_TO_MPP:

            raise ValueError(
                f"Unsupported magnification: {magnification}X"
            )

        return Magnification.MAGNIFICATION_TO_MPP[
            magnification
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def compute_scale_factor(
        source_magnification: int,
        target_magnification: int,
    ) -> float:
        """
        Compute image scaling factor.

        Parameters
        ----------
        source_magnification : int

        target_magnification : int

        Returns
        -------
        float
        """

        return target_magnification / source_magnification

    # ------------------------------------------------------------------

    @staticmethod
    def resize_to_magnification(
        image: np.ndarray,
        source_magnification: int,
        target_magnification: int,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Resize image from one magnification to another.

        Parameters
        ----------
        image : np.ndarray

        source_magnification : int

        target_magnification : int

        interpolation : int

        Returns
        -------
        np.ndarray
        """

        if source_magnification == target_magnification:
            return image

        scale = Magnification.compute_scale_factor(
            source_magnification,
            target_magnification,
        )

        new_width = int(round(image.shape[1] * scale))
        new_height = int(round(image.shape[0] * scale))

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def resize_image_and_mask(
        image: np.ndarray,
        mask: np.ndarray,
        source_magnification: int,
        target_magnification: int,
    ):
        """
        Resize image and mask together.

        Parameters
        ----------
        image : np.ndarray

        mask : np.ndarray

        source_magnification : int

        target_magnification : int

        Returns
        -------
        tuple
            (resized_image, resized_mask)
        """

        if source_magnification == target_magnification:
            return image, mask

        scale = Magnification.compute_scale_factor(
            source_magnification,
            target_magnification,
        )

        new_width = int(round(image.shape[1] * scale))
        new_height = int(round(image.shape[0] * scale))

        resized_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        resized_mask = cv2.resize(
            mask,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized_image, resized_mask

    # ------------------------------------------------------------------

    @staticmethod
    def requires_rescaling(
        source_magnification: int,
        target_magnification: int,
    ) -> bool:
        """
        Determine whether rescaling is required.

        Parameters
        ----------
        source_magnification : int

        target_magnification : int

        Returns
        -------
        bool
        """

        return source_magnification != target_magnification

    # ------------------------------------------------------------------

    @staticmethod
    def get_scale_percentage(
        source_magnification: int,
        target_magnification: int,
    ) -> float:
        """
        Return scaling percentage.

        Parameters
        ----------
        source_magnification : int

        target_magnification : int

        Returns
        -------
        float
        """

        scale = Magnification.compute_scale_factor(
            source_magnification,
            target_magnification,
        )

        return scale * 100.0