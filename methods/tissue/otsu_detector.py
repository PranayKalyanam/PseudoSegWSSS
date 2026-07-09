from __future__ import annotations

import cv2
import numpy as np


class OtsuDetector:
    """
    Performs tissue detection using Otsu thresholding.
    """

    @staticmethod
    def detect(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a binary tissue mask.

        Parameters
        ----------
        image
            RGB image.

        Returns
        -------
        np.ndarray
            Binary tissue mask.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        return mask