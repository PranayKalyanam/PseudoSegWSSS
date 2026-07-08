"""
magnification.py

Magnification management for the HistoGraphWSL preprocessing pipeline.

This module converts an image from its original magnification
to the desired working magnification while preserving
the coordinate mapping between both resolutions.

Unlike a simple resize utility, this module also provides
coordinate transformation utilities that are required by
patch extraction, visualization, graph construction,
and debugging.

Author
------
Your Name
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from utils.exceptions import ValidationError

@dataclass(frozen=True)
class MagnificationInfo:
    """
    Information describing the magnification transformation.
    """

    source_magnification: float

    target_magnification: float

    scale_factor: float

    original_width: int

    original_height: int

    working_width: int

    working_height: int
    
    
class MagnificationTransformer:
    """
    Handles magnification conversion and coordinate mapping.
    """

    def __init__(
        self,
        source_magnification: float,
        target_magnification: float,
    ) -> None:

        if source_magnification <= 0:
            raise ValidationError(
                "Source magnification must be positive."
            )

        if target_magnification <= 0:
            raise ValidationError(
                "Target magnification must be positive."
            )

        self.source_mag = float(source_magnification)

        self.target_mag = float(target_magnification)

        self.scale = (
            self.target_mag /
            self.source_mag
        )
        

    def resize_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, MagnificationInfo]:
        """
        Resize RGB image.
        """

        h, w = image.shape[:2]

        new_w = int(round(w * self.scale))

        new_h = int(round(h * self.scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        info = MagnificationInfo(
            source_magnification=self.source_mag,
            target_magnification=self.target_mag,
            scale_factor=self.scale,
            original_width=w,
            original_height=h,
            working_width=new_w,
            working_height=new_h,
        )

        return resized, info
    
    
    def resize_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Resize segmentation mask.

        Nearest-neighbor interpolation is used to
        preserve class labels.
        """

        h, w = mask.shape[:2]

        new_w = int(round(w * self.scale))

        new_h = int(round(h * self.scale))

        return cv2.resize(
            mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )
        

    def to_working(
        self,
        x: int,
        y: int,
    ) -> Tuple[int, int]:
        """
        Convert original coordinates to
        working-image coordinates.
        """

        return (
            int(round(x * self.scale)),
            int(round(y * self.scale)),
        )


    def to_original(
        self,
        x: int,
        y: int,
    ) -> Tuple[int, int]:
        """
        Convert working coordinates back
        to the original image.
        """

        return (
            int(round(x / self.scale)),
            int(round(y / self.scale)),
        )
        
        
    def map_bbox_to_working(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:

        x, y = self.to_working(x, y)

        width = int(round(width * self.scale))

        height = int(round(height * self.scale))

        return x, y, width, height


    def map_bbox_to_original(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:

        x, y = self.to_original(x, y)

        width = int(round(width / self.scale))

        height = int(round(height / self.scale))

        return x, y, width, height