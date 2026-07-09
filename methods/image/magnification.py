"""
magnification.py

Utilities for handling image magnification and resolution.

This module is responsible for

    • Detecting source magnification
    • Computing image scaling factors
    • Computing resized image dimensions

The implementation is intentionally independent of any
specific dataset or image loader.
"""

from __future__ import annotations

import re
from pathlib import Path

from data.image.image_size import ImageSize


class Magnification:
    """
    Utilities for image magnification handling.
    """

    DEFAULT_MAGNIFICATION = 40.0

    # ---------------------------------------------------------
    # Detection
    # ---------------------------------------------------------

    @staticmethod
    def detect(image_path: str | Path) -> float:
        """
        Detect the source magnification.

        Detection priority

        1. Filename
        2. Future OpenSlide metadata
        3. Default value

        Supported filename examples

            slide_40x.png
            slide_20X.png
            patient_10x.jpg

        Parameters
        ----------
        image_path
            Image filename.

        Returns
        -------
        float
            Source magnification.
        """

        image_path = Path(image_path)

        filename = image_path.stem

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*[xX]",
            filename,
        )

        if match:

            return float(match.group(1))

        return Magnification.DEFAULT_MAGNIFICATION

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def validate(
        source: float,
        target: float,
    ) -> None:

        if source <= 0:

            raise ValueError(
                "Source magnification must be positive."
            )

        if target <= 0:

            raise ValueError(
                "Target magnification must be positive."
            )

    # ---------------------------------------------------------
    # Scale
    # ---------------------------------------------------------

    @staticmethod
    def compute_scale_factor(
        source: float,
        target: float,
    ) -> float:
        """
        Compute resize scale factor.

        Examples

            40x → 20x = 0.5
            20x → 10x = 0.5
            20x → 40x = 2.0
        """

        Magnification.validate(
            source,
            target,
        )

        return target / source

    # ---------------------------------------------------------
    # Image Size
    # ---------------------------------------------------------

    @staticmethod
    def compute_target_size(
        original_size: ImageSize,
        scale_factor: float,
    ) -> ImageSize:
        """
        Compute resized image dimensions.
        """

        if scale_factor <= 0:

            raise ValueError(
                "Scale factor must be positive."
            )

        return ImageSize(
            width=max(
                1,
                int(round(
                    original_size.width * scale_factor
                )),
            ),
            height=max(
                1,
                int(round(
                    original_size.height * scale_factor
                )),
            ),
        )

    # ---------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------

    @staticmethod
    def requires_resizing(
        source: float,
        target: float,
    ) -> bool:
        """
        Determine whether resizing is required.
        """

        return source != target

    @staticmethod
    def to_string(
        magnification: float,
    ) -> str:
        """
        Convert magnification into human-readable form.
        """

        return f"{magnification:g}x"