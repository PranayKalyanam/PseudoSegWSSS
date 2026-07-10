"""
patch_extractor.py

Extracts image patches and annotation patches from the working
image using the coordinates stored inside Patch objects.

Responsibilities
----------------
1. Crop RGB image patches.
2. Crop annotation patches.
3. Store extracted data inside Patch objects.
4. Return updated Patch objects.
"""

from typing import List

import numpy as np

from data.patch.patch import Patch


class PatchExtractor:
    """
    Extracts image and annotation patches.

    This class performs no filtering or label generation.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------

    def extract(
        self,
        patches: List[Patch],
        image: np.ndarray,
        annotation: np.ndarray,
    ) -> List[Patch]:
        """
        Extract image and annotation patches.

        Parameters
        ----------
        patches : List[Patch]

        image : np.ndarray

        annotation : np.ndarray

        Returns
        -------
        List[Patch]
        """

        extracted_patches = []

        for patch in patches:

            self._extract_patch(
                patch,
                image,
                annotation,
            )

            extracted_patches.append(patch)

        return extracted_patches

    # ---------------------------------------------------------

    def _extract_patch(
        self,
        patch: Patch,
        image: np.ndarray,
        annotation: np.ndarray,
    ) -> None:
        """
        Extract one image patch and its corresponding
        annotation patch.
        """

        coordinate = patch.coordinate

        x = coordinate.x
        y = coordinate.y
        w = coordinate.width
        h = coordinate.height

        image_patch = image[
            y:y+h,
            x:x+w
        ].copy()

        annotation_patch = annotation[
            y:y+h,
            x:x+w
        ].copy()

        patch.image_patch = image_patch
        patch.annotation_patch = annotation_patch