"""
patch_generator.py

Generates candidate Patch objects from patch coordinates.

The PatchGenerator does not extract image pixels.
Instead, it converts spatial coordinates into Patch objects
that are passed to later preprocessing stages.

Responsibilities
----------------
1. Receive valid patch coordinates.
2. Create one Patch object per coordinate.
3. Initialize basic patch attributes.
4. Return a collection of candidate patches.
"""

from typing import List

from data.patch.patch import Patch
from data.patch.patch_coordinate import PatchCoordinate


class PatchGenerator:
    """
    Generates candidate Patch objects.

    No image processing occurs here.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------

    def generate(
        self,
        coordinates: List[PatchCoordinate],
    ) -> List[Patch]:
        """
        Create Patch objects from coordinates.

        Parameters
        ----------
        coordinates : List[PatchCoordinate]

        Returns
        -------
        List[Patch]
        """

        patches = []

        for index, coordinate in enumerate(coordinates):

            patch = self._create_patch(
                patch_id=index,
                coordinate=coordinate,
            )

            patches.append(patch)

        return patches

    # ---------------------------------------------------------

    def _create_patch(
        self,
        patch_id: int,
        coordinate: PatchCoordinate,
    ) -> Patch:
        """
        Construct a Patch object.

        Only spatial information is initialized.
        Image pixels, labels, metadata, and statistics
        are added by later preprocessing modules.
        """

        patch = Patch(
            patch_id=patch_id,
            coordinate=coordinate,
        )

        return patch