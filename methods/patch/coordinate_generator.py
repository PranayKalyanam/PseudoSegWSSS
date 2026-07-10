"""
coordinate_generator.py

Generates candidate patch coordinates for each tissue region.

The CoordinateGenerator computes the spatial locations where
patches may be extracted. It does not perform any image extraction
or tissue filtering.

Responsibilities
----------------
1. Iterate through tissue regions.
2. Generate sliding-window coordinates.
3. Respect patch size and overlap.
4. Keep coordinates within image boundaries.
5. Return a collection of PatchCoordinate objects.
"""

from typing import List
from configs.config import get_config

from data.tissue.tissue_region import TissueRegion
from data.patch.patch_coordinate import PatchCoordinate


class CoordinateGenerator:
    """
    Generates candidate patch coordinates.

    The generated coordinates represent potential patch locations.
    They are later validated and converted into Patch objects.
    """

    def __init__(
        self,
        config,
        patch_size: int,
        overlap: float,
    ):
        self.config = get_config()

        # self.patch_size = patch_size
        # self.overlap = overlap 

        self.patch_size = config.patch_size
        self.overlap = config.overlap

        self.stride = int(
            self.patch_size * (1.0 - self.overlap)
        )

    # ---------------------------------------------------------

    def generate(
        self,
        image,
        tissue_regions: List[TissueRegion],
    ) -> List[PatchCoordinate]:
        """
        Generate candidate coordinates for every tissue region.

        Parameters
        ----------
        image
            Working image.

        tissue_regions : List[TissueRegion]

        Returns
        -------
        List[PatchCoordinate]
        """

        image_height, image_width = image.shape[:2]

        coordinates = []

        for region in tissue_regions:

            region_coordinates = self._generate_region_coordinates(
                region,
                image_width,
                image_height,
            )

            coordinates.extend(region_coordinates)

        return coordinates

    # ---------------------------------------------------------

    def _generate_region_coordinates(
        self,
        region: TissueRegion,
        image_width: int,
        image_height: int,
    ) -> List[PatchCoordinate]:
        """
        Generate coordinates inside one tissue region.
        """

        coordinates = []

        xmin = region.bounding_box.x
        ymin = region.bounding_box.y

        xmax = region.bounding_box.x + region.bounding_box.width
        ymax = region.bounding_box.y + region.bounding_box.height

        y = ymin

        while y < ymax:

            x = xmin

            while x < xmax:

                coordinate = self._create_coordinate(
                    x,
                    y,
                    image_width,
                    image_height,
                )

                if coordinate is not None:
                    coordinates.append(coordinate)

                x += self.stride

            y += self.stride

        return coordinates

    # ---------------------------------------------------------

    def _create_coordinate(
        self,
        x: int,
        y: int,
        image_width: int,
        image_height: int,
    ):
        """
        Create one coordinate if it lies inside image boundaries.
        """

        if x + self.patch_size > image_width:
            return None

        if y + self.patch_size > image_height:
            return None

        return PatchCoordinate(
            x=x,
            y=y,
            width=self.patch_size,
            height=self.patch_size,
        )