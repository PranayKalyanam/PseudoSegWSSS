from __future__ import annotations

import cv2
import numpy as np

from data.geometry.bounding_box import BoundingBox
from data.tissue.tissue_region import TissueRegion


class ConnectedComponents:
    """
    Extract connected tissue regions.
    """

    @staticmethod
    def extract(
        mask: np.ndarray,
    ) -> list[TissueRegion]:
        """
        Extract connected tissue regions.

        Parameters
        ----------
        mask
            Binary tissue mask.

        Returns
        -------
        list[TissueRegion]
        """

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        regions = []

        for label in range(1, num_labels):

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            region_mask = (
                labels == label
            ).astype(np.uint8) * 255

            regions.append(

                TissueRegion(

                    region_id=label,
                    bounding_box=BoundingBox(
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                        ),
                    
                    area=area,

                    mask=region_mask,
                )

            )

        return regions