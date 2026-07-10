"""
weak_label_generator.py

Generates weak multi-label annotations from annotation patches.

Each annotation patch is converted into a weak binary label
representing the presence or absence of semantic tissue classes.

Responsibilities
----------------
1. Identify semantic classes present.
2. Ignore background pixels.
3. Generate binary weak labels.
4. Compute class statistics.
5. Store results inside Patch objects.
"""

from typing import List

import numpy as np

from data.patch.patch import Patch


class WeakLabelGenerator:
    """
    Generates weak multi-label annotations.
    """

    def __init__(
        self,
        class_ids=None,
        background_label: int = 0,
    ):
        """
        Parameters
        ----------
        class_ids
            Ordered list of semantic class IDs.

            Example:
                [1,2,3,4]

        background_label
            Background pixel value.
        """

        if class_ids is None:
            class_ids = [1, 2, 3, 4]

        self.class_ids = class_ids
        self.background_label = background_label

    # ---------------------------------------------------------

    def generate(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Generate weak labels for every patch.
        """

        labeled_patches = []

        for patch in patches:

            self._generate_patch_label(patch)

            labeled_patches.append(patch)

        return labeled_patches

    # ---------------------------------------------------------

    def _generate_patch_label(
        self,
        patch: Patch,
    ) -> None:
        """
        Generate weak labels for one patch.
        """

        annotation = patch.annotation_patch

        unique_classes = np.unique(annotation)

        # Remove background
        unique_classes = unique_classes[
            unique_classes != self.background_label
        ]

        weak_label = []
        pixel_counts = {}
        class_percentages = {}

        total_foreground_pixels = np.count_nonzero(
            annotation != self.background_label
        )

        if total_foreground_pixels == 0:
            total_foreground_pixels = 1

        for class_id in self.class_ids:

            pixel_count = np.sum(annotation == class_id)

            pixel_counts[class_id] = int(pixel_count)

            class_percentages[class_id] = (
                pixel_count / total_foreground_pixels
            )

            weak_label.append(
                1 if class_id in unique_classes else 0
            )

        patch.detected_classes = list(unique_classes)

        patch.weak_label = np.asarray(
            weak_label,
            dtype=np.uint8,
        )

        patch.class_pixel_counts = pixel_counts

        patch.class_percentages = class_percentages