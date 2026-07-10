"""
patch_collection_builder.py

Builds the final PatchDataset object from processed Patch objects.

The PatchCollectionBuilder performs no image processing.
It simply aggregates all generated Patch objects together with
dataset-level statistics into a PatchDataset.

Responsibilities
----------------
1. Collect processed patches.
2. Compute dataset-level statistics.
3. Build PatchDataset.
4. Return PatchDataset.
"""

from typing import List

from data.patient import Patient
from data.patch.patch import Patch
from data.patch.patch_dataset import PatchDataset
from data.patch.patch_dataset_statistics import PatchDatasetStatistics


class PatchCollectionBuilder:
    """
    Builds the final PatchDataset object.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------

    def build(
        self,
        patient: Patient,
        patches: List[Patch],
    ) -> PatchDataset:
        """
        Build PatchDataset.

        Parameters
        ----------
        patient : Patient

        patches : List[Patch]

        Returns
        -------
        PatchDataset
        """

        statistics = self._compute_statistics(
            patches
        )

        dataset = PatchDataset(
            patient_id=patient.image_pair.patient_id,
            patches=patches,
            statistics=statistics,
        )

        return dataset

    # ---------------------------------------------------------

    def _compute_statistics(
        self,
        patches: List[Patch],
    ) -> PatchDatasetStatistics:
        """
        Compute dataset-level statistics.
        """

        total_patches = len(patches)

        if total_patches == 0:

            return PatchDatasetStatistics(
                total_patches=0,
                average_tissue_percentage=0.0,
                average_patch_area=0.0,
                total_classes={},
                class_distribution={},
            )

        tissue_sum = 0.0
        patch_area_sum = 0

        class_distribution = {}

        for patch in patches:

            # -----------------------------
            # Tissue statistics
            # -----------------------------

            tissue_sum += patch.tissue_percentage

            # -----------------------------
            # Patch geometry
            # -----------------------------

            coordinate = patch.coordinate

            patch_area_sum += (
                coordinate.width *
                coordinate.height
            )

            # -----------------------------
            # Class statistics
            # -----------------------------

            for class_id in patch.detected_classes:

                class_distribution[class_id] = (
                    class_distribution.get(class_id, 0) + 1
                )

        average_tissue = (
            tissue_sum / total_patches
        )

        average_patch_area = (
            patch_area_sum / total_patches
        )

        statistics = PatchDatasetStatistics(

            total_patches=total_patches,

            average_tissue_percentage=average_tissue,

            average_patch_area=average_patch_area,

            total_classes=class_distribution,

            class_distribution=class_distribution,
        )

        return statistics