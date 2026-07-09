"""
dataset_converter.py

Convert Patch objects into DatasetRecord objects.
"""

from __future__ import annotations

from data.dataset_record import DatasetRecord
from data.patch import Patch
from methods.label_utils import LabelUtils


class DatasetConverter:
    """
    Converts Patch objects into DatasetRecord objects.
    """

    @staticmethod
    def patch_to_record(
        patch: Patch,
    ) -> DatasetRecord:

        metadata = patch.metadata

        return DatasetRecord(

            patch_id=metadata.patch_id,

            patient_id=metadata.patient_id,

            region_id=metadata.tissue_region_id,

            filename=metadata.filename,

            x=metadata.coordinate.x,

            y=metadata.coordinate.y,

            width=metadata.coordinate.width,

            height=metadata.coordinate.height,

            tissue_percentage=metadata.tissue_percentage,

            tumor=metadata.label.tumor,

            stroma=metadata.label.stroma,

            lymphocyte=metadata.label.lymphocyte,

            necrosis=metadata.label.necrosis,

            label_string=LabelUtils.label_string(
                metadata.label
            ),

            image_path=str(metadata.image_path)
            if metadata.image_path
            else "",

            mask_path=str(metadata.mask_path)
            if metadata.mask_path
            else "",

            metadata=metadata.additional_metadata,
        )