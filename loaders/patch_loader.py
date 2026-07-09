from __future__ import annotations

from pathlib import Path

from data.patch.patch_dataset import PatchDataset
from data.patch.patch_metadata import PatchMetadata

from methods.patch.coordinate_transform import CoordinateTransform
from methods.patch.patch_extractor import PatchExtractor
from methods.patch.patch_filter import PatchFilter
from methods.patch.patch_label_generator import PatchLabelGenerator
from methods.patch.patch_naming import PatchNaming
from methods.patch.patch_statistics import PatchStatistics
from methods.patch.sliding_window import SlidingWindow

from utils.logger import get_logger


class PatchLoader:
    """
    Extracts valid image patches from tissue regions.

    Responsibilities
    ----------------
    1. Generate candidate windows.
    2. Extract image and annotation patches.
    3. Filter invalid patches.
    4. Generate weak labels.
    5. Compute statistics.
    6. Create metadata.
    7. Populate Patient.patch_dataset.
    """

    def __init__(self, config):

        self.config = config

        self.logger = get_logger(
            self.__class__.__name__
        )

        self.window_generator = SlidingWindow(
            patch_size=config.patch_size,
            overlap=config.overlap,
        )

        self.extractor = PatchExtractor()

        self.filter = PatchFilter(
            minimum_tissue_percentage=config.tissue_threshold,
        )

    def load(self, patient):

        self.logger.info(
            "Extracting patches for %s",
            patient.patient_id,
        )

        dataset = PatchDataset()

        working_image = patient.working_image
        working_mask = patient.working_mask
        tissue_mask = patient.tissue_binary_mask

        metadata_index = 0

        for coordinate in self.window_generator.generate(
            tissue_regions=patient.tissue_regions,
            image_width=working_image.shape[1],
            image_height=working_image.shape[0],
        ):

            image_patch, mask_patch = self.extractor.extract(
                image=working_image,
                mask=working_mask,
                coordinate=coordinate,
            )

            tissue_crop = tissue_mask[
                coordinate.y:coordinate.y + coordinate.height,
                coordinate.x:coordinate.x + coordinate.width,
            ]

            if not self.filter.keep(
                image_patch, mask_patch,
                tissue_crop,
            ):
                continue

            label = PatchLabelGenerator.generate(
                patch.mask
            )

            patch.label = label

            patch.statistics = PatchStatistics.compute(
                patch,
                tissue_crop,
            )

            original_coordinate = (
                CoordinateTransform.working_to_original(
                    coordinate,
                    patient.metadata.scale_factor,
                )
            )

            filename = PatchNaming.generate(
                original_filename=patient.image_filename,
                global_x=original_coordinate.x,
                global_y=original_coordinate.y,
                label=label,
            )

            metadata = PatchMetadata(
                patch_id=metadata_index,
                patient_id=patient.patient_id,
                patch_name=Path(filename).stem,
                image_path=None,
                mask_path=None,
                coordinate=original_coordinate,
                tissue_region_id=-1,
                tissue_percentage=patch.statistics.tissue_percentage,
                magnification=patient.metadata.target_magnification,
                label=label,
                label_string=PatchLabelGenerator.binary_string(
                    label
                ),
                filename=filename,
                generated=True,
                generator=self.__class__.__name__,
                detected_classes=patch.statistics.detected_classes,
            )

            patch.metadata = metadata

            dataset.patches.append(
                patch
            )

            metadata_index += 1

        patient.patch_dataset = dataset

        self.logger.info(
            "Extracted %d valid patches.",
            len(dataset.patches),
        )

        return patient

    def load_all(
        self,
        patients,
    ):

        for patient in patients:

            self.load(patient)

        return patients