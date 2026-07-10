from __future__ import annotations

from configs.config import get_config

from loaders.patient_loader import PatientLoader
from loaders.image_loader import ImageLoader

from methods.patch.coordinate_generator import CoordinateGenerator
from methods.patch.patch_extractor import PatchExtractor
from methods.patch.tissue_patch_filter import TissuePatchFilter
from methods.patch.weak_label_generator import WeakLabelGenerator
from methods.patch.patch_metadata_generator import PatchMetadataGenerator
from methods.patch.patch_collection_builder import PatchCollectionBuilder

from methods.visualization.patch_visualizer import PatchVisualizer

from utils.logger import get_logger


def main():

    logger = get_logger("PatchPipelineV2Test")

    logger.info("=" * 80)
    logger.info("PATCH PIPELINE V2 TEST")
    logger.info("=" * 80)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = get_config()

    # --------------------------------------------------
    # Patient Discovery
    # --------------------------------------------------

    patient_loader = PatientLoader(
        config=config,
        logger=logger,
    )

    patients = patient_loader.load_all()

    logger.info("Patients discovered : %d", len(patients))

    # --------------------------------------------------
    # Image Loading
    # --------------------------------------------------

    image_loader = ImageLoader(
        config=config,
        logger=logger,
    )

    patients = image_loader.load_all(
        patients,
    )

    # ==================================================
    # Process Every Patient
    # ==================================================

    for patient in patients:

        logger.info("")
        logger.info("=" * 80)
        logger.info("PATIENT : %s", patient.patient_id)
        logger.info("=" * 80)

        logger.info(
            "Original Image Shape : %s",
            patient.image.shape,
        )

        logger.info(
            "Original Mask Shape  : %s",
            patient.mask.shape,
        )

        ##################################################
        # Coordinate Generator
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Coordinate Generator")
        logger.info("-" * 80)

        coordinate_generator = CoordinateGenerator(
            config=config,
            patch_size=config.patch_size,
            overlap=config.overlap,
        )

        coordinates = coordinate_generator.generate(
            image=patient.image,
            tissue_regions=[
                type(
                    "DummyRegion",
                    (),
                    {
                        "bounding_box": type(
                            "BBox",
                            (),
                            {
                                "x": 0,
                                "y": 0,
                                "width": patient.image.shape[1],
                                "height": patient.image.shape[0],
                            },
                        )()
                    },
                )()
            ],
        )

        logger.info(
            "Coordinates Generated : %d",
            len(coordinates),
        )

        for i, coordinate in enumerate(coordinates[:10]):

            logger.info(
                "[%04d] (%d,%d,%d,%d)",
                i,
                coordinate.x,
                coordinate.y,
                coordinate.width,
                coordinate.height,
            )

        ##################################################
        # Patch Extraction
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patch Extraction")
        logger.info("-" * 80)

        extractor = PatchExtractor()

        patches = extractor.extract(
            coordinates=coordinates,
            image=patient.image,
            annotation=patient.mask,
            source_filename=patient.image_filename,
        )

        logger.info(
            "Extracted Patches : %d",
            len(patches),
        )

        ##################################################
        # Tissue Filter
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Tissue Filter")
        logger.info("-" * 80)

        tissue_filter = TissuePatchFilter(
            threshold=0.50,
        )

        patches = tissue_filter.filter(
            patches,
        )

        logger.info(
            "Remaining Patches : %d",
            len(patches),
        )

        ##################################################
        # Weak Labels
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Weak Label Generation")
        logger.info("-" * 80)

        weak_label_generator = WeakLabelGenerator()

        patches = weak_label_generator.generate(
            patches,
        )

        ##################################################
        # Metadata
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Metadata Generation")
        logger.info("-" * 80)

        metadata_generator = PatchMetadataGenerator()

        patches = metadata_generator.generate(
            patches,
            original_filename=patient.image_filename,
        )

        ##################################################
        # Dataset Builder
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patch Collection Builder")
        logger.info("-" * 80)

        builder = PatchCollectionBuilder()

        dataset = builder.build(
            patient=patient,
            patches=patches,
        )

        logger.info(
            "Total Patches : %d",
            dataset.number_of_patches,
        )

        logger.info(
            "Average Tissue : %.4f",
            dataset.statistics.average_tissue_percentage,
        )

        logger.info(
            "Average Patch Area : %.2f",
            dataset.statistics.average_patch_area,
        )

        logger.info(
            "Class Distribution : %s",
            dataset.statistics.class_distribution,
        )

        ##################################################
        # Print Patch Information
        ##################################################

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patch Details")
        logger.info("-" * 80)

        # Create an output directory for the raw patches
        import os
        import cv2  # or torch, numpy, etc.
        patch_img_dir = f"outputs/patches/{patient.patient_id}/imgs"
        os.makedirs(patch_img_dir, exist_ok=True)
        patch_mask_dir = f"outputs/patches/{patient.patient_id}/mask"
        os.makedirs(patch_mask_dir, exist_ok=True)

        for patch in dataset.patches:
            # Save the image patch
            # img_path = os.path.join(patch_img_dir, f"patch_{patch.patch_id}_img.png")
            img_path = os.path.join(patch_img_dir, patch.metadata.image_filename)

            cv2.imwrite(img_path, patch.image_patch)
            
            # Save the mask patch
            # mask_path = os.path.join(patch_mask_dir, f"patch_{patch.patch_id}_mask.png")
            mask_path = os.path.join(patch_mask_dir, patch.metadata.annotation_filename)

            cv2.imwrite(mask_path, patch.annotation_patch)

            logger.info(
                "Patch %04d",
                patch.patch_id,
            )

            logger.info(
                "Coordinate : (%d,%d,%d,%d)",
                patch.coordinate.x,
                patch.coordinate.y,
                patch.coordinate.width,
                patch.coordinate.height,
            )

            logger.info(
                "Image Shape : %s",
                patch.image_patch.shape,
            )

            logger.info(
                "Mask Shape  : %s",
                patch.annotation_patch.shape,
            )

            logger.info(
                "Tissue %% : %.2f",
                patch.tissue_percentage * 100,
            )

            logger.info(
                "Detected Classes : %s",
                patch.detected_classes,
            )

            logger.info("")

        ##################################################
        # Visualization
        ##################################################

        PatchVisualizer.save(
            patient,
            f"outputs/patch_pipeline_v2/{patient.patient_id}",
        )
       
    logger.info("")
    logger.info("=" * 80)
    logger.info("PATCH PIPELINE V2 TEST FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()