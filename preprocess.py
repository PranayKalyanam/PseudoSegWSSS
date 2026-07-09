#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
File        : preprocess.py

Description
-----------
Main entry point for the complete histopathology preprocessing pipeline.

Pipeline
--------
WSI + Mask
      │
      ▼
ImageLoader
      │
      ▼
MagnificationConverter
      │
      ▼
TissueDetector
      │
      ▼
PatchExtractor
      │
      ▼
LabelGenerator
      │
      ▼
DatasetWriter

This script contains NO preprocessing logic.
It only orchestrates the pipeline.

==============================================================================
"""

from __future__ import annotations


    
    


# =============================================================================
# Standard Library Imports
# =============================================================================

from pathlib import Path
import gc
import time


# =============================================================================
# Project Imports
# =============================================================================

from configs.config import get_config

from dataset.data_structures import PatchRecord
from preprocessing.image_loader import ImageLoader
from preprocessing.tissue_detection import OtsuTissueDetector
from preprocessing.patch_extractor import BasePatchExtractor

import inspect

# print(BasePatchExtractor)
# print(inspect.getfile(BasePatchExtractor))
# print(inspect.signature(BasePatchExtractor.extract))

# print(inspect.getsource(BasePatchExtractor.extract))


from preprocessing.label_generator import BCSSLabelGenerator, BaseLabelGenerator, LabelGeneratorConfig

from dataio.dataset_writer import (
    DatasetWriter,
    DatasetManifest,
    DatasetWriterConfig,
    OutputFormat,
    OverwritePolicy,
)

from utils.logger import get_logger

from utils.exceptions import (
    ConfigurationError,
    DatasetError,
    FileReadError,
    FileWriteError,
    PreprocessingError,
)

# =============================================================================
# Constants
# =============================================================================

PROGRAM_NAME = "Histopathology Preprocessing Pipeline"

VERSION = "1.0.0"

EXIT_SUCCESS = 0

EXIT_FAILURE = 1

# =============================================================================
# Helper Functions
# =============================================================================


def banner() -> None:
    """
    Print program banner.
    """

    print("=" * 80)
    print(PROGRAM_NAME)
    print(f"Version : {VERSION}")
    print("=" * 80)


# -----------------------------------------------------------------------------


def elapsed_time(
    start_time: float,
) -> float:
    """
    Calculate elapsed execution time.

    Parameters
    ----------
    start_time : float

    Returns
    -------
    float
        Elapsed time in seconds.
    """

    return time.perf_counter() - start_time


# =============================================================================
# Main Entry Point
# =============================================================================


# =============================================================================
# Pipeline Initialization
# =============================================================================


def main() -> int:
    """
    Main preprocessing pipeline.
    """

    start_time = time.perf_counter()

    banner()

    logger = get_logger("PreprocessingPipeline")

    logger.info("=" * 80)
    logger.info("Starting preprocessing pipeline.")
    logger.info("=" * 80)

    try:

        # ------------------------------------------------------------------
        # Load configuration
        # ------------------------------------------------------------------

        logger.info("Loading configuration...")

        config = get_config()

        logger.info("Configuration loaded successfully.")

        # ------------------------------------------------------------------
        # Dry Run
        # ------------------------------------------------------------------

        if config.dry_run:

            logger.info("Dry-run completed successfully.")
            logger.info("Configuration is valid.")

            return EXIT_SUCCESS

        # ------------------------------------------------------------------
        # Initialize Components
        # ------------------------------------------------------------------

        logger.info("Initializing preprocessing modules...")

        image_loader = ImageLoader(config)


        tissue_detector = OtsuTissueDetector(config)

        patch_extractor = BasePatchExtractor(config)

        # label_generator = BCSSLabelGenerator(config)
        label_generator = BCSSLabelGenerator(
                                LabelGeneratorConfig()
                            )

        dataset_writer = DatasetWriter(
                                DatasetWriterConfig(
                                    output_dir=Path(config.output_dir),
                                    metadata_format=OutputFormat.BOTH,
                                    overwrite_policy=OverwritePolicy.SKIP,
                                    atomic_write=True,
                                    verify_after_write=True,
                                    save_statistics=True,
                                    debug=(config.log_level == "DEBUG"),
                                )
                            )

        logger.info(
            "All preprocessing modules initialized."
        )

        # ------------------------------------------------------------------
        # Pipeline Summary
        # ------------------------------------------------------------------

        logger.info("")
        logger.info("Pipeline Configuration")
        logger.info("-" * 60)

        logger.info(
            "Dataset Root : %s",
            config.image_dir,
        )

        logger.info(
            "Mask Root    : %s",
            config.mask_dir,
        )

        logger.info(
            "Output Root  : %s",
            config.output_dir,
        )

        logger.info(
            "Patch Size   : %d",
            config.patch_size,
        )

        logger.info(
            "Overlap      : %.2f",
            config.overlap,
        )

        logger.info(
            "Magnification: %dx",
            config.magnification,
        )

        logger.info(
            "Workers       : %d",
            config.workers,
        )

        logger.info("-" * 60)

        # ------------------------------------------------------------------
        # Processing starts in Section 2A
        # ------------------------------------------------------------------

        logger.info(
            "Initialization complete."
        )

        logger.info(
            "Beginning patient processing..."
        )
        
                # ==============================================================
        # Load Patient Dataset
        # ==============================================================

        logger.info(
            "Scanning dataset..."
        )

        patient_ids = image_loader.patient_ids

        logger.info(
            "Total Patients : %d",
            len(patient_ids),
        )

        processed_patients = 0

        failed_patients = 0

        # ==============================================================
        # Process Patients
        # ==============================================================

        for patient_id in patient_ids:

            logger.info("")
            logger.info("=" * 80)
            logger.info(
                "Processing Patient : %s",
                patient_id,
            )
            logger.info("=" * 80)

            try:

                # ------------------------------------------------------
                # Load Image and Mask
                # ------------------------------------------------------

                image_loader.load_patient(
                    patient_id
                )

                image = image_loader.get_image()

                mask = image_loader.get_mask()

                logger.info(
                    "Image Shape : %s",
                    image.shape,
                )

                logger.info(
                    "Mask Shape : %s",
                    mask.shape,
                )
                
                #
                # Images are already stored at the target magnification.
                #

                image_10x = image

                mask_10x = mask

                logger.info(
                    "Images already available at target magnification (%dx).",
                    config.magnification,
                )

                # ------------------------------------------------------
                # Tissue Detection
                # ------------------------------------------------------

                logger.info(
                    "Detecting tissue regions..."
                )

                tissue_result = tissue_detector.detect(
                    image_10x,
                    patient_id=patient_id,
                )
                logger.info("Return type: %s", type(tissue_result))
                
                # logger.info(tissue_result.tissue_mask)
                
                tissue_mask = tissue_result.tissue_mask

                # logger.info(
                #     "Attributes: %s",
                #     dir(tissue_result)
                #     )
                # logger.info("Returned object: %s", tissue_result)

                region_list = list(getattr(tissue_result, "tissue_regions", []) or getattr(tissue_result, "regions", []))

                logger.info(
                    "Detected %d tissue regions.",
                    len(region_list),
                )

                # ------------------------------------------------------
                # Processing continues in Section 2B
                # ------------------------------------------------------
                
                                # ==========================================================
                # Process Tissue Regions
                # ==========================================================

                if len(region_list) == 0:

                    logger.warning(
                        "No valid tissue regions detected for patient %s.",
                        patient_id,
                    )

                    continue

                logger.info(
                    "Beginning tissue region processing..."
                )

                patient_manifest = DatasetManifest()

                total_regions = len(region_list)

                logger.info(
                    "Total Regions : %d",
                    total_regions,
                )

                # ----------------------------------------------------------
                # Process each detected tissue region independently
                # ----------------------------------------------------------

                for region_index, region in enumerate(
                    region_list,
                    start=1,
                ):

                    logger.info("")
                    logger.info(
                        "-" * 60
                    )

                    logger.info(
                        "Region %d / %d",
                        region_index,
                        total_regions,
                    )

                    logger.info(
                        "Bounding Box : (%d, %d, %d, %d)",
                        region.x,
                        region.y,
                        region.width,
                        region.height,
                    )

                    logger.info(
                        "Region Size : %d × %d",
                        region.width,
                        region.height,
                    )

                    logger.info(
                        "Tissue Percentage : %.2f%%",
                        region.tissue_percentage,
                    )

                    # ------------------------------------------------------
                    # Ignore very small regions
                    # ------------------------------------------------------

                    if (
                        region.width
                        < config.patch_size
                        or
                        region.height
                        < config.patch_size
                    ):

                        logger.warning(
                            "Skipping small tissue region."
                        )

                        continue

                    # ------------------------------------------------------
                    # Crop image ROI
                    # ------------------------------------------------------

                    region_image = image_10x[
                        region.y:region.y + region.height,
                        region.x:region.x + region.width,
                    ]

                    region_mask = mask_10x[
                        region.y:region.y + region.height,
                        region.x:region.x + region.width,
                    ]
                    
                    region_tissue_mask = tissue_mask[
                        region.y:region.y + region.height,
                        region.x:region.x + region.width,
                    ]

                    logger.debug(
                        "ROI extracted successfully."
                    )

                    # ------------------------------------------------------
                    # Patch extraction begins in Part 2
                    # ------------------------------------------------------
                    
                                        # ======================================================
                    # Extract Patches from Current Tissue Region
                    # ======================================================

                    logger.info(
                        "Extracting patches..."
                    )

                    # extracted_patches = patch_extractor.extract(
                    #     patient_id=patient_id,
                    #     image=region_image,
                    #     mask=region_mask,
                    #     tissue_mask=region_mask,
                    #     tissue_regions=[region],
                    # )
                    extracted_patches = patch_extractor.extract(
                                image=region_image,
                                mask=region_mask,
                                patient_id=patient_id,
                                tissue_mask=region_tissue_mask,
                                roi_origin=(region.x, region.y),
                                region=region,
                            )

                    labelled_patches = label_generator.generate_labels(extracted_patches)
                    
                    logger.info("Extracted patches: %d", len(extracted_patches))
                    logger.info("Returned type: %s", type(labelled_patches))
                    # logger.info("Returned value: %s", labelled_patches)

                    extracted_patch_count = 0

                    for patch_candidate in labelled_patches:

                        extracted_patch_count += 1

                        logger.debug(

                            "Patch %d | WSI Coordinate (%d, %d)",

                            extracted_patch_count,

                            patch_candidate.metadata.coordinate.x,

                            patch_candidate.metadata.coordinate.y,

                        )

                        #
                        # PatchCandidate contains:
                        #
                        # patch_candidate.image
                        # patch_candidate.mask
                        # patch_candidate.x
                        # patch_candidate.y
                        # patch_candidate.patient_id
                        # patch_candidate.region_id
                        # patch_candidate.tissue_percentage
                        #

                        if patch_candidate.image is None:

                            logger.debug(
                                "Skipping invalid image patch."
                            )

                            continue

                        if patch_candidate.mask is None:

                            logger.debug(
                                "Skipping invalid mask patch."
                            )

                            continue

                        if (
                            patch_candidate.image.shape[0]
                            != config.patch_size
                            or
                            patch_candidate.image.shape[1]
                            != config.patch_size
                        ):

                            logger.debug(
                                "Ignoring incomplete border patch."
                            )

                            continue

                        if (
                            patch_candidate.mask.shape[0]
                            != config.patch_size
                            or
                            patch_candidate.mask.shape[1]
                            != config.patch_size
                        ):

                            logger.debug(
                                "Ignoring incomplete border mask."
                            )

                            continue

                        #
                        # --------------------------------------------------
                        # Weak-label generation begins in Part 3
                        # --------------------------------------------------
                        #
                        
                        # ==================================================
                        # Generate Weak Multi-label
                        # ==================================================

                        weak_label = [
                            int(patch_candidate.metadata.label.tumor),
                            int(patch_candidate.metadata.label.stroma),
                            int(patch_candidate.metadata.label.lymphocyte),
                            int(patch_candidate.metadata.label.necrosis),
                        ]

                        logger.debug(
                            "Weak Label : %s",
                            weak_label,
                        )

                        # --------------------------------------------------
                        # Skip background-only patches
                        # --------------------------------------------------

                        if not any(weak_label):

                            logger.debug(
                                "Skipping background-only patch."
                            )

                            continue

                        # --------------------------------------------------
                        # Convert label to filename representation
                        # Example:
                        # [1,0,1,0]
                        # ->
                        # 1010
                        # --------------------------------------------------

                        label_string = "".join(
                            str(int(value))
                            for value in weak_label
                        )

                        filename = (
                            f"{patient_id}_"
                            f"R{region_index:03d}_"
                            f"P{patch_candidate.metadata.patch_id:06d}_"
                            f"X{patch_candidate.metadata.coordinate.x:06d}_"
                            f"Y{patch_candidate.metadata.coordinate.y:06d}_"
                            f"L{label_string}.png"
                        )

                        logger.debug(
                            "Patch Filename : %s",
                            filename,
                        )

                        # --------------------------------------------------
                        # Create patch record
                        # --------------------------------------------------

                        patch_record = PatchRecord(
                            patch_id=patch_candidate.metadata.patch_id,
                            patient_id=patient_id,
                            region_id=region_index,
                            filename=filename,
                            image=patch_candidate.image,
                            mask=patch_candidate.mask,
                            x=patch_candidate.metadata.coordinate.x,
                            y=patch_candidate.metadata.coordinate.y,
                            width=patch_candidate.metadata.coordinate.width,
                            height=patch_candidate.metadata.coordinate.height,
                            tissue_percentage=patch_candidate.metadata.tissue_percentage,
                            labels=weak_label,
                        )

                        # --------------------------------------------------
                        # Store in patient manifest
                        # --------------------------------------------------

                        patient_manifest.records.append(patch_record)
                        patient_manifest.patches.append(patch_candidate)

                        logger.debug(
                            "Patch added to manifest."
                        )
                    
                        # ==========================================================
                     # Finalize Current Patient
                     # ==========================================================

                    logger.info("")
                    logger.info(
                        "=" * 70
                     )
                    logger.info(
                        "Finalizing Patient : %s",
                        patient_id,
                     )
                    logger.info(
                         "=" * 70
                    )

                    if len(patient_manifest.records) == 0:

                        logger.warning(
                            "Patient %s produced no valid patches.",
                            patient_id,
                        )

                        continue

                    logger.info(
                        "Total Valid Patches : %d",
                        len(patient_manifest.records),
                    )

                # ----------------------------------------------------------
                # Write Dataset
                # ----------------------------------------------------------

                    logger.info(
                        "Writing dataset..."
                    )

                    write_result = dataset_writer.write_dataset(
                        patient_manifest
                    )

                    if not write_result.success:

                        logger.error(
                            "Dataset writing failed for patient %s",
                            patient_id,
                        )

                        logger.error(
                            write_result.message
                        )

                        failed_patients += 1

                        continue

                    logger.info(
                        "Dataset successfully written."
                    )

                # ----------------------------------------------------------
                # Display Statistics
                # ----------------------------------------------------------

                    statistics = write_result.statistics

                    logger.info(
                        "Generated Files : %d",
                        statistics.generated_files,
                    )

                    logger.info(
                        "Failed Files : %d",
                        statistics.failed_files,
                    )

                    logger.info(
                        "Total Patches : %d",
                        statistics.total_patches,
                    )

                    logger.info(
                        "Average Tissue Percentage : %.2f%%",
                        statistics.average_tissue_percentage,
                    )

                # ----------------------------------------------------------
                # Verification
                # ----------------------------------------------------------

                    if statistics.failed_files > 0:

                        logger.warning(
                            "Some files failed during writing."
                        )

                    logger.info(
                        "Patient %s completed successfully.",
                        patient_id,
                    )

                    processed_patients += 1

                    del patient_manifest
                    del image
                    del mask
                    del image_10x
                    del mask_10x
                    del tissue_result
                    del region_image
                    del region_mask

                    gc.collect()

                    logger.debug(
                        "Memory released."
                    )

                    logger.info(

                        "Extracted %d patches from region %d.",

                        extracted_patch_count,

                        region_index,

                    )

                # processed_patients += 1

            except Exception as error:

                failed_patients += 1

                logger.exception(
                    "Failed patient %s",
                    patient_id,
                )

                logger.exception(
                    error,
                )

                continue

                # ==============================================================
        # Pipeline Summary
        # ==============================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 80)

        total_patients = len(patient_ids)

        logger.info(
            "Total Patients      : %d",
            total_patients,
        )

        logger.info(
            "Successfully Processed : %d",
            processed_patients,
        )

        logger.info(
            "Failed Patients        : %d",
            failed_patients,
        )

        skipped_patients = (
            total_patients
            - processed_patients
            - failed_patients
        )

        logger.info(
            "Skipped Patients       : %d",
            skipped_patients,
        )

        success_rate = (
            100.0 * processed_patients / total_patients
            if total_patients > 0
            else 0.0
        )

        logger.info(
            "Success Rate           : %.2f%%",
            success_rate,
        )

        logger.info(
            "Output Directory       : %s",
            config.output_dir,
        )

        logger.info(
            "Patch Directory        : %s",
            Path(config.output_dir) / "patches",
        )

        logger.info(
            "Patch Mask Directory   : %s",
            Path(config.output_dir) / "patchMasks",
        )

        logger.info(
            "Metadata Directory     : %s",
            Path(config.output_dir) / "metadata",
        )

        logger.info("=" * 80)

        return EXIT_SUCCESS
    finally:

        total_time = elapsed_time(start_time)

        logger.info("")

        logger.info("=" * 80)

        logger.info(
            "Pipeline Execution Time : %.2f seconds",
            total_time,
        )

        logger.info(
            "Pipeline Finished."
        )

        logger.info("=" * 80)


# -----------------------------------------------------------------------------

import sys
if __name__ == "__main__":

    sys.exit(main())