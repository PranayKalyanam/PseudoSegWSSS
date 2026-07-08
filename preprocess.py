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
import time


# =============================================================================
# Project Imports
# =============================================================================

from configs.config import get_config

from dataset.data_structures import PatchRecord
from preprocessing.image_loader import ImageLoader
from preprocessing.tissue_detection import OtsuTissueDetector
from preprocessing.patch_extractor import BasePatchExtractor
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
            config.mask_dir
        )

        logger.info(
            "Output Root  : %s",
            config.output_root,
        )

        logger.info(
            "Patch Size   : %d",
            config.patch_size,
        )

        logger.info(
            "Overlap      : %.2f",
            config.patch_overlap,
        )

        logger.info(
            "Magnification: %dx",
            config.target_magnification,
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

        patient_dataset = image_loader.load_dataset()

        logger.info(
            "Total Patients : %d",
            len(patient_dataset),
        )

        processed_patients = 0

        failed_patients = 0

        # ==============================================================
        # Process Patients
        # ==============================================================

        for patient in patient_dataset:

            patient_id = patient.patient_id

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

                image = image_loader.load_image(
                    patient
                )

                mask = image_loader.load_mask(
                    patient
                )

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
                    config.target_magnification,
                )

                # ------------------------------------------------------
                # Tissue Detection
                # ------------------------------------------------------

                logger.info(
                    "Detecting tissue regions..."
                )

                tissue_result = tissue_detector.detect(
                    image_10x
                )

                logger.info(
                    "Detected %d tissue regions.",
                    len(
                        tissue_result.regions
                    ),
                )

                # ------------------------------------------------------
                # Processing continues in Section 2B
                # ------------------------------------------------------
                
                                # ==========================================================
                # Process Tissue Regions
                # ==========================================================

                if len(tissue_result.regions) == 0:

                    logger.warning(
                        "No valid tissue regions detected for patient %s.",
                        patient_id,
                    )

                    continue

                logger.info(
                    "Beginning tissue region processing..."
                )

                #
                # Manifest for current patient.
                # All extracted patches and metadata will be accumulated here
                # before writing to disk.
                #

                patient_manifest = DatasetManifest()

                total_regions = len(
                    tissue_result.regions
                )

                logger.info(
                    "Total Regions : %d",
                    total_regions,
                )

                # ----------------------------------------------------------
                # Process each detected tissue region independently
                # ----------------------------------------------------------

                for region_index, region in enumerate(
                    tissue_result.regions,
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
                        region.x_min,
                        region.y_min,
                        region.x_max,
                        region.y_max,
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
                        region.y_min:region.y_max,
                        region.x_min:region.x_max,
                    ]

                    region_mask = mask_10x[
                        region.y_min:region.y_max,
                        region.x_min:region.x_max,
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

                    patch_iterator = patch_extractor.extract(
                        image=region_image,
                        mask=region_mask,
                        patient_id=patient_id,
                        region_id=region_index,
                        region_origin=(
                            region.x_min,
                            region.y_min,
                        ),
                    )

                    extracted_patch_count = 0

                    for patch_candidate in patch_iterator:

                        extracted_patch_count += 1

                        logger.debug(

                            "Patch %d | WSI Coordinate (%d, %d)",

                            extracted_patch_count,

                            patch_candidate.x,

                            patch_candidate.y,

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

                        weak_label = label_generator.generate(
                            patch_candidate.mask
                        )

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
                            f"{patch_candidate.x}_"
                            f"{patch_candidate.y}_"
                            f"[{label_string}].png"
                        )

                        logger.debug(
                            "Patch Filename : %s",
                            filename,
                        )

                        # --------------------------------------------------
                        # Create patch record
                        # --------------------------------------------------

                        patch_record = PatchRecord(

                            patch_id=f"{patient_id}_{patch_candidate.x}_{patch_candidate.y}",

                            patient_id=patient_id,

                            region_id=region_index,

                            filename=filename,

                            image=patch_candidate.image,

                            mask=patch_candidate.mask,

                            x=patch_candidate.x,

                            y=patch_candidate.y,

                            width=config.patch_size,

                            height=config.patch_size,

                            tissue_percentage=patch_candidate.tissue_percentage,

                            labels=weak_label,

                        )

                        # --------------------------------------------------
                        # Store in patient manifest
                        # --------------------------------------------------

                        patient_manifest.records.append(
                            patch_record
                        )

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

                    write_result = dataset_writer(
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

                    # ----------------------------------------------------------
                    # Release Memory
                    # ----------------------------------------------------------

                    del patient_manifest
                    del image
                    del mask
                    del image_10x
                    del mask_10x
                    del tissue_result

                    import gc

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

        total_patients = len(patient_dataset)

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
            config.output_root,
        )

        logger.info(
            "Patch Directory        : %s",
            config.output_root / "patches",
        )

        logger.info(
            "Patch Mask Directory   : %s",
            config.output_root / "patchesMasks",
        )

        logger.info(
            "Metadata Directory     : %s",
            config.output_root / "metadata",
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