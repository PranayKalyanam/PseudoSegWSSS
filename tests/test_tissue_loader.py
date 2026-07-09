from __future__ import annotations

from configs.config import get_config

from loaders.patient_loader import PatientLoader
from loaders.image_loader import ImageLoader
from loaders.tissue_loader import TissueLoader

from methods.visualization.tissue_visualizer import TissueVisualizer

from utils.logger import get_logger


def main():

    logger = get_logger("TissueLoaderTest")

    logger.info("=" * 80)
    logger.info("TISSUE LOADER TEST")
    logger.info("=" * 80)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = get_config()

    # --------------------------------------------------
    # Patient Discovery
    # --------------------------------------------------

    logger.info("Discovering patients...")

    patient_loader = PatientLoader(
        config=config,
        logger=logger,
    )

    patients = patient_loader.load_all()

    logger.info(
        "Patients discovered : %d",
        len(patients),
    )

    logger.info("")

    # --------------------------------------------------
    # Image Loading
    # --------------------------------------------------

    logger.info("Loading images...")

    image_loader = ImageLoader(
        config=config,
        logger=logger,
    )

    patients = image_loader.load_all(
        patients,
    )

    logger.info("")

    # --------------------------------------------------
    # Tissue Detection
    # --------------------------------------------------

    logger.info("Detecting tissue...")

    tissue_loader = TissueLoader(
        config=config,
        logger=logger,
    )

    patients = tissue_loader.load_all(
        patients,
    )

    logger.info("")

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    logger.info("=" * 80)
    logger.info("TISSUE DETECTION RESULTS")
    logger.info("=" * 80)

    for index, patient in enumerate(
        patients,
        start=1,
    ):

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patient %d", index)
        logger.info("-" * 80)

        logger.info(
            "Patient ID           : %s",
            patient.patient_id,
        )

        logger.info(
            "Detector             : %s",
            patient.tissue_mask.detector_name,
        )

        logger.info(
            "Mask Shape           : %s",
            patient.tissue_binary_mask.shape,
        )

        logger.info(
            "Number of Regions    : %d",
            patient.number_of_tissue_regions,
        )

        stats = patient.tissue_statistics

        logger.info("")
        logger.info("Pixel Statistics")

        logger.info(
            "Total Pixels         : %d",
            stats.total_pixels,
        )

        logger.info(
            "Tissue Pixels        : %d",
            stats.tissue_pixels,
        )

        logger.info(
            "Background Pixels    : %d",
            stats.background_pixels,
        )

        logger.info(
            "Tissue Percentage    : %.2f%%",
            stats.tissue_percentage,
        )

        logger.info(
            "Background Percentage: %.2f%%",
            stats.background_percentage,
        )

        logger.info("")
        logger.info("Region Statistics")

        logger.info(
            "Largest Region       : %d",
            stats.largest_region_area,
        )

        logger.info(
            "Smallest Region      : %d",
            stats.smallest_region_area,
        )

        logger.info(
            "Mean Region Area     : %.2f",
            stats.mean_region_area,
        )

        logger.info("")

        for region in patient.tissue_regions:

            logger.info(
                "Region %-3d "
                "BBox=(%d,%d,%d,%d) "
                "Area=%d",
                region.region_id,
                region.x,
                region.y,
                region.width,
                region.height,
                region.area,
            )
        TissueVisualizer.save(
            patient,
            f"outputs/tissue_loader/{patient.patient_id}",
            )

    logger.info("")
    logger.info("=" * 80)
    logger.info("TISSUE LOADER TEST PASSED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()