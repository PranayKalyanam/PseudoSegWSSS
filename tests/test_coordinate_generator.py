from __future__ import annotations

from configs.config import get_config

from loaders.patient_loader import PatientLoader
from loaders.image_loader import ImageLoader
from loaders.tissue_loader import TissueLoader

from methods.patch.coordinate_generator import CoordinateGenerator

from utils.logger import get_logger


def main():

    logger = get_logger("CoordinateGeneratorTest")

    logger.info("=" * 80)
    logger.info("COORDINATE GENERATOR TEST")
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
    # Coordinate Generator
    # --------------------------------------------------

    coordinate_generator = CoordinateGenerator(
        config=config,
        patch_size=0,
        overlap=0,
    )

    logger.info("=" * 80)
    logger.info("COORDINATE GENERATION RESULTS")
    logger.info("=" * 80)

    for patient_index, patient in enumerate(
        patients,
        start=1,
    ):

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "PATIENT %d : %s",
            patient_index,
            patient.patient_id,
        )
        logger.info("=" * 80)

        logger.info(
            "Original Image Shape : %s",
            patient.image.shape,
        )

        logger.info(
            "Original Mask Shape  : %s",
            patient.mask.shape,
        )

        logger.info(
            "Working Image Shape : %s",
            patient.working_image.shape,
        )

        logger.info(
            "Number of Tissue Regions : %d",
            patient.number_of_tissue_regions,
        )

        logger.info("")

        logger.info("Tissue Regions")

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

        logger.info("")

        coordinates = coordinate_generator.generate(
            patient.working_image,
            patient.tissue_regions,
        )

        logger.info(
            "Coordinates Generated : %d",
            len(coordinates),
        )

        logger.info("")

        for index, coordinate in enumerate(coordinates):

            logger.info(
                "[%04d] "
                "x=%4d "
                "y=%4d "
                "w=%3d "
                "h=%3d "
                "area=%6d",
                index,
                coordinate.x,
                coordinate.y,
                coordinate.width,
                coordinate.height,
                coordinate.area,
            )

        logger.info("")
        logger.info("-" * 80)

    logger.info("")
    logger.info("=" * 80)
    logger.info("COORDINATE GENERATOR TEST FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()