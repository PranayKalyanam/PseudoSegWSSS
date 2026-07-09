"""
test_image_loader.py

Tests the complete image loading stage.

Pipeline
--------
Dataset
    ↓
PatientLoader
    ↓
ImageLoader
    ↓
ImageData
"""

# from configs.config import get_config

# config = get_config()

# print(config)




from configs.config import get_config

from loaders.patient_loader import PatientLoader
from loaders.image_loader import ImageLoader

from utils.logger import get_logger


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = get_config()

    logger = get_logger("ImageLoaderTest")

    logger.info("=" * 80)
    logger.info("IMAGE LOADER TEST")
    logger.info("=" * 80)

    # --------------------------------------------------
    # Discover Patients
    # --------------------------------------------------

    patient_loader = PatientLoader(
        config=config,
        logger=logger,
    )

    patients = patient_loader.load_all()

    logger.info("Patients discovered : %d", len(patients))

    if len(patients) == 0:

        logger.error("No patients discovered.")

        return

    # --------------------------------------------------
    # Image Loader
    # --------------------------------------------------

    image_loader = ImageLoader(
        config=config,
        logger=logger,
    )

    logger.info("")
    logger.info("Loading images...")
    logger.info("")

    patients = image_loader.load_all(patients)

    logger.info("")
    logger.info("=" * 80)
    logger.info("IMAGE LOADING RESULTS")
    logger.info("=" * 80)

    for index, patient in enumerate(patients, start=1):

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patient %d", index)
        logger.info("-" * 80)

        logger.info("Patient ID          : %s", patient.patient_id)

        logger.info("Image Path          : %s", patient.image_path)
        logger.info("Mask Path           : %s", patient.mask_path)

        logger.info("")

        logger.info("Image Loaded        : %s", patient.has_image)

        if not patient.has_image:

            continue

        metadata = patient.metadata

        logger.info("")

        logger.info("Original Size       : %d x %d",
                    metadata.width,
                    metadata.height)

        logger.info("Working Size        : %d x %d",
                    metadata.working_width,
                    metadata.working_height)

        logger.info("Channels            : %d",
                    metadata.channels)

        logger.info("Data Type           : %s",
                    metadata.image_dtype)

        logger.info("")

        logger.info("Source Magnification: %.2fx",
                    metadata.source_magnification)

        logger.info("Target Magnification: %.2fx",
                    metadata.target_magnification)

        logger.info("Scale Factor        : %.4f",
                    metadata.scale_factor)

        logger.info("")

        logger.info("Original Image Shape: %s",
                    patient.image.shape)

        logger.info("Original Mask Shape : %s",
                    patient.mask.shape)

        logger.info("Working Image Shape : %s",
                    patient.image_data.working_image.shape)

        logger.info("Working Mask Shape  : %s",
                    patient.image_data.working_mask.shape)

    logger.info("")
    logger.info("=" * 80)
    logger.info("IMAGE LOADER TEST PASSED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()