"""
test_image_loader.py

Comprehensive test for ImageLoader.
"""

from configs.config import get_config

from loaders.image_loader import ImageLoader

from utils.logger import get_logger
from utils.logger import close_logger



def print_summary(logger, summary):

    logger.info("=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)

    for key, value in summary.items():
        logger.info("%-25s : %s", key, value)


def test_patient(loader, patient_id, logger):

    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading Patient : %s", patient_id)
    logger.info("=" * 80)

    pair = loader.get_pair(patient_id)

    logger.info("Image Path : %s", pair.image_path)
    logger.info("Mask Path  : %s", pair.mask_path)

    loader.load_patient(patient_id)

    image = loader.get_image()
    mask = loader.get_mask()
    metadata = loader.get_metadata()

    logger.info("")
    logger.info("IMAGE")

    logger.info("Shape      : %s", image.shape)
    logger.info("Dtype      : %s", image.dtype)
    logger.info("Minimum    : %d", image.min())
    logger.info("Maximum    : %d", image.max())

    logger.info("")
    logger.info("MASK")

    logger.info("Shape      : %s", mask.shape)
    logger.info("Dtype      : %s", mask.dtype)
    logger.info("Minimum    : %d", mask.min())
    logger.info("Maximum    : %d", mask.max())

    logger.info("")
    logger.info("METADATA")

    logger.info("Patient ID      : %s", metadata.patient_id)
    logger.info("Width           : %d", metadata.width)
    logger.info("Height          : %d", metadata.height)
    logger.info("Channels        : %d", metadata.channels)
    logger.info("Scale Factor    : %.4f", metadata.scale_factor)
    logger.info("Working Width   : %d", metadata.working_width)
    logger.info("Working Height  : %d", metadata.working_height)

    if hasattr(loader, "get_working_image"):

        working_image = loader.get_working_image()

        logger.info("")
        logger.info("WORKING IMAGE")

        logger.info("Shape      : %s", working_image.shape)
        logger.info("Dtype      : %s", working_image.dtype)

    if hasattr(loader, "get_working_mask"):

        working_mask = loader.get_working_mask()

        logger.info("")
        logger.info("WORKING MASK")

        logger.info("Shape      : %s", working_mask.shape)
        logger.info("Dtype      : %s", working_mask.dtype)

    logger.info("")
    logger.info("CACHE STATUS")

    logger.info("Current Patient : %s", loader.current_patient())
    logger.info("Image Loaded    : %s", loader.get_image() is not None)
    logger.info("Mask Loaded     : %s", loader.get_mask() is not None)
    logger.info("Metadata Loaded : %s", loader.get_metadata() is not None)


def main():

    logger = get_logger(
        name="ImageLoaderTest",
        log_dir="logs",
        level="INFO",
    )

    logger.info("=" * 80)
    logger.info("STARTING IMAGE LOADER TEST")
    logger.info("=" * 80)

    config = get_config()

    logger.info("Image Directory           : %s", config.image_dir)
    logger.info("Mask Directory            : %s", config.mask_dir)
    logger.info("Source Magnification      : %.2f",
                getattr(config, "source_magnification", 40.0))
    logger.info("Target Magnification      : %.2f",
                getattr(config,
                        "target_magnification",
                        getattr(config, "magnification", 10.0)))

    logger.info("")
    logger.info("Initializing ImageLoader...")

    loader = ImageLoader(
        config=config,
        logger=logger,
    )

    logger.info("Initialization Complete.")

    print_summary(
        logger,
        loader.summary(),
    )

    logger.info("")
    logger.info("Total Patients : %d",
                loader.number_of_patients)

    logger.info("")

    for patient in loader.patient_ids:
        logger.info(patient)

    logger.info("")

    for patient in loader.patient_ids:
        test_patient(
            loader,
            patient,
            logger,
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL VALIDATION")
    logger.info("=" * 80)

    logger.info("Total Patients : %d",
                loader.number_of_patients)

    logger.info("len(loader)    : %d",
                len(loader))

    logger.info("repr(loader)   : %s",
                loader)

    logger.info("")
    logger.info("=" * 80)
    logger.info("IMAGE LOADER TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    close_logger(logger)


if __name__ == "__main__":
    main()