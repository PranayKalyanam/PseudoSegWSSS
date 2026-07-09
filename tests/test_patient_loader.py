from configs.config import get_config
from loaders.patient_loader import PatientLoader
from utils.logger import get_logger


def main():

    config = get_config()

    logger = get_logger(
        "PatientLoaderTest"
    )

    logger.info("=" * 80)
    logger.info("PATIENT LOADER TEST")
    logger.info("=" * 80)

    loader = PatientLoader(
        config=config,
        logger=logger,
    )

    patients = loader.load_all()

    logger.info("")

    logger.info("Patients discovered : %d", len(patients))

    logger.info("")

    for i, patient in enumerate(patients, 1):

        logger.info("-" * 60)

        logger.info("Patient %d", i)

        logger.info("ID          : %s", patient.patient_id)

        logger.info("Image       : %s", patient.image_path)

        logger.info("Mask        : %s", patient.mask_path)

    logger.info("")

    logger.info("Test completed successfully.")


if __name__ == "__main__":
    main()