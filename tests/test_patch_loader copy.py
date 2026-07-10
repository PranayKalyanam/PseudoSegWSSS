from __future__ import annotations

import logging
from pathlib import Path

# from configs import config
from configs.config import get_config
from loaders.patient_loader import PatientLoader
from loaders.image_loader import ImageLoader
from loaders.tissue_loader import TissueLoader
from loaders.patch_loader import PatchLoader

# from methods.visualization.patch_visualizer import PatchVisualizer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [PatchLoaderTest] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("PatchLoaderTest")


def main():

    logger.info("=" * 80)
    logger.info("PATCH LOADER TEST")
    logger.info("=" * 80)

    # --------------------------------------------------
    # Patient Discovery
    # --------------------------------------------------
    config = get_config()
    patient_loader = PatientLoader(config)

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

    image_loader = ImageLoader(config)

    image_loader.load_all(patients)

    logger.info("")

    # --------------------------------------------------
    # Tissue Detection
    # --------------------------------------------------

    logger.info("Detecting tissue...")

    tissue_loader = TissueLoader(config)

    tissue_loader.load_all(patients)

    logger.info("")

    # --------------------------------------------------
    # Patch Extraction
    # --------------------------------------------------

    logger.info("Extracting patches...")

    patch_loader = PatchLoader(config)

    patch_loader.load_all(patients)

    logger.info("")

    logger.info("=" * 80)
    logger.info("PATCH EXTRACTION RESULTS")
    logger.info("=" * 80)

    output_dir = Path("outputs/patch_loader")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, patient in enumerate(patients, start=1):

        logger.info("")
        logger.info("-" * 80)
        logger.info("Patient %d", index)
        logger.info("-" * 80)

        logger.info(
            "Patient ID        : %s",
            patient.patient_id,
        )

        logger.info(
            "Patch Count       : %d",
            patient.number_of_patches,
        )

        logger.info(
            "Average Tissue %%  : %.2f",
            patient.patch_statistics.mean_tissue_percentage,
        )

        logger.info(
            "Maximum Tissue %%  : %.2f",
            patient.patch_statistics.max_tissue_percentage,
        )

        logger.info(
            "Minimum Tissue %%  : %.2f",
            patient.patch_statistics.min_tissue_percentage,
        )

        logger.info(
            "Tumor Patches     : %d",
            patient.patch_statistics.number_of_tumor_patches,
        )

        logger.info(
            "Stroma Patches    : %d",
            patient.patch_statistics.number_of_stroma_patches,
        )

        logger.info(
            "Lymphocyte Patches: %d",
            patient.patch_statistics.number_of_lymphocyte_patches,
        )

        logger.info(
            "Necrosis Patches  : %d",
            patient.patch_statistics.number_of_necrosis_patches,
        )

        logger.info("")

        logger.info("First Five Patches")

        for patch in patient.patches[:5]:

            logger.info(
                "Patch %-4d  (%4d,%4d)  Tissue=%6.2f%%  Labels=%s",
                patch.metadata.patch_id,
                patch.coordinate.global_x,
                patch.coordinate.global_y,
                patch.metadata.tissue_percentage,
                patch.label.to_binary_string(),
            )

        # --------------------------------------------------
        # Visualization
        # --------------------------------------------------

        # PatchVisualizer.save(
        #     patient=patient,
        #     output_directory=output_dir / patient.patient_id,
        # )

    logger.info("")
    logger.info("=" * 80)
    logger.info("PATCH LOADER TEST PASSED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()