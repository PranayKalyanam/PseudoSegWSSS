"""
patient_loader.py

Discovers all patients available in the dataset.

Unlike the ImageLoader, this loader does not read
images from disk. It simply identifies valid
image-mask pairs and creates Patient objects that
will be progressively enriched during the
preprocessing pipeline.

Pipeline
--------
Dataset
    ↓
PatientLoader
    ↓
List[Patient]
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from configs.config import get_config

from data.patient import Patient
from data.image.image_pair import ImagePair

from methods.image.image_discovery import pair_images_and_masks

from utils.logger import get_logger
from utils.validator import Validator


class PatientLoader:
    """
    Dataset patient discovery loader.

    Responsibilities
    ----------------
    1. Validate dataset directories.
    2. Discover image-mask pairs.
    3. Create Patient objects.
    4. Return complete patient list.

    This loader performs no image processing.
    """

    def __init__(
        self,
        config: get_config(),
        logger=None,
    ) -> None:

        self.config = config

        self.logger = logger or get_logger("PatientLoader")

        self.image_directory = Path(config.image_dir)

        self.mask_directory = Path(config.mask_dir)

        Validator.validate_directory(
            self.image_directory
        )

        Validator.validate_directory(
            self.mask_directory
        )

    # ---------------------------------------------------------
    # Public Interface
    # ---------------------------------------------------------

    # def load(self) -> List[Patient]:
    #     """
    #     Discover all patients.

    #     Returns
    #     -------
    #     List[Patient]
    #         Dataset patients.
    #     """

    #     self.logger.info(
    #         "Discovering patients..."
    #     )

    #     image_pairs = pair_images_and_masks(
    #         self.image_directory,
    #         self.mask_directory,
    #     )

    #     patients = [
    #         Patient(image_pair=pair)
    #         for pair in image_pairs
    #     ]

    #     self.logger.info(
    #         "Discovered %d patients.",
    #         len(patients),
    #     )

    #     return patients
    def load(
        self,
        image_pair: ImagePair,
    ) -> Patient:
        """
        Create a Patient object from an image-mask pair.

        Parameters
        ----------
        image_pair
            Image and annotation mask pair.

        Returns
        -------
        Patient
            Initialized patient object.
        """

        return Patient(
            image_pair=image_pair
        )


    def load_all(self) -> List[Patient]:
        """
        Discover all patients in the dataset.

        Returns
        -------
        List[Patient]
            List of discovered patients.
        """

        self.logger.info(
            "Discovering patients..."
        )

        image_pairs = pair_images_and_masks(
            self.image_directory,
            self.mask_directory,
        )

        patients = []

        for image_pair in image_pairs:

            patients.append(
                self.load(image_pair)
            )

        self.logger.info(
            "Discovered %d patients.",
            len(patients),
        )

        return patients
    # ---------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------

    @property
    def dataset_name(self) -> str:
        """Dataset name."""

        return self.config.dataset

    @property
    def image_directory_path(self) -> Path:
        """Image directory."""

        return self.image_directory

    @property
    def mask_directory_path(self) -> Path:
        """Mask directory."""

        return self.mask_directory

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"dataset='{self.dataset_name}')"
        )