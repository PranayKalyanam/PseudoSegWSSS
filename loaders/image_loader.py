"""
image_loader.py

Coordinates the image loading stage of the preprocessing
pipeline.

The ImageLoader performs no image processing itself.
Instead, it orchestrates the reusable processing modules
responsible for image reading, validation, magnification
handling, resizing, and metadata generation.

Pipeline
--------
Patient
    ↓
ImagePair
    ↓
ImageReader
    ↓
ImageAlignment
    ↓
Magnification
    ↓
ImageResizer
    ↓
ImageMetadata
    ↓
ImageData
    ↓
Patient
"""

from __future__ import annotations

from data.patient import Patient

from data.image.image_size import ImageSize
from data.image.image_data import ImageData
from data.image.image_metadata import ImageMetadata

from methods.image.image_reader import ImageReader
from methods.image.image_alignment import ImageAlignment
from methods.image.image_resizer import ImageResizer
from methods.image.magnification import Magnification

from utils.logger import get_logger


class ImageLoader:
    """
    Image loading coordinator.

    This loader enriches a Patient object by reading
    its image and annotation mask and producing an
    ImageData object.
    """

    def __init__(
        self,
        config,
        logger=None,
    ) -> None:

        self.config = config

        self.logger = logger or get_logger(
            "ImageLoader"
        )

    # --------------------------------------------------
    # Public Interface
    # --------------------------------------------------

    def load(
        self,
        patient: Patient,
    ) -> Patient:
        """
        Load image information for a patient.

        Parameters
        ----------
        patient
            Patient discovered by PatientLoader.

        Returns
        -------
        Patient
            Updated patient.
        """

        self.logger.info(
            "Loading patient %s",
            patient.patient_id,
        )

        # ------------------------------------------
        # Read image and mask
        # ------------------------------------------

        image = ImageReader.read_rgb(
            patient.image_path
        )

        mask = ImageReader.read_mask(
            patient.mask_path
        )

        # ------------------------------------------
        # Validate alignment
        # ------------------------------------------

        ImageAlignment.validate(
            image,
            mask,
        )

        # ------------------------------------------
        # Original image size
        # ------------------------------------------

        original_size = ImageReader.image_size(
            image
        )

        # ------------------------------------------
        # Magnification
        # ------------------------------------------

        # Detect source magnification from the image
        source_mag = Magnification.detect(
            patient.image_path
        )

        # Target magnification from configuration
        target_mag = self.config.magnification

        # Compute resize scale factor
        scale_factor = Magnification.compute_scale_factor(
            source=source_mag,
            target=target_mag,
        )

        # ------------------------------------------
        # Resize
        # ------------------------------------------

        (
            working_image,
            working_mask,
            working_size,
        ) = ImageResizer.resize_by_scale(
            image=image,
            mask=mask,
            scale_factor=scale_factor,
        )

        # ------------------------------------------
        # Metadata
        # ------------------------------------------

        metadata = ImageMetadata(
            patient_id=patient.patient_id,

            width=original_size.width,
            height=original_size.height,

            channels=ImageReader.number_of_channels(
                image
            ),

            image_dtype=ImageReader.dtype(
                image
            ),

            source_magnification=source_mag,
            target_magnification=target_mag,

            scale_factor=scale_factor,

            working_width=working_size.width,
            working_height=working_size.height,
        )

        # ------------------------------------------
        # ImageData
        # ------------------------------------------

        patient.image_data = ImageData(
            image=image,
            mask=mask,

            working_image=working_image,
            working_mask=working_mask,

            metadata=metadata,
        )

        self.logger.info(
            "Successfully loaded %s",
            patient.patient_id,
        )

        return patient

    # --------------------------------------------------
    # Batch Interface
    # --------------------------------------------------

    def load_all(
        self,
        patients: list[Patient],
    ) -> list[Patient]:
        """
        Load image data for every patient.
        """

        loaded_patients = []

        for patient in patients:

            loaded_patients.append(
                self.load(patient)
            )

        return loaded_patients