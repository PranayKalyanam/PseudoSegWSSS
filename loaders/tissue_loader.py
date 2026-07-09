"""
tissue_loader.py

Coordinates the tissue detection stage of the
preprocessing pipeline.

The loader performs no image processing itself.
Instead, it orchestrates the reusable tissue
processing modules.

Pipeline
--------
Patient
    ↓
ImageData
    ↓
OtsuDetector
    ↓
ConnectedComponents
    ↓
TissueFilter
    ↓
TissueStatisticsGenerator
    ↓
TissueMask
    ↓
Patient
"""

from __future__ import annotations

from data.patient import Patient
from data.tissue.tissue_mask import TissueMask

from methods.tissue.otsu_detector import OtsuDetector
from methods.tissue.connected_components import ConnectedComponents
from methods.tissue.tissue_filter import TissueFilter
from methods.tissue.detector_factory import DetectorFactory
from methods.tissue.tissue_statistics import (
    TissueStatisticsGenerator,
)

from utils.logger import get_logger


class TissueLoader:
    """
    Coordinates tissue detection.
    """

    def __init__(
        self,
        config,
        logger=None,
    ) -> None:

        self.config = config

        self.logger = logger or get_logger(
            "TissueLoader"
        )

    # --------------------------------------------------
    # Public Interface
    # --------------------------------------------------

    def load(
        self,
        patient: Patient,
    ) -> Patient:
        """
        Detect tissue for a patient.

        Parameters
        ----------
        patient
            Loaded patient.

        Returns
        -------
        Patient
        """

        if not patient.has_image:

            raise RuntimeError(
                "ImageData must be loaded before tissue detection."
            )

        self.logger.info(
            "Detecting tissue for %s",
            patient.patient_id,
        )

   

        # --------------------------------------------------
        # Tissue Detection
        # --------------------------------------------------

        detector = DetectorFactory.create(
            self.config.tissue_detector
        )

        binary_mask = detector.detect(
            patient.image_data.working_image
        )

        # --------------------------------------------------
        # Connected Components
        # --------------------------------------------------

        regions = ConnectedComponents.extract(
            binary_mask
        )

        # --------------------------------------------------
        # Region Filtering
        # --------------------------------------------------

        regions = TissueFilter.filter(
            regions=regions,
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        statistics = (
            TissueStatisticsGenerator.compute(
                mask=binary_mask,
                regions=regions,
            )
        )

        # --------------------------------------------------
        # Create TissueMask
        # --------------------------------------------------

        patient.tissue_mask = TissueMask(

            mask=binary_mask,

            regions=regions,

            statistics=statistics,

            # detector_name=detector,
            detector_name=detector.__class__.__name__,

            threshold=self.config.tissue_threshold,
        )

        self.logger.info(
            "Detected %d tissue regions",
            patient.number_of_tissue_regions,
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
        Detect tissue for every patient.
        """

        processed = []

        for patient in patients:

            processed.append(
                self.load(patient)
            )

        return processed