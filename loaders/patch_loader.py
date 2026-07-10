"""
patch_loader.py

Coordinates the complete patch extraction stage.

The PatchLoader itself contains no image processing logic.
Instead, it orchestrates reusable modules responsible for

    • Patch candidate generation
    • Tissue filtering
    • Coordinate computation
    • Overlap handling
    • Patch extraction
    • Weak label generation
    • Patch metadata generation
    • Patch collection

The output is stored inside

    Patient.patch_dataset
"""

from typing import Optional

from configs.config import get_config

# -----------------------------
# Data Classes
# -----------------------------
from data.patient import Patient
from data.patch.patch_dataset import PatchDataset

# -----------------------------
# Patch Processing Modules
# -----------------------------
from methods.patch.patch_generator import PatchGenerator
from methods.patch.coordinate_generator import CoordinateGenerator
from methods.patch.overlap_handler import OverlapHandler
from methods.patch.tissue_patch_filter import TissuePatchFilter
from methods.patch.patch_extractor import PatchExtractor
from methods.patch.weak_label_generator import WeakLabelGenerator
from methods.patch.patch_metadata_generator import PatchMetadataGenerator
from methods.patch.patch_collection_builder import PatchCollectionBuilder


class PatchLoader:
    """
    Coordinates the complete patch extraction pipeline.

    The loader does not implement any image processing logic.
    It simply invokes reusable processing modules and stores
    the generated PatchDataset inside the Patient object.
    """

    def __init__(
        self,
        config: get_config(),
        logger=None,
        patch_size: int = 224,
        overlap: float = 0.50,
        tissue_threshold: float = 0.50,
    ):

        self.patch_size = patch_size
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold

    # ---------------------------------------------------------

    def load(self, patient: Patient) -> Patient:
        """
        Execute complete patch extraction pipeline.

        Parameters
        ----------
        patient : Patient

        Returns
        -------
        Patient
        """

        # -----------------------------------------
        # Retrieve previously computed information
        # -----------------------------------------

        image_data = patient.image_data
        tissue_mask = patient.tissue_mask

        # -----------------------------------------
        # Generate candidate coordinates
        # -----------------------------------------

        coordinate_generator = CoordinateGenerator(
            patch_size=self.patch_size,
            overlap=self.overlap,
        )

        candidate_coordinates = coordinate_generator.generate(
            image=image_data.working_image,
            tissue_regions=tissue_mask.regions,
        )

        # -----------------------------------------
        # Handle overlap strategy
        # -----------------------------------------

        overlap_handler = OverlapHandler(
            patch_size=self.patch_size,
            overlap=self.overlap,
        )

        coordinates = overlap_handler.process(
            candidate_coordinates
        )

        # -----------------------------------------
        # Generate candidate patches
        # -----------------------------------------

        patch_generator = PatchGenerator()

        candidate_patches = patch_generator.generate(
            # image=image_data.working_image,
            # mask=image_data.working_mask,
            coordinates=coordinates,
        )

        # -----------------------------------------
        # Remove patches with insufficient tissue
        # -----------------------------------------

        tissue_filter = TissuePatchFilter(
            threshold=self.tissue_threshold
        )

        valid_patches = tissue_filter.filter(
            candidate_patches,
            tissue_mask.mask,
        )

        # -----------------------------------------
        # Extract image and annotation patches
        # -----------------------------------------

        extractor = PatchExtractor()

        extracted_patches = extractor.extract(
            valid_patches,
            image=image_data.working_image,
            annotation=image_data.working_mask,
        )

        # -----------------------------------------
        # Generate weak labels
        # -----------------------------------------

        label_generator = WeakLabelGenerator()

        labeled_patches = label_generator.generate(
            extracted_patches
        )

        # -----------------------------------------
        # Generate metadata
        # -----------------------------------------

        metadata_generator = PatchMetadataGenerator()

        patches = metadata_generator.generate(
            labeled_patches
        )

        # -----------------------------------------
        # Build patch dataset
        # -----------------------------------------

        builder = PatchCollectionBuilder()

        patch_dataset: PatchDataset = builder.build(
            patient=patient,
            patches=patches,
        )

        # -----------------------------------------
        # Update patient
        # -----------------------------------------

        patient.patch_dataset = patch_dataset

        return patient
    

    def load_all(self, patients: list[Patient]) -> list[Patient]:
        """
        Execute complete patch extraction pipeline for a collection of patients.

        Parameters
        ----------
        patients : list[Patient]

        Returns
        -------
        list[Patient]
        """
        processed_patients = []
        
        for patient in patients:
            # You could add logging here if self.logger is implemented, e.g.:
            # if hasattr(self, 'logger') and self.logger:
            #     self.logger.info(f"Processing patches for patient: {patient.id}")
            
            processed_patient = self.load(patient)
            processed_patients.append(processed_patient)
            
        return processed_patients