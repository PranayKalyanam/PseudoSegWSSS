"""
==============================================================================
File        : label_generator.py

Description :
    Generates weak multi-label vectors from extracted BCSS annotation
    mask patches. The generated labels are attached to Patch objects
    produced by patch_extractor.py.

Pipeline
--------
Patch Object
        │
        ▼
Read Mask Patch
        │
        ▼
Semantic Class Detection
        │
        ▼
Binary Multi-label Generation
        │
        ▼
Update Patch Metadata
        │
        ▼
Generate Final Filename
        │
        ▼
Return Updated Patch Object

Author : Your Name
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Any

import cv2
import numpy as np

from preprocessing.patch_extractor import (
    Patch,
    PatchLabel,
    PatchMetadata,
)

from utils.exceptions import ValidationError
from utils.logger import get_logger
from utils.validator import Validator

# =============================================================================
# Constants
# =============================================================================

BACKGROUND_CLASS = 0

DEFAULT_LABEL_ORDER = (
    "tumor",
    "stroma",
    "lymphocyte",
    "necrosis",
)

# =============================================================================
# BCSS Semantic Classes
# =============================================================================


class BCSSClass(Enum):
    """
    BCSS semantic classes used for weak supervision.

    Background is intentionally excluded from the generated
    weak labels.
    """

    BACKGROUND = 0

    TUMOR = 1

    STROMA = 2

    LYMPHOCYTE = 3

    NECROSIS = 4


# =============================================================================
# Mask Format
# =============================================================================


class MaskFormat(Enum):
    """
    Annotation mask representation.
    """

    INDEXED = "indexed"

    RGB = "rgb"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LabelGeneratorConfig:
    """
    Configuration for weak label generation.
    """

    #
    # BCSS class mapping
    #

    class_mapping: Dict[int, str] = field(
        default_factory=lambda: {
            1: "tumor",
            2: "stroma",
            3: "lymphocyte",
            4: "necrosis",
        }
    )

    #
    # Label order
    #

    label_order: Tuple[str, ...] = DEFAULT_LABEL_ORDER

    #
    # Ignore background
    #

    background_class: int = BACKGROUND_CLASS

    #
    # Automatically detect RGB/indexed masks
    #

    auto_detect_mask_format: bool = True

    #
    # Optional RGB color lookup.
    # Used only when masks are RGB encoded.
    #

    rgb_mapping: Optional[
        Dict[
            Tuple[int, int, int],
            int,
        ]
    ] = None

    debug: bool = False


# =============================================================================
# Label Generation Result
# =============================================================================


@dataclass
class LabelGenerationResult:
    """
    Output generated for a single patch.
    """

    patch: Patch

    detected_classes: Set[int]

    label: PatchLabel

    label_string: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )
    
    
# =============================================================================
# Base Label Generator
# =============================================================================

from abc import ABC, abstractmethod


class BaseLabelGenerator(ABC):
    """
    Abstract base class for weak label generation.

    Responsibilities
    ----------------
    1. Validate configuration.
    2. Detect mask format.
    3. Parse semantic classes.
    4. Generate binary weak labels.
    5. Update Patch metadata.

    Child classes implement the actual label generation logic.
    """

    def __init__(
        self,
        config: LabelGeneratorConfig,
    ) -> None:

        Validator.validate_type(
            config,
            LabelGeneratorConfig,
            "config",
        )

        self.config = config

        self.logger = get_logger(
            self.__class__.__name__
        )

        self._validate_configuration()

    # -------------------------------------------------------------------------
    # Configuration Validation
    # -------------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """
        Validate configuration parameters.
        """

        if len(self.config.class_mapping) == 0:

            raise ValidationError(
                "Class mapping cannot be empty."
            )

        if len(self.config.label_order) == 0:

            raise ValidationError(
                "Label order cannot be empty."
            )

        for label_name in self.config.label_order:

            if label_name not in self.config.class_mapping.values():

                raise ValidationError(
                    f"'{label_name}' is not present "
                    f"in class_mapping."
                )

        self.logger.info(
            "Loaded %d semantic classes.",
            len(self.config.class_mapping),
        )

    # -------------------------------------------------------------------------
    # Mask Validation
    # -------------------------------------------------------------------------

    def validate_mask(
        self,
        mask: np.ndarray,
    ) -> None:
        """
        Validate annotation mask.
        """

        if mask is None:

            raise ValidationError(
                "Mask cannot be None."
            )

        if not isinstance(mask, np.ndarray):

            raise ValidationError(
                "Mask must be a NumPy array."
            )

        if mask.size == 0:

            raise ValidationError(
                "Mask is empty."
            )

    # -------------------------------------------------------------------------
    # Patch Validation
    # -------------------------------------------------------------------------

    def validate_patch(
        self,
        patch: Patch,
    ) -> None:
        """
        Validate Patch object.
        """

        if patch is None:

            raise ValidationError(
                "Patch cannot be None."
            )

        if patch.mask is None:

            raise ValidationError(
                "Patch mask is missing."
            )

    # -------------------------------------------------------------------------
    # Mask Format Detection
    # -------------------------------------------------------------------------

    def detect_mask_format(
        self,
        mask: np.ndarray,
    ) -> MaskFormat:
        """
        Automatically determine mask format.

        Returns
        -------
        MaskFormat
        """

        self.validate_mask(mask)

        #
        # RGB mask
        #

        if mask.ndim == 3:

            if mask.shape[2] == 3:

                return MaskFormat.RGB

        #
        # Indexed mask
        #

        if mask.ndim == 2:

            return MaskFormat.INDEXED

        raise ValidationError(
            "Unsupported mask format."
        )

    # -------------------------------------------------------------------------
    # RGB → Indexed Conversion
    # -------------------------------------------------------------------------

    def rgb_to_indexed(
        self,
        rgb_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Convert RGB annotation mask into indexed mask.

        Requires rgb_mapping to be defined.
        """

        if self.config.rgb_mapping is None:

            raise ValidationError(
                "RGB mapping has not been configured."
            )

        indexed = np.full(
            rgb_mask.shape[:2],
            self.config.background_class,
            dtype=np.uint8,
        )

        for color, class_id in self.config.rgb_mapping.items():

            matches = np.all(
                rgb_mask == color,
                axis=-1,
            )

            indexed[matches] = class_id

        return indexed

    # -------------------------------------------------------------------------
    # Normalize Mask
    # -------------------------------------------------------------------------

    def normalize_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Return indexed semantic mask regardless
        of original representation.
        """

        fmt = self.detect_mask_format(
            mask
        )

        if fmt == MaskFormat.INDEXED:

            return mask

        return self.rgb_to_indexed(mask)

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def class_name(
        self,
        class_id: int,
    ) -> Optional[str]:
        """
        Return semantic class name.
        """

        return self.config.class_mapping.get(
            class_id,
            None,
        )

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    @abstractmethod
    def generate_labels(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Generate weak labels for extracted patches.
        """
        pass
    
# =============================================================================
# Semantic Mask Parsing
# =============================================================================

    def extract_semantic_classes(
        self,
        mask: np.ndarray,
    ) -> Set[int]:
        """
        Extract semantic classes present in the annotation mask.

        Background is ignored.

        Parameters
        ----------
        mask : np.ndarray

        Returns
        -------
        Set[int]
        """

        indexed_mask = self.normalize_mask(
            mask
        )

        unique_classes = np.unique(
            indexed_mask
        )

        detected_classes: Set[int] = set()

        valid_classes = set(
            self.config.class_mapping.keys()
        )

        for class_id in unique_classes:

            class_id = int(class_id)

            #
            # Ignore background
            #

            if class_id == self.config.background_class:
                continue

            #
            # Ignore unknown classes
            #

            if class_id not in valid_classes:

                self.logger.warning(

                    "Unknown class ID detected: %d",

                    class_id,

                )

                continue

            detected_classes.add(
                class_id
            )

        return detected_classes

    # -------------------------------------------------------------------------
    # Pixel Statistics
    # -------------------------------------------------------------------------

    def class_pixel_statistics(
        self,
        mask: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute pixel statistics for each semantic class.

        Parameters
        ----------
        mask : np.ndarray

        Returns
        -------
        Dict
        """

        indexed_mask = self.normalize_mask(
            mask
        )

        total_pixels = indexed_mask.size

        statistics = {}

        for class_id in np.unique(indexed_mask):

            class_id = int(class_id)

            if class_id == self.config.background_class:
                continue

            if class_id not in self.config.class_mapping:
                continue

            pixel_count = int(
                np.sum(indexed_mask == class_id)
            )

            statistics[class_id] = {

                "class_name":
                    self.class_name(class_id),

                "pixel_count":
                    pixel_count,

                "pixel_percentage":
                    pixel_count / total_pixels,

            }

        return statistics

    # -------------------------------------------------------------------------
    # Parse Patch
    # -------------------------------------------------------------------------

    def parse_patch(
        self,
        patch: Patch,
    ) -> Tuple[
        Set[int],
        Dict[int, Dict[str, float]]
    ]:
        """
        Parse one mask patch.

        Parameters
        ----------
        patch : Patch

        Returns
        -------
        detected_classes,
        statistics
        """

        self.validate_patch(
            patch
        )

        detected_classes = self.extract_semantic_classes(
            patch.mask
        )

        statistics = self.class_pixel_statistics(
            patch.mask
        )

        self.logger.info(

            "Patch %s contains %d semantic classes.",

            patch.metadata.patch_name,

            len(detected_classes),

        )

        return (

            detected_classes,

            statistics,

        )

    # -------------------------------------------------------------------------
    # Parse Multiple Patches
    # -------------------------------------------------------------------------

    def parse_patches(
        self,
        patches: List[Patch],
    ) -> List[
        Tuple[
            Patch,
            Set[int],
            Dict[int, Dict[str, float]]
        ]
    ]:
        """
        Parse semantic classes from all patches.

        Returns
        -------
        List[
            (
                patch,
                detected_classes,
                statistics
            )
        ]
        """

        parsed = []

        for patch in patches:

            classes, statistics = self.parse_patch(
                patch
            )

            parsed.append(

                (

                    patch,

                    classes,

                    statistics,

                )

            )

        self.logger.info(

            "Parsed %d patches.",

            len(parsed),

        )

        return parsed
    
# =============================================================================
# Weak Multi-label Generation
# =============================================================================

    def generate_binary_label(
        self,
        detected_classes: Set[int],
    ) -> PatchLabel:
        """
        Generate weak binary multi-label vector.

        Label Order
        -----------
        [
            Tumor,
            Stroma,
            Lymphocyte,
            Necrosis
        ]

        Parameters
        ----------
        detected_classes

        Returns
        -------
        PatchLabel
        """

        #
        # Initialize all labels to zero.
        #

        label = PatchLabel()

        #
        # Iterate according to configured label order.
        #

        for index, class_name in enumerate(
            self.config.label_order
        ):

            #
            # Find corresponding class ID.
            #

            class_id = None

            for cid, cname in self.config.class_mapping.items():

                if cname == class_name:

                    class_id = cid

                    break

            if class_id is None:

                continue

            if class_id in detected_classes:

                #
                # Update PatchLabel.
                #

                if index == 0:

                    label.tumor = 1

                elif index == 1:

                    label.stroma = 1

                elif index == 2:

                    label.lymphocyte = 1

                elif index == 3:

                    label.necrosis = 1

        return label

    # -------------------------------------------------------------------------

    def update_patch_label(
        self,
        patch: Patch,
        detected_classes: Set[int],
        statistics: Dict[int, Dict[str, float]],
    ) -> Patch:
        """
        Update Patch object with weak label.

        Parameters
        ----------
        patch

        detected_classes

        statistics

        Returns
        -------
        Updated Patch
        """

        self.validate_patch(
            patch
        )

        label = self.generate_binary_label(
            detected_classes
        )

        #
        # Update Patch label.
        #

        patch.metadata.label = label
        

        #
        # Save semantic information.
        #

        patch.metadata.detected_classes = sorted(
            list(detected_classes)
        )

        patch.metadata.class_statistics = statistics

        self.logger.info(

            "Patch %s -> Label %s",

            patch.metadata.patch_name,

            label.as_string(),

        )

        return patch

    # -------------------------------------------------------------------------

    def generate_patch_label(
        self,
        patch: Patch,
    ) -> Patch:
        """
        Generate weak label for one Patch.

        Parameters
        ----------
        patch

        Returns
        -------
        Patch
        """

        detected_classes, statistics = self.parse_patch(
            patch
        )

        patch = self.update_patch_label(

            patch,

            detected_classes,

            statistics,

        )

        return patch

    # -------------------------------------------------------------------------

    def generate_patch_labels(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Generate weak labels for every patch.

        Parameters
        ----------
        patches

        Returns
        -------
        List[Patch]
        """

        labelled_patches = []

        for patch in patches:

            labelled_patch = self.generate_patch_label(
                patch
            )

            labelled_patches.append(
                labelled_patch
            )

        self.logger.info(

            "Generated labels for %d patches.",

            len(labelled_patches),

        )

        return labelled_patches
    
# =============================================================================
# Filename Generation and Metadata Update
# =============================================================================

    def generate_patch_filename(
        self,
        patch: Patch,
    ) -> str:
        """
        Generate final filename using the generated
        weak multi-label.

        Format
        ------
        PatientID_Rxxx_Pxxxxxx_Xxxxx_Yyyyy_10X_L1100.png
        """

        metadata = patch.metadata

        label_string = metadata.label.as_string()

        filename = (

            f"{metadata.patient_id}"

            f"_R{metadata.tissue_region_id:03d}"

            f"_P{metadata.patch_id:06d}"

            f"_X{metadata.coordinate.x:06d}"

            f"_Y{metadata.coordinate.y:06d}"

            f"_{int(metadata.magnification)}X"

            f"_L{label_string}"

            f".png"

        )

        return filename

    # ---------------------------------------------------------------------

    def update_patch_filename(
        self,
        patch: Patch,
    ) -> Patch:
        """
        Update Patch filename after label generation.
        """

        filename = self.generate_patch_filename(
            patch
        )

        patch.metadata.patch_name = filename

        return patch

    # ---------------------------------------------------------------------

    def update_patch_metadata(
        self,
        patch: Patch,
    ) -> Patch:
        """
        Update metadata after weak-label generation.
        """

        metadata = patch.metadata

        metadata.label_string = metadata.label.as_string()

        metadata.filename = metadata.patch_name

        metadata.generated = True

        metadata.generator = self.__class__.__name__

        return patch

    # ---------------------------------------------------------------------

    def finalize_patch(
        self,
        patch: Patch,
    ) -> Patch:
        """
        Finalize patch metadata before saving.
        """

        patch = self.update_patch_filename(
            patch
        )

        patch = self.update_patch_metadata(
            patch
        )

        self.logger.info(

            "Finalized Patch %06d",

            patch.metadata.patch_id,

        )

        return patch

    # ---------------------------------------------------------------------

    def finalize_patches(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Finalize all generated patches.
        """

        finalized = []

        for patch in patches:

            finalized.append(

                self.finalize_patch(
                    patch
                )

            )

        self.logger.info(

            "Finalized %d patches.",

            len(finalized),

        )

        return finalized
    
    # =============================================================================
# Patch Finalization
# =============================================================================

    def prepare_patch_record(
        self,
        patch: Patch,
    ) -> Dict[str, Any]:
        """
        Prepare a serializable record describing a patch.

        This record will later be written to CSV/JSON by the
        preprocessing pipeline.
        """

        metadata = patch.metadata

        record = {

            #
            # Identification
            #

            "patient_id":
                metadata.patient_id,

            "patch_id":
                metadata.patch_id,

            "region_id":
                metadata.tissue_region_id,

            #
            # Coordinates
            #

            "x":
                metadata.coordinate.x,

            "y":
                metadata.coordinate.y,

            "width":
                metadata.coordinate.width,

            "height":
                metadata.coordinate.height,

            #
            # Magnification
            #

            "magnification":
                metadata.magnification,

            #
            # Tissue
            #

            "tissue_percentage":
                metadata.tissue_percentage,

            #
            # Weak labels
            #

            "tumor":
                metadata.label.tumor,

            "stroma":
                metadata.label.stroma,

            "lymphocyte":
                metadata.label.lymphocyte,

            "necrosis":
                metadata.label.necrosis,

            "label_string":
                metadata.label.as_string(),

            #
            # Files
            #

            "filename":
                metadata.patch_name,

            #
            # Semantic information
            #

            "detected_classes":
                metadata.detected_classes,

            "statistics":
                metadata.class_statistics,

        }

        return record

    # -------------------------------------------------------------------------

    def prepare_dataset_manifest(
        self,
        patches: List[Patch],
    ) -> List[Dict[str, Any]]:
        """
        Prepare metadata records for all patches.
        """

        manifest = []

        for patch in patches:

            manifest.append(

                self.prepare_patch_record(
                    patch
                )

            )

        self.logger.info(

            "Prepared dataset manifest containing %d patches.",

            len(manifest),

        )

        return manifest

    # -------------------------------------------------------------------------

    def summarize_labels(
        self,
        patches: List[Patch],
    ) -> Dict[str, int]:
        """
        Compute dataset statistics.
        """

        summary = {

            "tumor": 0,

            "stroma": 0,

            "lymphocyte": 0,

            "necrosis": 0,

        }

        for patch in patches:

            summary["tumor"] += patch.metadata.label.tumor

            summary["stroma"] += patch.metadata.label.stroma

            summary["lymphocyte"] += patch.metadata.label.lymphocyte

            summary["necrosis"] += patch.metadata.label.necrosis

        self.logger.info(

            "Dataset Summary : %s",

            summary,

        )

        return summary
    
    
class BCSSLabelGenerator(BaseLabelGenerator):
    """
    Concrete implementation of the BCSS weak label generator.
    """

    def __init__(
        self,
        config: LabelGeneratorConfig,
    ) -> None:

        super().__init__(config)

    def generate_labels(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """
        Generate weak labels for all extracted patches.

        Workflow
        --------
        1. Parse semantic classes.
        2. Generate binary multi-label.
        3. Generate filename.
        4. Update metadata.
        5. Return finalized patches.
        """

        labelled = self.generate_patch_labels(
            patches
        )

        finalized = self.finalize_patches(
            labelled
        )

        self.logger.info(
            "Generated labels for %d patches.",
            len(finalized),
        )

        return finalized
    """
    Concrete implementation for the BCSS dataset.
    """
