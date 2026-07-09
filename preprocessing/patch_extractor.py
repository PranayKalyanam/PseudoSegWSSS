"""
==============================================================================
File        : patch_extractor.py

Description :
    Extracts overlapping image patches and corresponding mask patches from
    10X magnified Whole Slide Images (WSIs).

Pipeline
--------
10X RGB Image
        │
10X Mask
        │
Detected Tissue Regions
        │
        ▼
224 × 224 Sliding Window
        │
50% Overlap
        │
        ▼
Image Patch
Mask Patch
        │
        ▼
Patch Metadata
        │
        ▼
Save to Disk

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
from typing import Tuple
from typing import Any

import cv2
import numpy as np

from preprocessing.tissue_detection import TissueRegion
from utils.exceptions import ValidationError
from utils.logger import get_logger
from utils.validator import Validator

# =============================================================================
# Constants
# =============================================================================

PATCH_SIZE = 224

PATCH_OVERLAP = 0.50

PATCH_STRIDE = int(PATCH_SIZE * (1 - PATCH_OVERLAP))

MINIMUM_TISSUE_PERCENTAGE = 0.70

IMAGE_EXTENSION = ".png"

PATCH_FOLDER_NAME = "patches"

PATCH_MASK_FOLDER_NAME = "patchesMasks"

METADATA_FOLDER_NAME = "metadata"

# =============================================================================
# Enumerations
# =============================================================================


class PatchStatus(Enum):
    """
    Status of extracted patch.
    """

    ACCEPTED = "accepted"

    REJECTED = "rejected"


class PatchType(Enum):
    """
    Patch modality.
    """

    IMAGE = "image"

    MASK = "mask"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PatchExtractionConfig:
    """
    Patch extraction configuration.
    """

    patch_size: int = PATCH_SIZE

    overlap: float = PATCH_OVERLAP

    tissue_threshold: float = MINIMUM_TISSUE_PERCENTAGE

    save_image_patch: bool = True

    save_mask_patch: bool = True

    image_extension: str = IMAGE_EXTENSION

    output_directory: Optional[Path] = None

    debug: bool = False


# =============================================================================
# Coordinate
# =============================================================================


@dataclass
class PatchCoordinate:
    """
    Coordinate of extracted patch.
    """

    x: int

    y: int

    width: int

    height: int

    @property
    def bbox(self):

        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )


# =============================================================================
# Patch Labels
# =============================================================================


@dataclass
class PatchLabel:
    """
    Binary multi-label.

    Order

    [TumorType1,
     TumorType2,
     Cancer,
     Stroma]
    """

    tumor: int = 0

    stroma: int = 0

    lymphocyte: int = 0

    necrosis: int = 0

    def as_list(self):

        return [

            self.tumor,

            self.stroma,

            self.lymphocyte,

            self.necrosis,
        ]

    def as_string(self):

        return "".join(

            map(str, self.as_list())

        )


# =============================================================================
# Patch Metadata
# =============================================================================


@dataclass
class PatchMetadata:
    """
    Metadata associated with each extracted patch.
    """
    patch_id: int
    
    patient_id: str

    patch_name: str

    image_path: Optional[Path]

    mask_path: Optional[Path]

    coordinate: PatchCoordinate

    tissue_region_id: int

    tissue_percentage: float

    magnification: float

    label: PatchLabel

    additional_metadata: Dict[str, Any] = field(
        default_factory=dict
    )
    
    label_string: str = ""

    filename: str = ""

    generated: bool = False

    generator: str = ""
    
    detected_classes: List[int] = field(default_factory=list)

    class_statistics: Dict[int, Dict[str, float]] = field(
        default_factory=dict
    )


# =============================================================================
# Patch Object
# =============================================================================


@dataclass
class Patch:
    """
    Complete patch object.
    """

    image: np.ndarray

    mask: np.ndarray

    metadata: PatchMetadata

    status: PatchStatus = PatchStatus.ACCEPTED
    
    
# =============================================================================
# Base Patch Extractor
# =============================================================================

from abc import ABC, abstractmethod
import os
import shutil


class BasePatchExtractor(ABC):
    """
    Abstract base class for patch extraction.

    Responsibilities
    ----------------
    1. Validate inputs.
    2. Create output directories.
    3. Generate patch filenames.
    4. Save image and mask patches.
    5. Maintain patch metadata.

    Child classes implement the actual extraction strategy.
    """

    def __init__(
        self,
        config: PatchExtractionConfig | Any,
    ) -> None:

        if isinstance(config, PatchExtractionConfig):
            normalized_config = config
        else:
            normalized_config = PatchExtractionConfig(
                patch_size=getattr(config, "patch_size", PATCH_SIZE),
                overlap=getattr(config, "overlap", PATCH_OVERLAP),
                tissue_threshold=getattr(
                    config,
                    "tissue_threshold",
                    MINIMUM_TISSUE_PERCENTAGE,
                ),
                save_image_patch=(
                    getattr(config, "save_image_patch", None)
                    if hasattr(config, "save_image_patch")
                    else getattr(config, "save_patches", True)
                ),
                save_mask_patch=(
                    getattr(config, "save_mask_patch", None)
                    if hasattr(config, "save_mask_patch")
                    else getattr(config, "save_masks", True)
                ),
                image_extension=getattr(
                    config,
                    "image_extension",
                    IMAGE_EXTENSION,
                ),
                output_directory=(
                    getattr(config, "output_directory", None)
                    or getattr(config, "output_dir", None)
                ),
                debug=getattr(config, "debug", False),
            )

        Validator.validate_type(
            normalized_config,
            PatchExtractionConfig,
            "config",
        )

        self.config = normalized_config

        self.logger = get_logger(
            self.__class__.__name__
        )

        self.stride = int(
            self.config.patch_size *
            (1 - self.config.overlap)
        )

        if self.stride <= 0:

            raise ValidationError(
                "Invalid stride computed from overlap."
            )

        self.logger.info(
            "Patch size      : %d",
            self.config.patch_size,
        )

        self.logger.info(
            "Patch overlap   : %.2f",
            self.config.overlap,
        )

        self.logger.info(
            "Patch stride    : %d",
            self.stride,
        )

        self._initialize_output_directories()

    # -----------------------------------------------------------------
    # Directory Management
    # -----------------------------------------------------------------

    def _initialize_output_directories(self):

        if self.config.output_directory is None:

            raise ValidationError(
                "Output directory cannot be None."
            )

        self.output_directory = Path(
            self.config.output_directory
        )

        self.patch_directory = (
            self.output_directory /
            PATCH_FOLDER_NAME
        )

        self.mask_directory = (
            self.output_directory /
            PATCH_MASK_FOLDER_NAME
        )

        self.metadata_directory = (
            self.output_directory /
            METADATA_FOLDER_NAME
        )

        self.patch_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.mask_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------
    # Patient Directories
    # -----------------------------------------------------------------

    def create_patient_directories(
        self,
        patient_id: str,
    ):

        image_dir = (
            self.patch_directory /
            patient_id
        )

        mask_dir = (
            self.mask_directory /
            patient_id
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return image_dir, mask_dir

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate_patch_inputs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):

        if image is None:

            raise ValidationError(
                "Input image cannot be None."
            )

        if mask is None:

            raise ValidationError(
                "Input mask cannot be None."
            )

        if image.shape[:2] != mask.shape[:2]:

            raise ValidationError(
                "Image and mask dimensions do not match."
            )


    # -----------------------------------------------------------------
    # Filename Generation
    # -----------------------------------------------------------------

    def generate_patch_filename(
        self,
        patient_id: str,
        coordinate: PatchCoordinate,
        label: PatchLabel,
    ) -> str:
        """
        Generate filename.

        Example
        -------
        TCGA01_x000112_y000224_lbl1100.png
        """

        filename = (
            f"{patient_id}"
            f"_x{coordinate.x:06d}"
            f"_y{coordinate.y:06d}"
            f"_lbl{label.as_string()}"
            f"{self.config.image_extension}"
        )

        return filename

    # -----------------------------------------------------------------
    # Save Patch
    # -----------------------------------------------------------------

    def save_patch(
        self,
        image: np.ndarray,
        save_path: Path,
    ):

        success = cv2.imwrite(
            str(save_path),
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            ),
        )

        if not success:

            raise IOError(
                f"Unable to save {save_path}"
            )

    # -----------------------------------------------------------------
    # Save Mask
    # -----------------------------------------------------------------

    def save_mask(
        self,
        mask: np.ndarray,
        save_path: Path,
    ):

        success = cv2.imwrite(
            str(save_path),
            mask,
        )

        if not success:

            raise IOError(
                f"Unable to save {save_path}"
            )

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def clear_patient_directory(
        self,
        patient_id: str,
    ):

        image_dir = (
            self.patch_directory /
            patient_id
        )

        mask_dir = (
            self.mask_directory /
            patient_id
        )

        if image_dir.exists():

            shutil.rmtree(image_dir)

        if mask_dir.exists():

            shutil.rmtree(mask_dir)

    # -----------------------------------------------------------------
    # Abstract Interface
    # -----------------------------------------------------------------

    @abstractmethod
    def extract(
        self,
        patient_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        tissue_mask: np.ndarray,
        roi_origin: tuple[int, int],
        region: TissueRegion,
    ) -> List[Patch]:
        """
        Extract patches from a WSI.

        Returns
        -------
        List[Patch]
        """
        pass
    
# =============================================================================
# Sliding Window Coordinate Generator
# =============================================================================

    def generate_patch_coordinates(
        self,
        region: TissueRegion,
        image_shape: Tuple[int, int],
    ) -> List[PatchCoordinate]:
        """
        Generate candidate patch coordinates for a tissue region.

        Parameters
        ----------
        region : TissueRegion

        image_shape : Tuple[int, int]

        Returns
        -------
        List[PatchCoordinate]
        """

        image_height, image_width = image_shape

        coordinates: List[PatchCoordinate] = []

        patch_size = self.config.patch_size

        stride = self.stride

        #
        # Region boundaries
        #

        # start_x = region.x

        # start_y = region.y

        # end_x = region.x + region.width

        # end_y = region.y + region.height
        
        start_x = 0
        start_y = 0

        end_x = image_width - patch_size
        end_y = image_height - patch_size

        #
        # Sliding Window
        #

        for y in range(start_y, end_y + 1, stride):

            for x in range(start_x, end_x + 1, stride):

                coordinates.append(

                    PatchCoordinate(

                        x=x,

                        y=y,

                        width=patch_size,

                        height=patch_size,

                    )

                )

        self.logger.info(

            "Generated %d candidate patches for Region %d.",

            len(coordinates),

            region.region_id,

        )

        return coordinates

    # -----------------------------------------------------------------
    # Coordinate Validation
    # -----------------------------------------------------------------

    def validate_coordinate(
        self,
        coordinate: PatchCoordinate,
        image_shape: Tuple[int, int],
    ) -> bool:
        """
        Check whether a coordinate is completely inside the image.
        """

        image_height, image_width = image_shape

        if coordinate.x < 0:
            return False

        if coordinate.y < 0:
            return False

        if coordinate.x + coordinate.width > image_width:
            return False

        if coordinate.y + coordinate.height > image_height:
            return False

        return True

    # -----------------------------------------------------------------
    # Region Coordinate Generation
    # -----------------------------------------------------------------

    def generate_all_coordinates(
        self,
        tissue_regions: List[TissueRegion],
        image_shape: Tuple[int, int],
    ) -> Dict[int, List[PatchCoordinate]]:
        """
        Generate candidate coordinates for every tissue region.

        Returns
        -------
        Dict[
            region_id,
            List[PatchCoordinate]
        ]
        """

        coordinate_dict = {}

        total = 0

        for region in tissue_regions:

            coords = self.generate_patch_coordinates(

                region,

                image_shape,

            )

            coordinate_dict[region.region_id] = coords

            total += len(coords)

        self.logger.info(

            "Generated %d candidate coordinates.",

            total,

        )

        return coordinate_dict
    
# =============================================================================
# Patch Cropping
# =============================================================================

    def crop_image_patch(
        self,
        image: np.ndarray,
        coordinate: PatchCoordinate,
    ) -> np.ndarray:
        """
        Crop RGB patch.

        Parameters
        ----------
        image : np.ndarray

        coordinate : PatchCoordinate

        Returns
        -------
        np.ndarray
        """

        self.validate_coordinate(
            coordinate,
            image.shape[:2],
        )

        x = coordinate.x
        y = coordinate.y

        return image[
            y:y + coordinate.height,
            x:x + coordinate.width,
        ].copy()

    # -------------------------------------------------------------------------

    def crop_mask_patch(
        self,
        mask: np.ndarray,
        coordinate: PatchCoordinate,
    ) -> np.ndarray:
        """
        Crop mask patch.

        Parameters
        ----------
        mask : np.ndarray

        coordinate : PatchCoordinate

        Returns
        -------
        np.ndarray
        """

        self.validate_coordinate(
            coordinate,
            mask.shape[:2],
        )

        x = coordinate.x
        y = coordinate.y

        return mask[
            y:y + coordinate.height,
            x:x + coordinate.width,
        ].copy()

    # -------------------------------------------------------------------------

    def extract_patch_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        coordinate: PatchCoordinate,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract synchronized image and mask patches.

        Parameters
        ----------
        image

        mask

        coordinate

        Returns
        -------
        image_patch,
        mask_patch
        """

        image_patch = self.crop_image_patch(
            image,
            coordinate,
        )

        mask_patch = self.crop_mask_patch(
            mask,
            coordinate,
        )

        if image_patch.shape[:2] != (
            self.config.patch_size,
            self.config.patch_size,
        ):

            raise ValidationError(
                "Invalid RGB patch size."
            )

        if mask_patch.shape[:2] != (
            self.config.patch_size,
            self.config.patch_size,
        ):

            raise ValidationError(
                "Invalid mask patch size."
            )

        return image_patch, mask_patch

    # -------------------------------------------------------------------------

    def extract_region_patches(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        region: TissueRegion,
    ) -> List[
        Tuple[
            PatchCoordinate,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """
        Extract all candidate patches
        from one tissue region.

        Returns
        -------
        List[
            (
                coordinate,
                image_patch,
                mask_patch,
            )
        ]
        """

        coordinates = self.generate_patch_coordinates(
            region,
            image.shape[:2],
        )

        extracted = []

        for coordinate in coordinates:

            if not self.validate_coordinate(
                coordinate,
                image.shape[:2],
            ):
                continue

            image_patch, mask_patch = \
                self.extract_patch_pair(
                    image,
                    mask,
                    coordinate,
                )

            extracted.append(
                (
                    coordinate,
                    image_patch,
                    mask_patch,
                )
            )

        self.logger.info(
            "Extracted %d candidate patches "
            "from Region %d.",
            len(extracted),
            region.region_id,
        )

        return extracted

    # -------------------------------------------------------------------------

    def extract_all_region_patches(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        tissue_regions: List[TissueRegion],
    ) -> Dict[
        int,
        List[
            Tuple[
                PatchCoordinate,
                np.ndarray,
                np.ndarray,
            ]
        ]
    ]:
        """
        Extract candidate patches
        from every tissue region.

        Returns
        -------
        Dict[
            region_id,
            extracted patches
        ]
        """

        extracted = {}

        total = 0

        for region in tissue_regions:

            patches = self.extract_region_patches(
                image,
                mask,
                region,
            )

            extracted[
                region.region_id
            ] = patches

            total += len(patches)

        self.logger.info(
            "Total extracted candidate patches : %d",
            total,
        )

        return extracted
    
    
# =============================================================================
# Tissue Coverage Calculation
# =============================================================================

    def calculate_tissue_percentage(
        self,
        tissue_mask_patch: np.ndarray,
    ) -> float:
        """
        Calculate percentage of tissue inside one patch.

        Parameters
        ----------
        tissue_mask_patch : np.ndarray

        Returns
        -------
        float
        """

        if tissue_mask_patch is None:

            raise ValidationError(
                "Tissue mask patch is None."
            )

        if tissue_mask_patch.ndim != 2:

            raise ValidationError(
                "Expected binary tissue mask."
            )

        total_pixels = tissue_mask_patch.size

        tissue_pixels = np.count_nonzero(
            tissue_mask_patch
        )

        if total_pixels == 0:

            return 0.0

        return tissue_pixels / total_pixels


    # -------------------------------------------------------------------------

    def patch_passes_tissue_threshold(
        self,
        tissue_percentage: float,
    ) -> bool:
        """
        Determine whether a patch satisfies the
        configured tissue threshold.
        """

        return (
            tissue_percentage >=
            self.config.tissue_threshold
        )


    # -------------------------------------------------------------------------

    def filter_candidate_patches(
        self,
        candidate_patches,
        tissue_mask: np.ndarray,
    ):
        """
        Filter candidate patches using tissue coverage.

        Parameters
        ----------
        candidate_patches

        tissue_mask

        Returns
        -------
        accepted_patches
        """

        accepted = []

        rejected = 0

        for coordinate, image_patch, mask_patch in candidate_patches:

            #
            # Crop tissue mask using the
            # same coordinates.
            #

            tissue_patch = self.crop_mask_patch(
                tissue_mask,
                coordinate,
            )

            tissue_percentage = \
                self.calculate_tissue_percentage(
                    tissue_patch
                )

            if not self.patch_passes_tissue_threshold(
                tissue_percentage
            ):

                rejected += 1

                continue

            accepted.append(

                (
                    coordinate,
                    image_patch,
                    mask_patch,
                    tissue_patch,
                    tissue_percentage,
                )

            )

        self.logger.info(

            "Accepted %d patches | Rejected %d patches",

            len(accepted),

            rejected,

        )

        return accepted


    # -------------------------------------------------------------------------

    def filter_all_regions(
        self,
        extracted_regions,
        tissue_mask: np.ndarray,
    ):
        """
        Filter patches from every tissue region.

        Parameters
        ----------
        extracted_regions

        tissue_mask

        Returns
        -------
        Dictionary containing accepted patches.
        """

        accepted = {}

        total = 0

        for region_id, patches in extracted_regions.items():

            filtered = self.filter_candidate_patches(

                patches,

                tissue_mask,

            )

            accepted[region_id] = filtered

            total += len(filtered)

        self.logger.info(

            "Total accepted patches : %d",

            total,

        )

        return accepted
    
    # =============================================================================
# Patch Object Creation
# =============================================================================

    def create_patch(
        self,
        currPatch_id: int,
        patient_id: str,
        region: TissueRegion,
        coordinate: PatchCoordinate,
        image_patch: np.ndarray,
        mask_patch: np.ndarray,
        tissue_percentage: float,
    ) -> Patch:
        """
        Create a Patch object.

        Parameters
        ----------
        patch_id
        patient_id
        region
        coordinate
        image_patch
        mask_patch
        tissue_percentage

        Returns
        -------
        Patch
        """

        #
        # Empty label placeholder.
        # label_generator.py will populate this.
        #

        label = PatchLabel()

        filename = (
            f"{patient_id}"
            f"_x{coordinate.x:06d}"
            f"_y{coordinate.y:06d}"
            f"_p{currPatch_id:06d}"
            f"_lbl{label.as_string()}"
            f"{self.config.image_extension}"
        )

        metadata = PatchMetadata(

            patch_id=currPatch_id,

            patient_id=patient_id,

            patch_name=filename,

            image_path=None,

            mask_path=None,

            coordinate=coordinate,

            tissue_region_id=region.region_id,

            tissue_percentage=tissue_percentage,

            magnification=10.0,

            label=label,

        )

        return Patch(

            image=image_patch,

            mask=mask_patch,

            metadata=metadata,

            status=PatchStatus.ACCEPTED,

        )

    # ---------------------------------------------------------------------

    def save_patch_pair(
        self,
        patch: Patch,
    ) -> Patch:
        """
        Save RGB patch and mask patch.

        Returns
        -------
        Updated Patch object.
        """

        patient_id = patch.metadata.patient_id

        image_dir, mask_dir = \
            self.create_patient_directories(
                patient_id
            )

        image_path = (
            image_dir /
            patch.metadata.patch_name
        )

        mask_path = (
            mask_dir /
            patch.metadata.patch_name
        )

        #
        # Save RGB patch
        #

        if self.config.save_image_patch:

            self.save_patch(

                patch.image,

                image_path,

            )

        #
        # Save mask patch
        #

        if self.config.save_mask_patch:

            self.save_mask(

                patch.mask,

                mask_path,

            )

        patch.metadata.image_path = image_path

        patch.metadata.mask_path = mask_path

        return patch

    # ---------------------------------------------------------------------

    def create_patch_objects(
        self,
        patient_id: str,
        region: TissueRegion,
        accepted_patches,
        start_patch_id: int,
    ) -> Tuple[List[Patch], int]:
        """
        Convert accepted candidate patches into Patch objects.
        """

        patches = []

        patch_id = start_patch_id

        for (
            coordinate,
            image_patch,
            mask_patch,
            tissue_patch,
            tissue_percentage,
        ) in accepted_patches:

            patch = self.create_patch(

                patch_id,

                patient_id,

                region,

                coordinate,

                image_patch,

                mask_patch,

                tissue_percentage,

            )

            patch = self.save_patch_pair(
                patch
            )

            patches.append(
                patch
            )

            patch_id += 1

        self.logger.info(

            "Created %d patch objects.",

            len(patches),

        )

        return patches, patch_id
    
# =============================================================================
# Region Processing
# =============================================================================

    def process_region(
        self,
        patient_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        tissue_mask: np.ndarray,
        region: TissueRegion,
        roi_origin: tuple[int, int],
        start_patch_id: int,
    ) -> Tuple[List[Patch], int]:
        """
        Process one tissue region.

        Parameters
        ----------
        patient_id
        image
        mask
        tissue_mask
        region
        start_patch_id

        Returns
        -------
        patches,
        next_patch_id
        """

        #
        # Candidate extraction
        #

        candidate_patches = self.extract_region_patches(
            image=image,
            mask=mask,
            region=region,
        )

        #
        # Tissue filtering
        #

        accepted_patches = self.filter_candidate_patches(
            candidate_patches,
            tissue_mask,
        )
        
        
        origin_x, origin_y = roi_origin

        global_patches = []

        for (
                coordinate,
                image_patch,
                mask_patch,
                tissue_patch,
                tissue_percentage,
            ) in accepted_patches:

            global_coordinate = PatchCoordinate(
                    x=origin_x + coordinate.x,
                    y=origin_y + coordinate.y,
                    width=coordinate.width,
                    height=coordinate.height,
                )

            global_patches.append(
                    (
                        global_coordinate,
                        image_patch,
                        mask_patch,
                        tissue_patch,
                        tissue_percentage,
                    )
                )

        accepted_patches = global_patches

        #
        # Create Patch objects
        #

        patches, next_patch_id = self.create_patch_objects(
            patient_id=patient_id,
            region=region,
            accepted_patches=accepted_patches,
            start_patch_id=start_patch_id,
        )

        return patches, next_patch_id

    # -------------------------------------------------------------------------

    def extract(
        self,
        patient_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        tissue_mask: np.ndarray,
        roi_origin: tuple[int, int],
        region: TissueRegion,
    ) -> List[Patch]:
        """
        Extract patches from a complete WSI.

        Parameters
        ----------
        patient_id

        image

        mask

        tissue_mask

        tissue_regions

        Returns
        -------
        List[Patch]
        """

        self.logger.info(
            "Starting patch extraction for %s",
            patient_id,
        )

        #
        # Validate inputs
        #

        self.validate_patch_inputs(
            image,
            mask,
        )

        if tissue_mask is None:

            raise ValidationError(
                "Tissue mask cannot be None."
            )

        #
        # Storage
        #



        #
        # Region-wise processing
        #

        # for region in tissue_regions:

        #     self.logger.info(
        #         "Processing Region %d",
        #         region.region_id,
        #     )

        #     patches, patch_id = self.process_region(

        #         patient_id=patient_id,

        #         image=image,

        #         mask=mask,

        #         tissue_mask=tissue_mask,

        #         region=region,

        #         start_patch_id=patch_id,

        #     )

        #     extracted_patches.extend(
        #         patches
        #     )
        patches, _ = self.process_region(
                patient_id=patient_id,
                image=image,
                mask=mask,
                tissue_mask=tissue_mask,
                region=region,
                roi_origin=roi_origin,
                start_patch_id=0,
            )
        self.logger.info("Finished extraction.")
        self.logger.info("Total extracted patches : %d", len(patches))

        return patches