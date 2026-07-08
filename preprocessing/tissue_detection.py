"""
===============================================================================
File        : tissue_detection.py
Author      : Your Name
Project     : Graph-based Weakly Supervised Histopathology Framework
Description :
    Tissue detection module responsible for identifying valid tissue regions
    from histopathology whole-slide images (WSIs).

    This module provides an extensible framework for tissue detection using
    multiple algorithms. Initially, Otsu thresholding is implemented, while the
    architecture allows easy integration of future algorithms such as the
    Double Pass Tissue Detection method.

Pipeline
--------
Image Loader
        │
        ▼
Magnification (10x)
        │
        ▼
Tissue Detection
        │
        ▼
Patch Extraction

Supported Algorithms
--------------------
1. Otsu Thresholding
2. Double Pass (Future Implementation)

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Any

from abc import ABC

import cv2
import numpy as np

from utils.logger import get_logger
from utils.validator import Validator
from utils.exceptions import ValidationError

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_TISSUE_THRESHOLD = 0.70
DEFAULT_MIN_REGION_AREA = 1000
DEFAULT_KERNEL_SIZE = 5
DEFAULT_BLUR_KERNEL = (5, 5)

DEFAULT_OPEN_ITERATIONS = 2
DEFAULT_CLOSE_ITERATIONS = 2

MIN_BINARY_VALUE = 0
MAX_BINARY_VALUE = 255

SUPPORTED_ALGORITHMS = (
    "otsu",
    "double_pass",
)

# =============================================================================
# Enumerations
# =============================================================================


class TissueDetectionAlgorithm(Enum):
    """
    Available tissue detection algorithms.
    """

    OTSU = "otsu"

    DOUBLE_PASS = "double_pass"


class TissueRegionType(Enum):
    """
    Tissue region category.
    """

    VALID = "valid"

    SMALL = "small"

    BACKGROUND = "background"

    UNKNOWN = "unknown"


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class TissueDetectionConfig:
    """
    Configuration parameters for tissue detection.
    """

    algorithm: TissueDetectionAlgorithm = TissueDetectionAlgorithm.OTSU

    tissue_threshold: float = DEFAULT_TISSUE_THRESHOLD

    minimum_region_area: int = DEFAULT_MIN_REGION_AREA

    gaussian_kernel: Tuple[int, int] = DEFAULT_BLUR_KERNEL

    morphology_kernel_size: int = DEFAULT_KERNEL_SIZE

    open_iterations: int = DEFAULT_OPEN_ITERATIONS

    close_iterations: int = DEFAULT_CLOSE_ITERATIONS

    debug: bool = False

    save_intermediate_results: bool = False

    output_directory: Optional[Path] = None


# =============================================================================
# Region Dataclass
# =============================================================================


@dataclass
class TissueRegion:
    """
    Represents one connected tissue region.
    """

    region_id: int

    x: int

    y: int

    width: int

    height: int

    area: float

    tissue_percentage: float

    contour: Optional[np.ndarray] = None

    region_type: TissueRegionType = TissueRegionType.UNKNOWN

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Return bounding box coordinates.
        """
        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )

    @property
    def center(self) -> Tuple[int, int]:
        """
        Return center coordinates.
        """
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )


# =============================================================================
# Detection Result Dataclass
# =============================================================================


@dataclass
class TissueDetectionResult:
    """
    Complete output of tissue detection.
    """

    patient_id: str

    tissue_mask: np.ndarray

    refined_mask: np.ndarray

    tissue_regions: List[TissueRegion]

    tissue_percentage: float

    processing_time: float

    image_height: int

    image_width: int

    algorithm: str

    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Morphology Parameters
# =============================================================================


@dataclass
class MorphologyParameters:
    """
    Parameters controlling morphological operations.
    """

    kernel_size: int = DEFAULT_KERNEL_SIZE

    open_iterations: int = DEFAULT_OPEN_ITERATIONS

    close_iterations: int = DEFAULT_CLOSE_ITERATIONS

    kernel_shape: int = cv2.MORPH_ELLIPSE

    @property
    def kernel(self) -> np.ndarray:
        """
        Create morphology kernel.
        """
        return cv2.getStructuringElement(
            self.kernel_shape,
            (self.kernel_size, self.kernel_size),
        )


# =============================================================================
# Utility Functions
# =============================================================================


def validate_tissue_threshold(threshold: float) -> None:
    """
    Validate tissue percentage threshold.

    Parameters
    ----------
    threshold : float

    Raises
    ------
    ValidationError
    """

    if not isinstance(threshold, (float, int)):
        raise ValidationError(
            "Tissue threshold must be numeric."
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError(
            "Tissue threshold must lie in [0, 1]."
        )


def validate_binary_mask(mask: np.ndarray) -> None:
    """
    Validate binary mask.

    Parameters
    ----------
    mask : np.ndarray
    """

    if mask is None:
        raise ValidationError(
            "Mask cannot be None."
        )

    if not isinstance(mask, np.ndarray):
        raise ValidationError(
            "Mask must be NumPy array."
        )

    if mask.ndim != 2:
        raise ValidationError(
            "Binary mask must be 2-dimensional."
        )


def create_logger():
    """
    Create module logger.
    """

    return get_logger("TissueDetection")



# =============================================================================
# Base Tissue Detector
# =============================================================================

class BaseTissueDetector(ABC):
    """
    Abstract base class for all tissue detection algorithms.

    Every tissue detector must inherit from this class and implement
    the detect() method.

    Responsibilities
    ----------------
    1. Validate input image.
    2. Perform preprocessing.
    3. Generate binary tissue mask.
    4. Refine the tissue mask.
    5. Extract tissue regions.
    6. Return TissueDetectionResult.
    """

    def __init__(
        self,
        config: TissueDetectionConfig | Any,
    ) -> None:

        if isinstance(config, TissueDetectionConfig):
            normalized_config = config
        elif hasattr(config, "tissue_threshold"):
            normalized_config = TissueDetectionConfig(
                tissue_threshold=getattr(
                    config,
                    "tissue_threshold",
                    DEFAULT_TISSUE_THRESHOLD,
                ),
                minimum_region_area=getattr(
                    config,
                    "minimum_region_area",
                    DEFAULT_MIN_REGION_AREA,
                ),
                gaussian_kernel=getattr(
                    config,
                    "gaussian_kernel",
                    DEFAULT_BLUR_KERNEL,
                ),
                morphology_kernel_size=getattr(
                    config,
                    "morphology_kernel_size",
                    DEFAULT_KERNEL_SIZE,
                ),
                open_iterations=getattr(
                    config,
                    "open_iterations",
                    DEFAULT_OPEN_ITERATIONS,
                ),
                close_iterations=getattr(
                    config,
                    "close_iterations",
                    DEFAULT_CLOSE_ITERATIONS,
                ),
                debug=getattr(config, "debug", False),
                save_intermediate_results=getattr(
                    config,
                    "save_intermediate_results",
                    False,
                ),
                output_directory=getattr(
                    config,
                    "output_directory",
                    None,
                ),
            )
        else:
            raise ValidationError(
                "Tissue detector config must provide tissue_threshold."
            )

        validate_tissue_threshold(
            normalized_config.tissue_threshold,
        )

        self.config = normalized_config

        self.logger = create_logger()

        self.logger.info(
            "Initialized %s",
            self.__class__.__name__,
        )

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def validate_image(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate input RGB image.

        Parameters
        ----------
        image : np.ndarray
        """

        if image is None:
            raise ValidationError(
                "Input image is None."
            )

        if not isinstance(image, np.ndarray):
            raise ValidationError(
                "Input must be NumPy array."
            )

        if image.ndim != 3:
            raise ValidationError(
                "Expected RGB image."
            )

        if image.shape[2] != 3:
            raise ValidationError(
                "Expected three image channels."
            )

    # ---------------------------------------------------------------------
    # Image preprocessing
    # ---------------------------------------------------------------------

    def rgb_to_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert RGB image to grayscale.
        """

        self.validate_image(image)

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    def gaussian_blur(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Gaussian smoothing before thresholding.
        """

        return cv2.GaussianBlur(
            image,
            self.config.gaussian_kernel,
            0,
        )

    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Common preprocessing pipeline.

        RGB
            ↓
        Gray
            ↓
        Gaussian Blur
        """

        gray = self.rgb_to_gray(image)

        gray = self.gaussian_blur(gray)

        return gray

    # ---------------------------------------------------------------------
    # Morphology kernel
    # ---------------------------------------------------------------------

    @property
    def morphology_kernel(self) -> np.ndarray:
        """
        Create morphology kernel.
        """

        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.config.morphology_kernel_size,
                self.config.morphology_kernel_size,
            ),
        )

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    @staticmethod
    def image_area(
        image: np.ndarray,
    ) -> int:
        """
        Compute image area.
        """

        return image.shape[0] * image.shape[1]

    @staticmethod
    def mask_area(
        mask: np.ndarray,
    ) -> int:
        """
        Number of foreground pixels.
        """

        validate_binary_mask(mask)

        return int(np.count_nonzero(mask))

    @staticmethod
    def tissue_percentage(
        mask: np.ndarray,
    ) -> float:
        """
        Percentage of tissue in binary mask.
        """

        validate_binary_mask(mask)

        total = mask.shape[0] * mask.shape[1]

        tissue = np.count_nonzero(mask)

        return float(tissue) / float(total)

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------

    def log_statistics(
        self,
        percentage: float,
    ) -> None:
        """
        Log tissue statistics.
        """

        self.logger.info(
            "Detected %.2f%% tissue",
            percentage * 100.0,
        )

    # ---------------------------------------------------------------------
    # Abstract Interface
    # ---------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        patient_id: str = "",
    ) -> TissueDetectionResult:
        """
        Detect tissue regions.

        Must be implemented by every algorithm.

        Returns
        -------
        TissueDetectionResult
        """

        raise NotImplementedError(
            "Subclasses must implement detect()."
        )
        

# =============================================================================
# Otsu Tissue Detector
# =============================================================================

import time


class OtsuTissueDetector(BaseTissueDetector):
    """
    Tissue detector based on Otsu thresholding.

    Pipeline
    --------
    RGB Image
            ↓
    Gray Image
            ↓
    Gaussian Blur
            ↓
    Otsu Threshold
            ↓
    Binary Tissue Mask
            ↓
    Morphological Refinement (Section 2B)
            ↓
    Tissue Region Extraction (Section 3B)

    Notes
    -----
    This implementation assumes that tissue appears darker than
    the white slide background. Therefore, the Otsu threshold is
    inverted to produce a foreground tissue mask.
    """

    def __init__(
        self,
        config: TissueDetectionConfig,
    ) -> None:

        super().__init__(config)

        self.logger.info(
            "Using Otsu Tissue Detector."
        )

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def otsu_threshold(
        self,
        gray_image: np.ndarray,
    ) -> np.ndarray:
        """
        Compute binary tissue mask using Otsu thresholding.

        Parameters
        ----------
        gray_image : np.ndarray

        Returns
        -------
        np.ndarray
            Binary tissue mask.
        """

        if gray_image.ndim != 2:

            raise ValidationError(
                "Otsu threshold expects grayscale image."
            )

        _, binary = cv2.threshold(
            gray_image,
            0,
            MAX_BINARY_VALUE,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        return binary

    # ------------------------------------------------------------------
    # Automatic inversion verification
    # ------------------------------------------------------------------

    def verify_mask(
        self,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Ensure tissue occupies the minority of pixels.

        Occasionally Otsu produces an inverted mask depending
        on image contrast. If foreground exceeds 90% of the
        image, invert the mask.
        """

        validate_binary_mask(binary_mask)

        ratio = np.count_nonzero(binary_mask) / binary_mask.size

        if ratio > 0.90:

            self.logger.warning(
                "Detected inverted tissue mask. Automatically correcting."
            )

            binary_mask = cv2.bitwise_not(binary_mask)

        return binary_mask

    # ------------------------------------------------------------------
    # Binary mask generation
    # ------------------------------------------------------------------

    def generate_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Generate initial tissue mask.

        Parameters
        ----------
        image

        Returns
        -------
        np.ndarray
        """

        gray = self.preprocess(image)

        binary = self.otsu_threshold(gray)

        binary = self.verify_mask(binary)

        return binary

    # ------------------------------------------------------------------
    # Main detection function
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        patient_id: str = "",
    ) -> TissueDetectionResult:
        """
        Perform tissue detection.

        Parameters
        ----------
        image : np.ndarray

        patient_id : str

        Returns
        -------
        TissueDetectionResult
        """

        self.validate_image(image)

        start_time = time.perf_counter()

        self.logger.info(
            "Running tissue detection for %s",
            patient_id if patient_id else "Unknown Patient",
        )

        #
        # Initial binary mask
        #

        tissue_mask = self.generate_mask(image)

        #
        # Morphological refinement
        #
        # Implemented in Section 2B.
        #

        # refined_mask = tissue_mask.copy()
        refined_mask = self.refine_mask(
            tissue_mask)

        #
        # Tissue percentage
        #
        # Updated after refinement in Section 3A.
        #

        # tissue_percent = self.tissue_percentage(refined_mask)
        #
        # Region extraction
        #

        # regions = self.extract_regions(
        #     refined_mask,
        # )
        
        regions = self.generate_regions(
            refined_mask,
        )
        
        statistics = self.compute_slide_statistics(
            refined_mask,
            regions,
        )

        tissue_percent = statistics[
            "overall_tissue_percentage"
        ]

        self.log_statistics(
            tissue_percent,
        )

        elapsed = time.perf_counter() - start_time

        #
        # Regions will be extracted in Section 3B.
        #

        regions = []
        return TissueDetectionResult(
            patient_id=patient_id,
            tissue_mask=tissue_mask,
            refined_mask=refined_mask,
            tissue_regions=regions,
            tissue_percentage=tissue_percent,
            processing_time=elapsed,
            image_height=image.shape[0],
            image_width=image.shape[1],
            algorithm=TissueDetectionAlgorithm.OTSU.value,
            metadata=statistics,
        )
                                 
        
        
        
    # ------------------------------------------------------------------
    # Morphological Operations
    # ------------------------------------------------------------------

    def morphological_open(
        self,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remove small foreground noise.

        Opening = Erosion followed by Dilation.
        """

        kernel = self.morphology_kernel

        return cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.config.open_iterations,
        )


    def morphological_close(
        self,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Fill small holes inside tissue.

        Closing = Dilation followed by Erosion.
        """

        kernel = self.morphology_kernel

        return cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.close_iterations,
        )


    # ------------------------------------------------------------------
    # Connected Component Filtering
    # ------------------------------------------------------------------

    def remove_small_regions(
        self,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remove connected tissue regions smaller than
        the configured minimum region area.
        """

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )

        filtered = np.zeros_like(binary_mask)

        for label in range(1, num_labels):

            area = stats[label, cv2.CC_STAT_AREA]

            if area >= self.config.minimum_region_area:

                filtered[labels == label] = MAX_BINARY_VALUE

        return filtered


    # ------------------------------------------------------------------
    # Hole Filling
    # ------------------------------------------------------------------

    def fill_holes(
        self,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Fill internal holes within tissue regions.

        Uses contour filling rather than flood fill,
        making it robust for histopathology images.
        """

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        filled = np.zeros_like(binary_mask)

        cv2.drawContours(
            filled,
            contours,
            contourIdx=-1,
            color=MAX_BINARY_VALUE,
            thickness=cv2.FILLED,
        )

        return filled


    # ------------------------------------------------------------------
    # Complete Refinement Pipeline
    # ------------------------------------------------------------------

    def refine_mask(
        self,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Complete refinement pipeline.

        Raw Mask
            ↓
        Opening
            ↓
        Closing
            ↓
        Remove Small Regions
            ↓
        Fill Holes
            ↓
        Refined Tissue Mask
        """

        validate_binary_mask(binary_mask)

        self.logger.info("Refining tissue mask.")

        refined = self.morphological_open(binary_mask)

        refined = self.morphological_close(refined)

        refined = self.remove_small_regions(refined)

        refined = self.fill_holes(refined)

        return refined
    
    
        # ------------------------------------------------------------------
    # Connected Component Analysis
    # ------------------------------------------------------------------

    def connected_components(
        self,
        refined_mask: np.ndarray,
    ):
        """
        Compute connected components from the refined tissue mask.

        Parameters
        ----------
        refined_mask : np.ndarray

        Returns
        -------
        tuple
            labels, stats, centroids
        """

        validate_binary_mask(refined_mask)

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                refined_mask,
                connectivity=8,
            )

        return num_labels, labels, stats, centroids

    # ------------------------------------------------------------------
    # Region Statistics
    # ------------------------------------------------------------------

    def extract_regions(
        self,
        refined_mask: np.ndarray,
    ) -> List[TissueRegion]:
        """
        Extract connected tissue regions.

        Parameters
        ----------
        refined_mask

        Returns
        -------
        List[TissueRegion]
        """

        num_labels, labels, stats, centroids = \
            self.connected_components(refined_mask)

        regions: List[TissueRegion] = []

        region_index = 0

        #
        # Skip background (label = 0)
        #

        for label in range(1, num_labels):

            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.config.minimum_region_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])

            y = int(stats[label, cv2.CC_STAT_TOP])

            width = int(stats[label, cv2.CC_STAT_WIDTH])

            height = int(stats[label, cv2.CC_STAT_HEIGHT])

            region_mask = np.zeros_like(refined_mask)

            region_mask[labels == label] = MAX_BINARY_VALUE

            contours, _ = cv2.findContours(
                region_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contour = contours[0] if len(contours) > 0 else None

            bbox_area = width * height

            tissue_ratio = (
                float(area) / float(bbox_area)
                if bbox_area > 0 else 0.0
            )

            region = TissueRegion(
                region_id=region_index,
                x=x,
                y=y,
                width=width,
                height=height,
                area=float(area),
                tissue_percentage=tissue_ratio,
                contour=contour,
                region_type=TissueRegionType.VALID,
                metadata={
                    "centroid_x": float(centroids[label][0]),
                    "centroid_y": float(centroids[label][1]),
                },
            )

            regions.append(region)

            region_index += 1

        self.logger.info(
            "Detected %d tissue regions.",
            len(regions),
        )

        return regions

    # ------------------------------------------------------------------
    # Slide Statistics
    # ------------------------------------------------------------------

    def compute_slide_statistics(
        self,
        refined_mask: np.ndarray,
        regions: List[TissueRegion],
    ) -> Dict[str, Any]:
        """
        Compute slide-level statistics.

        Parameters
        ----------
        refined_mask
        regions

        Returns
        -------
        Dict
        """

        total_pixels = refined_mask.size

        tissue_pixels = int(
            np.count_nonzero(refined_mask)
        )

        tissue_percentage = (
            tissue_pixels / total_pixels
            if total_pixels > 0 else 0.0
        )

        largest_region = 0.0

        if len(regions) > 0:
            largest_region = max(
                region.area
                for region in regions
            )

        statistics = {
            "number_of_regions": len(regions),
            "tissue_pixels": tissue_pixels,
            "background_pixels":
                total_pixels - tissue_pixels,
            "overall_tissue_percentage":
                tissue_percentage,
            "largest_region_area":
                largest_region,
            "average_region_area":
                np.mean(
                    [r.area for r in regions]
                ) if regions else 0.0,
        }

        return statistics
    
    # ------------------------------------------------------------------
    # Contour Extraction
    # ------------------------------------------------------------------

    def find_contours(
        self,
        refined_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Extract external tissue contours.

        Parameters
        ----------
        refined_mask

        Returns
        -------
        List[np.ndarray]
        """

        validate_binary_mask(refined_mask)

        contours, _ = cv2.findContours(
            refined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        self.logger.info(
            "Found %d contours.",
            len(contours),
        )

        return contours


    # ------------------------------------------------------------------
    # Bounding Box Extraction
    # ------------------------------------------------------------------

    def contour_to_region(
        self,
        contour: np.ndarray,
        region_id: int,
        refined_mask: np.ndarray,
    ) -> TissueRegion:
        """
        Convert contour into TissueRegion.
        """

        x, y, width, height = cv2.boundingRect(contour)

        contour_area = float(
            cv2.contourArea(contour)
        )

        roi_mask = np.zeros_like(refined_mask)

        cv2.drawContours(
            roi_mask,
            [contour],
            -1,
            MAX_BINARY_VALUE,
            thickness=cv2.FILLED,
        )

        tissue_pixels = int(
            np.count_nonzero(roi_mask)
        )

        bbox_area = width * height

        tissue_ratio = (
            tissue_pixels / bbox_area
            if bbox_area > 0 else 0.0
        )

        return TissueRegion(
            region_id=region_id,
            x=x,
            y=y,
            width=width,
            height=height,
            area=contour_area,
            tissue_percentage=tissue_ratio,
            contour=contour,
            region_type=TissueRegionType.VALID,
            metadata={
                "bbox_area": bbox_area,
                "tissue_pixels": tissue_pixels,
                "contour_points": len(contour),
            },
        )
        
    # ------------------------------------------------------------------
    # ROI Generation
    # ------------------------------------------------------------------

    def generate_regions(
        self,
        refined_mask: np.ndarray,
    ) -> List[TissueRegion]:
        """
        Generate TissueRegion objects from contours.
        """

        contours = self.find_contours(
            refined_mask
        )

        regions = []

        region_id = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.config.minimum_region_area:
                continue

            region = self.contour_to_region(
                contour,
                region_id,
                refined_mask,
            )

            regions.append(region)

            region_id += 1

        self.logger.info(
            "Generated %d ROI objects.",
            len(regions),
        )

        return regions