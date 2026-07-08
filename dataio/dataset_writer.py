"""
==============================================================================
File        : dataset_writer.py

Description :
    Production-quality dataset writer responsible for persisting the
    preprocessed dataset generated during the preprocessing pipeline.

Responsibilities
----------------
1. Create dataset directory structure.
2. Save RGB image patches.
3. Save annotation mask patches.
4. Export metadata (CSV / JSON).
5. Export dataset statistics.
6. Verify dataset integrity.
7. Generate preprocessing reports.

Pipeline
--------
Patch Objects
        │
        ▼
Dataset Manifest
        │
        ▼
Dataset Writer
        │
        ├── patches/
        ├── patchMasks/
        ├── metadata/
        ├── statistics/
        ├── graphs/
        └── debug/

Author : Your Name
==============================================================================
"""

from __future__ import annotations


from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import csv
import json
import shutil
import tempfile
import hashlib
import os

import cv2
import numpy as np

from preprocessing.patch_extractor import PATCH_SIZE, Patch


from utils.exceptions import (
    FileWriteError,
    ValidationError,
)

from utils.logger import get_logger
from utils.validator import Validator

from dataset.data_structures import PatchRecord

from dataclasses import (
    dataclass,
    field,
    asdict,
)



# =============================================================================
# Constants
# =============================================================================

DEFAULT_IMAGE_EXTENSION = ".png"

DEFAULT_CSV_NAME = "dataset.csv"

DEFAULT_JSON_NAME = "dataset.json"

DEFAULT_STATISTICS_NAME = "dataset_statistics.json"

DEFAULT_ENCODING = "utf-8"

WRITER_VERSION = "1.0.0"

BCSS_LABEL_INDEX = {
    "tumor": 0,
    "stroma": 1,
    "lymphocyte": 2,
    "necrosis": 3,
}

# =============================================================================
# Enumerations
# =============================================================================


class OutputFormat(Enum):
    """
    Supported metadata formats.
    """

    CSV = "csv"

    JSON = "json"

    BOTH = "both"


class OverwritePolicy(Enum):
    """
    File overwrite strategy.
    """

    NEVER = "never"

    OVERWRITE = "overwrite"

    SKIP = "skip"


class CompressionType(Enum):
    """
    Image compression method.
    """

    PNG = "png"

    NONE = "none"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DatasetWriterConfig:
    """
    Configuration for DatasetWriter.
    """

    #
    # Output root directory
    #

    output_dir: Union[str, Path]

    #
    # Folder names
    #

    patches_dir: str = "patches"

    masks_dir: str = "patchMasks"

    metadata_dir: str = "metadata"

    statistics_dir: str = "statistics"

    graphs_dir: str = "graphs"

    debug_dir: str = "debug"

    #
    # Metadata
    #

    metadata_format: OutputFormat = OutputFormat.BOTH

    #
    # Images
    #

    image_extension: str = DEFAULT_IMAGE_EXTENSION

    compression: CompressionType = CompressionType.PNG

    png_compression_level: int = 3

    #
    # Writing policy
    #

    overwrite_policy: OverwritePolicy = (
        OverwritePolicy.SKIP
    )

    atomic_write: bool = True

    verify_after_write: bool = True

    #
    # Statistics
    #

    save_statistics: bool = True

    #
    # Debugging
    #

    debug: bool = False


# =============================================================================
# Dataset Statistics
# =============================================================================


@dataclass
class DatasetStatistics:
    """
    Summary statistics of the generated dataset.
    """

    total_patients: int = 0

    total_regions: int = 0

    total_patches: int = 0

    tumor_patches: int = 0

    stroma_patches: int = 0

    lymphocyte_patches: int = 0

    necrosis_patches: int = 0

    average_tissue_percentage: float = 0.0

    generated_files: int = 0

    skipped_files: int = 0

    failed_files: int = 0


# =============================================================================
# Dataset Manifest
# =============================================================================


@dataclass
class DatasetManifest:
    """
    Serializable dataset manifest.

    This object is produced by label_generator.py and consumed
    by DatasetWriter.
    """

    patches: List[Patch] = field(
        default_factory=list
    )

    # records: List[Dict[str, Any]] = field(
    #     default_factory=list
    # )
    records: List[PatchRecord] = field(
        default_factory=list
    )

    statistics: Optional[DatasetStatistics] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# =============================================================================
# Write Result
# =============================================================================


@dataclass
class DatasetWriteResult:
    """
    Result returned after dataset writing.
    """

    success: bool

    output_directory: Path

    statistics: DatasetStatistics

    manifest_path: Optional[Path] = None

    metadata_path: Optional[Path] = None

    message: str = ""
    
    
# =============================================================================
# Dataset Writer
# =============================================================================

from datetime import datetime


class DatasetWriter:
    """
    Production-quality dataset writer.

    Responsibilities
    ----------------
    * Validate configuration
    * Create directory structure
    * Save dataset
    * Export metadata
    * Verify integrity
    """

    def __init__(
        self,
        config: DatasetWriterConfig,
    ) -> None:

        Validator.validate_type(
            config,
            DatasetWriterConfig,
            "config",
        )

        self.config = config

        self.logger = get_logger(
            self.__class__.__name__
        )

        self._initialize()

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def _initialize(self) -> None:
        """
        Initialize DatasetWriter.
        """

        self.output_dir = Path(
            self.config.output_dir
        ).expanduser().resolve()

        self.patches_dir = (
            self.output_dir /
            self.config.patches_dir
        )

        self.masks_dir = (
            self.output_dir /
            self.config.masks_dir
        )

        self.metadata_dir = (
            self.output_dir /
            self.config.metadata_dir
        )

        self.statistics_dir = (
            self.output_dir /
            self.config.statistics_dir
        )

        self.graphs_dir = (
            self.output_dir /
            self.config.graphs_dir
        )

        self.debug_dir = (
            self.output_dir /
            self.config.debug_dir
        )

        self._created_patient_dirs: set[str] = set()

        self.validate_configuration()

        self.create_directory_structure()

        self.logger.info(
            "DatasetWriter initialized."
        )
        self.logger.info(
            "Output directory      : %s",
            self.output_dir,
        )
        self.logger.info(
            "Metadata format       : %s",
            self.config.metadata_format.value,
        )
        self.logger.info(
            "Compression           : %s",
            self.config.compression.value,
        )
        self.logger.info(
            "Overwrite policy      : %s",
            self.config.overwrite_policy.value,
        )
        self.logger.info(
            "Atomic writing        : %s",
            self.config.atomic_write,
        )

    # ---------------------------------------------------------------------
    # Configuration Validation
    # ---------------------------------------------------------------------

    def validate_configuration(
        self,
    ) -> None:
        """
        Validate configuration.
        """

        if self.config.png_compression_level < 0 \
                or self.config.png_compression_level > 9:

            raise ValidationError(
                "PNG compression must be "
                "between 0 and 9."
            )

        if self.config.image_extension.lower() != ".png":

            raise ValidationError(
                "Only PNG images are currently "
                "supported."
            )

    # ---------------------------------------------------------------------
    # Directory Creation
    # ---------------------------------------------------------------------

    def create_directory_structure(
        self,
    ) -> None:
        """
        Create dataset directory hierarchy.
        """

        directories = [

            self.output_dir,

            self.patches_dir,

            self.masks_dir,

            self.metadata_dir,

            self.statistics_dir,

            self.graphs_dir,

            self.debug_dir,

        ]

        for directory in directories:

            try:

                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            except Exception as e:

                raise FileWriteError(

                    f"Unable to create "

                    f"{directory}"

                ) from e

        self.logger.info(
            "Directory structure created."
        )

    # ---------------------------------------------------------------------
    # Patient Directories
    # ---------------------------------------------------------------------

    def create_patient_directories(
        self,
        patient_id: str,
    ) -> tuple[Path, Path]:
        """
        Create patient-specific directories.

        Returns
        -------
        image_dir,
        mask_dir
        """

        patient_key = str(patient_id)

        if patient_key in self._created_patient_dirs:
            return (
                self.patches_dir / patient_id,
                self.masks_dir / patient_id,
            )

        image_dir = (
            self.patches_dir /
            patient_id
        )

        mask_dir = (
            self.masks_dir /
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

        self._created_patient_dirs.add(patient_key)

        return image_dir, mask_dir

    # ---------------------------------------------------------------------
    # File Existence
    # ---------------------------------------------------------------------

    def file_exists(
        self,
        path: Path,
    ) -> bool:
        """
        Check whether file exists.
        """

        return path.exists()

    # ---------------------------------------------------------------------
    # Overwrite Policy
    # ---------------------------------------------------------------------

    def should_write(
        self,
        path: Path,
    ) -> bool:
        """
        Determine whether file should
        be written.
        """

        if not path.exists():

            return True

        policy = self.config.overwrite_policy

        if policy == OverwritePolicy.OVERWRITE:

            return True

        if policy == OverwritePolicy.SKIP:

            self.logger.debug(
                "Skipping existing file: %s",
                path.name,
            )

            return False

        if policy == OverwritePolicy.NEVER:

            raise FileWriteError(
                f"{path} already exists."
            )

        return False

    # ---------------------------------------------------------------------
    # Temporary File
    # ---------------------------------------------------------------------

    def temporary_file(
        self,
        destination: Path,
    ) -> Path:
        """
        Create temporary file path
        for atomic writing.
        """

        suffix = destination.suffix

        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            dir=str(destination.parent),
        )

        os.close(fd)

        return Path(temp_path)

    # ---------------------------------------------------------------------
    # Atomic Move
    # ---------------------------------------------------------------------

    def atomic_move(
        self,
        temporary: Path,
        destination: Path,
    ) -> None:
        """
        Atomic file replacement.
        """

        shutil.move(
            str(temporary),
            str(destination),
        )

    # ---------------------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------------------

    @staticmethod
    def current_timestamp() -> str:
        """
        Current ISO timestamp.
        """

        return datetime.now().isoformat()

    # ---------------------------------------------------------------------
    # MD5
    # ---------------------------------------------------------------------

    @staticmethod
    def compute_md5(
        path: Path,
    ) -> str:
        """
        Compute MD5 checksum.
        """

        md5 = hashlib.md5()

        with open(path, "rb") as file:

            while True:

                chunk = file.read(8192)

                if not chunk:

                    break

                md5.update(chunk)

        return md5.hexdigest()
    
# =============================================================================
# Image and Mask Saving
# =============================================================================

    def _write_png(
        self,
        image: np.ndarray,
        destination: Path,
    ) -> None:
        """
        Write PNG image using atomic file writing.
        """

        Validator.validate_type(
            image,
            np.ndarray,
            "image",
        )

        if image.dtype != np.uint8:
            raise ValidationError(
                "Image data must use uint8 dtype."
            )

        if not self.should_write(destination):
            return

        temporary = destination

        if self.config.atomic_write:
            temporary = self.temporary_file(destination)

        compression_level = self.config.png_compression_level
        if self.config.compression == CompressionType.NONE:
            compression_level = 0

        success = cv2.imwrite(
            str(temporary),
            image,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                compression_level,
            ],
        )

        if not success:

            if temporary.exists():
                temporary.unlink(missing_ok=True)

            raise FileWriteError(
                f"Failed to save image: {destination}"
            )

        if self.config.atomic_write:

            self.atomic_move(
                temporary,
                destination,
            )

        if self.config.verify_after_write:

            self.verify_file(destination)

    # ---------------------------------------------------------------------

    def verify_file(
        self,
        path: Path,
    ) -> None:
        """
        Verify that a written file exists and
        is not empty.
        """

        if not path.exists():

            raise FileWriteError(
                f"Missing file: {path}"
            )

        if path.stat().st_size == 0:

            raise FileWriteError(
                f"Empty file: {path}"
            )

    # ---------------------------------------------------------------------

    def save_patch(
        self,
        patch: Patch,
    ) -> None:
        """
        Save RGB image patch.
        """

        Validator.validate_type(
            patch,
            Patch,
            "patch",
        )

        patient_dir, _ = self.create_patient_directories(
            patch.metadata.patient_id
        )

        destination = (
            patient_dir /
            patch.metadata.filename
        )

        self._write_png(
            patch.image,
            destination,
        )

        self.logger.debug(
            "Saved RGB patch: %s",
            destination.name,
        )

    # ---------------------------------------------------------------------

    def save_mask(
        self,
        patch: Patch,
    ) -> None:
        """
        Save annotation mask patch.
        """

        Validator.validate_type(
            patch,
            Patch,
            "patch",
        )

        _, patient_dir = self.create_patient_directories(
            patch.metadata.patient_id
        )

        destination = (
            patient_dir /
            patch.metadata.filename
        )

        self._write_png(
            patch.mask,
            destination,
        )

        self.logger.debug(
            "Saved mask patch: %s",
            destination.name,
        )

    # ---------------------------------------------------------------------

    def save_patch_pair(
        self,
        patch: Patch,
    ) -> None:
        """
        Save RGB patch and corresponding mask.
        """

        self.save_patch(
            patch
        )

        self.save_mask(
            patch
        )
        

# ---------------------------------------------------------------------

    def save_patch_batch(
        self,
        patches: List[Patch],
    ) -> DatasetStatistics:
        """
        Save a collection of patches.

        Returns
        -------
        DatasetStatistics
        """

        statistics = DatasetStatistics()

        processed_patients = set()

        for patch in patches:

            try:

                self.save_patch_pair(
                    patch
                )

                statistics.generated_files += 2

                processed_patients.add(
                    patch.metadata.patient_id
                )

            except Exception as error:

                statistics.failed_files += 1

                self.logger.exception(
                    "Failed to save patch %s : %s",
                    patch.metadata.filename,
                    error,
                )

        statistics.total_patches = len(patches)

        statistics.total_patients = len(
            processed_patients
        )

        return statistics
    

# =============================================================================
# Metadata Export
# =============================================================================

    def _json_serializer(
        self,
        obj: Any,
    ) -> Any:
        """
        JSON serializer for unsupported objects.
        """

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        raise TypeError(
            f"Object of type {type(obj)} "
            "is not JSON serializable."
        )

    # ---------------------------------------------------------------------

    def export_csv(
        self,
        # records: List[Dict[str, Any]],
        records: List[PatchRecord],
        filename: str = DEFAULT_CSV_NAME,
    ) -> Path:
        """
        Export metadata as CSV.
        """

        if len(records) == 0:

            raise ValidationError(
                "No metadata records available."
            )

        destination = (
            self.metadata_dir /
            filename
        )

        if not self.should_write(destination):
            return destination

        temporary = destination

        if self.config.atomic_write:

            temporary = self.temporary_file(
                destination
            )

        try:

            sorted_records = sorted(
                records,
                key=lambda record: (
                    record.patient_id,
                    record.region_id,
                    record.y,
                    record.x,
                ),
            )

            fieldnames = list(
                 asdict(sorted_records[0]).keys()
            )
           

            with open(
                temporary,
                "w",
                newline="",
                encoding=DEFAULT_ENCODING,
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                writer.writerows(
                     asdict(record)
                     for record in sorted_records
                )

            if self.config.atomic_write:

                self.atomic_move(
                    temporary,
                    destination,
                )

            if self.config.verify_after_write:

                self.verify_file(
                    destination
                )

            self.logger.info(

                "CSV metadata written: %s",

                destination.name,

            )

            return destination

        except Exception as error:

            if temporary.exists():

                temporary.unlink(
                    missing_ok=True
                )

            raise FileWriteError(

                f"Unable to write CSV: {destination}"

            ) from error

    # ---------------------------------------------------------------------

    def export_json(
        self,
        # records: List[Dict[str, Any]],
        records: List[PatchRecord],
        filename: str = DEFAULT_JSON_NAME,
    ) -> Path:
        """
        Export metadata as JSON.
        """

        destination = (
            self.metadata_dir /
            filename
        )

        if not self.should_write(
            destination
        ):
            return destination

        temporary = destination

        if self.config.atomic_write:

            temporary = self.temporary_file(
                destination
            )

        try:

            with open(
                temporary,
                "w",
                encoding=DEFAULT_ENCODING,
            ) as file:

                json.dump(
                    [
                        asdict(record)
                        for record in sorted(
                            records,
                            key=lambda record: (
                                record.patient_id,
                                record.region_id,
                                record.y,
                                record.x,
                            ),
                        )
                    ],
                    file,
                    indent=4,
                    default=self._json_serializer,
                    )

            if self.config.atomic_write:

                self.atomic_move(

                    temporary,

                    destination,

                )

            if self.config.verify_after_write:

                self.verify_file(
                    destination
                )

            self.logger.info(

                "JSON metadata written: %s",

                destination.name,

            )

            return destination

        except Exception as error:

            if temporary.exists():

                temporary.unlink(
                    missing_ok=True
                )

            raise FileWriteError(

                f"Unable to write JSON: {destination}"

            ) from error

    # ---------------------------------------------------------------------

    def export_metadata(
        self,
        manifest: DatasetManifest,
    ) -> Dict[str, Path]:
        """
        Export metadata according to the configured format.
        """

        Validator.validate_type(

            manifest,

            DatasetManifest,

            "manifest",

        )

        exported = {}

        if self.config.metadata_format in (

            OutputFormat.CSV,

            OutputFormat.BOTH,

        ):

            exported["csv"] = self.export_csv(
                manifest.records
            )

        if self.config.metadata_format in (

            OutputFormat.JSON,

            OutputFormat.BOTH,

        ):

            exported["json"] = self.export_json(
                manifest.records
            )

        self.logger.info(

            "Metadata export completed."

        )

        return exported

    # ---------------------------------------------------------------------

    def export_patient_metadata(
        self,
        patient_id: str,
        # records: List[Dict[str, Any]],
        records: List[PatchRecord],
    ) -> Path:
        """
        Export metadata for one patient.
        """

        patient_records = [

            record

            for record in records

            if record.patient_id == patient_id

        ]

        filename = (
            f"{patient_id}.csv"
        )

        return self.export_csv(

            patient_records,

            filename,

        )
        
        
# =============================================================================
# Dataset Statistics Generation
# =============================================================================

    def generate_statistics(
        self,
        manifest: DatasetManifest,
    ) -> DatasetStatistics:
        """
        Generate dataset statistics from metadata records.

        Parameters
        ----------
        manifest : DatasetManifest

        Returns
        -------
        DatasetStatistics
        """

        Validator.validate_type(
            manifest,
            DatasetManifest,
            "manifest",
        )

        statistics = DatasetStatistics()

        if len(manifest.records) == 0:

            self.logger.warning(
                "No metadata records available."
            )

            return statistics

        patient_ids = set()
        region_ids = set()

        tissue_percentages = []

        for record in manifest.records:

            patient_ids.add(
                record.patient_id
            )

            region_ids.add(
                (
                    record.patient_id,
                    record.region_id,
                )
            )

            statistics.total_patches += 1
            Validator.validate_label_vector(
                    record.labels
                )
            labels = record.labels

            statistics.tumor_patches += labels[BCSS_LABEL_INDEX["tumor"]]
            statistics.stroma_patches += labels[BCSS_LABEL_INDEX["stroma"]]
            statistics.lymphocyte_patches += labels[BCSS_LABEL_INDEX["lymphocyte"]]
            statistics.necrosis_patches += labels[BCSS_LABEL_INDEX["necrosis"]]

            tissue_percentages.append(
                float(
                    record.tissue_percentage
                )
            )

        statistics.total_patients = len(
            patient_ids
        )

        statistics.total_regions = len(
            region_ids
        )

        if len(tissue_percentages) > 0:

            statistics.average_tissue_percentage = round(
                float(
                    np.mean(
                        tissue_percentages
                    )
                ),
                2,
            )

        self.logger.info(

            "Dataset Statistics Generated"

        )

        self.logger.info(

            "Patients      : %d",

            statistics.total_patients,

        )

        self.logger.info(

            "Regions       : %d",

            statistics.total_regions,

        )

        self.logger.info(

            "Patches       : %d",

            statistics.total_patches,

        )

        self.logger.info(

            "Tumor         : %d",

            statistics.tumor_patches,

        )

        self.logger.info(

            "Stroma        : %d",

            statistics.stroma_patches,

        )

        self.logger.info(

            "Lymphocyte    : %d",

            statistics.lymphocyte_patches,

        )

        self.logger.info(

            "Necrosis      : %d",

            statistics.necrosis_patches,

        )

        self.logger.info(

            "Average Tissue: %.2f%%",

            statistics.average_tissue_percentage,

        )

        return statistics

    # -------------------------------------------------------------------------

    def export_statistics(
        self,
        statistics: DatasetStatistics,
        filename: str = DEFAULT_STATISTICS_NAME,
    ) -> Path:
        """
        Export dataset statistics as JSON.

        Parameters
        ----------
        statistics : DatasetStatistics

        Returns
        -------
        Path
        """

        destination = (
            self.statistics_dir /
            filename
        )

        if not self.should_write(
            destination
        ):
            return destination

        temporary = destination

        if self.config.atomic_write:

            temporary = self.temporary_file(
                destination
            )

        try:

            with open(
                temporary,
                "w",
                encoding=DEFAULT_ENCODING,
            ) as file:

                json.dump(

                    asdict(statistics),

                    file,

                    indent=4,

                )

            if self.config.atomic_write:

                self.atomic_move(
                    temporary,
                    destination,
                )

            if self.config.verify_after_write:

                self.verify_file(
                    destination
                )

            self.logger.info(

                "Dataset statistics saved."

            )

            return destination

        except Exception as error:

            if temporary.exists():

                temporary.unlink(
                    missing_ok=True
                )

            raise FileWriteError(

                f"Unable to save statistics: {destination}"

            ) from error

    # -------------------------------------------------------------------------

    def compute_label_distribution(
        self,
        statistics: DatasetStatistics,
    ) -> Dict[str, float]:
        """
        Compute normalized label distribution.

        Returns
        -------
        Dictionary containing label percentages.
        """

        total = max(
            statistics.total_patches,
            1,
        )

        distribution = {

            "tumor":

                statistics.tumor_patches
                / total,

            "stroma":

                statistics.stroma_patches
                / total,

            "lymphocyte":

                statistics.lymphocyte_patches
                / total,

            "necrosis":

                statistics.necrosis_patches
                / total,

        }

        return distribution

    # -------------------------------------------------------------------------

    def print_statistics(
        self,
        statistics: DatasetStatistics,
    ) -> None:
        """
        Print statistics to logger.
        """

        distribution = self.compute_label_distribution(
            statistics
        )

        self.logger.info(
            "========== Dataset Summary =========="
        )

        self.logger.info(
            "Patients : %d",
            statistics.total_patients,
        )

        self.logger.info(
            "Regions  : %d",
            statistics.total_regions,
        )

        self.logger.info(
            "Patches  : %d",
            statistics.total_patches,
        )

        self.logger.info(
            "Label Distribution : %s",
            distribution,
        )
        

# =============================================================================
# Dataset Integrity Verification
# =============================================================================

    def verify_dataset(
        self,
        manifest: DatasetManifest,
    ) -> Dict[str, Any]:
        """
        Perform integrity verification of the generated dataset.

        Parameters
        ----------
        manifest : DatasetManifest

        Returns
        -------
        Dictionary containing verification report.
        """

        Validator.validate_type(
            manifest,
            DatasetManifest,
            "manifest",
        )

        report = {

            "valid": True,

            "total_records": len(
                manifest.records
            ),

            "missing_images": [],

            "missing_masks": [],

            "duplicate_filenames": [],

            "duplicate_patch_ids": [],

            "invalid_coordinates": [],

            "invalid_labels": [],

            "invalid_shapes": [],

            "verified_at": self.current_timestamp(),

            "dataset_version": "1.0",

            "writer_version": WRITER_VERSION,

            "warnings": [],

        }

        if len(manifest.patches) != len(manifest.records):
            report["warnings"].append(
                "Manifest patch/record count mismatch."
            )
            report["valid"] = False

        filenames = set()

        patch_ids = set()

        for record in manifest.records:

            image = None
            mask = None

            #
            # ------------------------------------------------------------
            # Duplicate filename
            # ------------------------------------------------------------
            #

            filename = record.filename
            filename_key = (
                record.patient_id,
                filename,
            )

            if filename_key in filenames:

                report["duplicate_filenames"].append(
                    f"{record.patient_id}/{filename}"
                )

            else:

                filenames.add(
                    filename_key
                )

            #
            # ------------------------------------------------------------
            # Duplicate Patch ID
            # ------------------------------------------------------------
            #

            patch_id = record.patch_id

            if patch_id in patch_ids:

                report["duplicate_patch_ids"].append(
                    patch_id
                )

            else:

                patch_ids.add(
                    patch_id
                )

            #
            # ------------------------------------------------------------
            # Image existence
            # ------------------------------------------------------------
            #

            image_path = (

                self.patches_dir /

                record.patient_id /

                filename

            )

            if not image_path.exists():

                report["missing_images"].append(

                    str(image_path)

                )

            else:

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_UNCHANGED,
                )

                if image is None or image.size == 0:

                    report["missing_images"].append(

                        str(image_path)

                    )

            #
            # ------------------------------------------------------------
            # Mask existence
            # ------------------------------------------------------------
            #

            mask_path = (

                self.masks_dir /

                record.patient_id /

                filename

            )

            if not mask_path.exists():

                report["missing_masks"].append(

                    str(mask_path)

                )

            else:

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_UNCHANGED,
                )

                if mask is None or mask.size == 0:

                    report["missing_masks"].append(

                        str(mask_path)

                    )

            #
            # ------------------------------------------------------------
            # Coordinates
            # ------------------------------------------------------------
            #

            try:
                Validator.validate_patch_coordinate(
                    record.x,
                    record.y,
                    record.width,
                    record.height,
                    PATCH_SIZE,
                )
            except ValidationError:
                report["invalid_coordinates"].append(
                    filename
                )

            #
            # ------------------------------------------------------------
            # Labels
            # ------------------------------------------------------------
            #
            label_invalid = False

            try:
                Validator.validate_label_vector(
                    record.labels
                )
            except ValidationError:
                label_invalid = True

            labels = record.labels

            if any(label not in (0, 1) for label in labels):
                label_invalid = True

            if label_invalid and filename not in report["invalid_labels"]:
                report["invalid_labels"].append(
                    filename
                )

            if image is not None and mask is not None:
                if image.shape[:2] != mask.shape[:2]:
                    report["invalid_shapes"].append(
                        filename
                    )

        #
        # ----------------------------------------------------------------
        # Overall validity
        # ----------------------------------------------------------------
        #

        if (

            report["missing_images"]

            or report["missing_masks"]

            or report["duplicate_filenames"]

            or report["duplicate_patch_ids"]

            or report["invalid_coordinates"]

            or report["invalid_labels"]

            or report["invalid_shapes"]

            or report["warnings"]

        ):

            report["valid"] = False

        self.logger.info(

            "Dataset verification completed."

        )

        self.logger.info(

            "Dataset valid : %s",

            report["valid"],

        )

        return report

    # -------------------------------------------------------------------------

    def export_verification_report(
        self,
        report: Dict[str, Any],
        filename: str = "verification_report.json",
    ) -> Path:
        """
        Export dataset verification report.
        """

        destination = (

            self.statistics_dir /

            filename

        )

        temporary = destination

        if self.config.atomic_write:

            temporary = self.temporary_file(
                destination
            )

        try:

            with open(

                temporary,

                "w",

                encoding=DEFAULT_ENCODING,

            ) as file:

                json.dump(

                    report,

                    file,

                    indent=4,

                )

            if self.config.atomic_write:

                self.atomic_move(

                    temporary,

                    destination,

                )

            self.logger.info(

                "Verification report saved."

            )

            return destination

        except Exception as error:

            raise FileWriteError(

                "Unable to save verification report."

            ) from error

    # -------------------------------------------------------------------------

    def verify_dataset_statistics(
        self,
        statistics: DatasetStatistics,
    ) -> None:
        """
        Perform sanity checks on dataset statistics.
        """

        if statistics.total_patches == 0:

            self.logger.warning(

                "Dataset contains no patches."

            )

        if statistics.total_patients == 0:

            self.logger.warning(

                "No patients detected."

            )

        if statistics.generated_files == 0:

            self.logger.warning(

                "No files were written."

            )

        if statistics.failed_files > 0:

            self.logger.warning(

                "%d files failed during writing.",

                statistics.failed_files,

            )

        self.logger.info(

            "Statistics verification completed."

        )
        
        # =============================================================================
# Public API
# =============================================================================



    def write_dataset(
        self,
        manifest: DatasetManifest,
    ) -> DatasetWriteResult:
        """
        Complete dataset writing pipeline.

        Parameters
        ----------
        manifest : DatasetManifest

        Returns
        -------
        DatasetWriteResult
        """

        Validator.validate_type(
            manifest,
            DatasetManifest,
            "manifest",
        )

        start_time = time.perf_counter()

        self.logger.info(
            "=" * 80
        )
        self.logger.info(
            "Starting Dataset Writing Pipeline"
        )
        self.logger.info(
            "=" * 80
        )

        try:

            # -------------------------------------------------------------
            # Save RGB patches and masks
            # -------------------------------------------------------------

            write_statistics = self.save_patch_batch(
                manifest.patches
            )

            # -------------------------------------------------------------
            # Export metadata
            # -------------------------------------------------------------

            metadata_files = self.export_metadata(
                manifest
            )

            # -------------------------------------------------------------
            # Generate dataset statistics
            # -------------------------------------------------------------

            statistics = self.generate_statistics(
                manifest
            )

            #
            # Preserve file-writing statistics.
            #

            statistics.generated_files = (
                write_statistics.generated_files
            )

            statistics.failed_files = (
                write_statistics.failed_files
            )

            statistics.skipped_files = (
                write_statistics.skipped_files
            )

            statistics.total_patients = max(
                statistics.total_patients,
                write_statistics.total_patients,
            )

            self.export_statistics(
                statistics
            )

            # -------------------------------------------------------------
            # Dataset verification
            # -------------------------------------------------------------

            verification_report = self.verify_dataset(
                manifest
            )

            self.export_verification_report(
                verification_report
            )

            self.verify_dataset_statistics(
                statistics
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            self.logger.info(
                "Dataset successfully written."
            )

            self.logger.info(
                "Elapsed Time : %.2f seconds",
                elapsed,
            )

            self.logger.info(
                "=" * 80
            )

            return DatasetWriteResult(

                success=True,

                output_directory=self.output_dir,

                statistics=statistics,

                manifest_path=metadata_files.get(
                    "csv",
                    None,
                ),

                metadata_path=metadata_files.get(
                    "json",
                    None,
                ),

                message=(
                    f"Dataset successfully written "
                    f"in {elapsed:.2f} seconds."
                ),

            )

        except Exception as error:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            self.logger.exception(
                "Dataset writing failed."
            )

            self.logger.info(
                "Elapsed Time : %.2f seconds",
                elapsed,
            )

            return DatasetWriteResult(

                success=False,

                output_directory=self.output_dir,

                statistics=DatasetStatistics(),

                message=str(error),

            )

    # -------------------------------------------------------------------------

    def __call__(
        self,
        manifest: DatasetManifest,
    ) -> DatasetWriteResult:
        """
        Callable interface.

        Example
        -------
        writer(manifest)
        """

        return self.write_dataset(
            manifest
        )