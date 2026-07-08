"""
exceptions.py

Custom exception hierarchy for the HistoGraphWSL project.

Every project-specific exception should inherit from
HistoGraphWSLError so that preprocessing can gracefully
handle expected failures.

1. Every project-specific exception inherits from a single base class (HistoGraphWSLError), allowing global error handling with a single except.
2. Exceptions are organized hierarchically by module (configuration, WSI handling, patch extraction, graph construction, etc.), making the codebase easier to navigate and extend.
3. Future modules can introduce more specialized exceptions without changing existing code.
4. Logging becomes more informative because exception types clearly indicate the source of an error.

Example
-------
try:
    ...
except HistoGraphWSLError as error:
    logger.error(error)
"""

from __future__ import annotations


class HistoGraphWSLError(Exception):
    """
    Base class for all project-specific exceptions.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==========================================================
# Configuration Exceptions
# ==========================================================

class ConfigurationError(HistoGraphWSLError):
    """
    Raised when the configuration is invalid.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ValidationError(HistoGraphWSLError):
    """
    Raised when user inputs fail validation.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==========================================================
# File System Exceptions
# ==========================================================

class FileSystemError(HistoGraphWSLError):
    """
    Base class for filesystem related exceptions.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WSINotFoundError(FileSystemError):
    """
    Raised when a WSI cannot be located.
    """


class MaskNotFoundError(FileSystemError):
    """
    Raised when the corresponding mask is missing.
    """


class OutputDirectoryError(FileSystemError):
    """
    Raised when an output directory cannot be created.
    """


class FileWriteError(FileSystemError):
    """
    Raised when writing to a file fails.
    """


class FileReadError(FileSystemError):
    """
    Raised when reading from a file fails.
    """


# ==========================================================
# Whole Slide Image Exceptions
# ==========================================================

class WSIError(HistoGraphWSLError):
    """
    Base class for WSI-related exceptions.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class SlideOpenError(WSIError):
    """
    Raised when OpenSlide fails to open a slide.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedSlideFormatError(WSIError):
    """
    Raised when the slide format is unsupported.
    """


class InvalidMagnificationError(WSIError):
    """
    Raised when the requested magnification level
    is unavailable.
    """


# ==========================================================
# Tissue Detection
# ==========================================================

class TissueDetectionError(HistoGraphWSLError):
    """
    Raised when tissue detection fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==========================================================
# Patch Extraction
# ==========================================================

class PatchExtractionError(HistoGraphWSLError):
    """
    Raised when patch extraction fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidPatchCoordinateError(PatchExtractionError):
    """
    Raised when patch coordinates are outside
    the slide boundary.
    """


class EmptyPatchError(PatchExtractionError):
    """
    Raised when an extracted patch contains
    no valid tissue.
    """


# ==========================================================
# Mask Processing
# ==========================================================

class MaskProcessingError(HistoGraphWSLError):
    """
    Raised when processing annotation masks fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidMaskValueError(MaskProcessingError):
    """
    Raised when unexpected mask values are found.
    """


class MaskAlignmentError(MaskProcessingError):
    """
    Raised when the WSI and mask are misaligned.
    """


# ==========================================================
# Label Generation
# ==========================================================

class LabelGenerationError(HistoGraphWSLError):
    """
    Raised when weak label generation fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==========================================================
# Graph Construction
# ==========================================================

class GraphConstructionError(HistoGraphWSLError):
    """
    Raised when graph construction fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class EmptyGraphError(GraphConstructionError):
    """
    Raised when no graph nodes are available.
    """


class GraphConnectivityError(GraphConstructionError):
    """
    Raised when graph connectivity cannot be established.
    """


# ==========================================================
# Dataset
# ==========================================================

class DatasetError(HistoGraphWSLError):
    """
    Raised for dataset-related failures.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MetadataError(DatasetError):
    """
    Raised when metadata is missing or corrupted.
    """


# ==========================================================
# Preprocessing
# ==========================================================

class PreprocessingError(HistoGraphWSLError):
    """
    Raised when preprocessing fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==========================================================
# Visualization
# ==========================================================

class VisualizationError(HistoGraphWSLError):
    """
    Raised when visualization fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==========================================================
# Debugging
# ==========================================================

class DebuggingError(HistoGraphWSLError):
    """
    Raised by debugging utilities.
    """