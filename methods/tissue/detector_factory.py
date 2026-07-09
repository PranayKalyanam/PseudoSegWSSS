"""
detector_factory.py

Factory for creating tissue detection algorithms.

The factory decouples TissueLoader from concrete
detector implementations. New detectors can be added
without modifying the loader.
"""

from __future__ import annotations

from methods.tissue.otsu_detector import OtsuDetector

# Uncomment when implemented
# from methods.tissue.double_pass_detector import DoublePassDetector


class DetectorFactory:
    """
    Creates tissue detector instances.
    """

    _DETECTORS = {
        "otsu": OtsuDetector,
        # "double_pass": DoublePassDetector,
    }

    @classmethod
    def create(cls, name: str):
        """
        Create a tissue detector.

        Parameters
        ----------
        name
            Detector name from configuration.

        Returns
        -------
        object
            Detector instance.

        Raises
        ------
        ValueError
            If the detector is not registered.
        """

        key = name.strip().lower()

        if key not in cls._DETECTORS:

            available = ", ".join(cls._DETECTORS.keys())

            raise ValueError(
                f"Unknown tissue detector '{name}'. "
                f"Available detectors: {available}"
            )

        return cls._DETECTORS[key]()

    @classmethod
    def available_detectors(cls) -> list[str]:
        """
        Return all registered detector names.
        """

        return sorted(cls._DETECTORS.keys())

    @classmethod
    def register(
        cls,
        name: str,
        detector_class,
    ) -> None:
        """
        Register a new detector.

        This allows external modules to add detectors
        without modifying this file.
        """

        cls._DETECTORS[name.lower()] = detector_class