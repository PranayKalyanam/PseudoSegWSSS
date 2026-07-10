"""
patch_visualizer.py

Visualization utilities for the patch extraction pipeline.

This visualizer is compatible with both

1. Old pipeline
   Image -> Tissue Detection -> Patch Extraction

2. New pipeline
   Image -> Patch Extraction -> Tissue Filtering

It never modifies any patient data.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class PatchVisualizer:
    """
    Saves intermediate visualizations for debugging.
    """

    # ==========================================================
    # Public API
    # ==========================================================

    @staticmethod
    def save(
        patient,
        output_directory: str,
    ) -> None:

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------
        # Working image
        # -----------------------------------------

        if patient.working_image is not None:
            PatchVisualizer._save_working_image(
                patient,
                output_directory,
            )

        # -----------------------------------------
        # Working annotation
        # -----------------------------------------

        if patient.working_mask is not None:
            PatchVisualizer._save_working_mask(
                patient,
                output_directory,
            )

        # -----------------------------------------
        # Tissue mask (only if available)
        # -----------------------------------------

        if patient.has_tissue_mask:
            PatchVisualizer._save_tissue_mask(
                patient,
                output_directory,
            )

        # -----------------------------------------
        # Patch overlay
        # -----------------------------------------

        if patient.has_patches:
            PatchVisualizer._save_patch_overlay(
                patient,
                output_directory,
            )

            PatchVisualizer._save_patch_images(
                patient,
                output_directory,
            )

            PatchVisualizer._save_annotation_patches(
                patient,
                output_directory,
            )

    # ==========================================================
    # Working Image
    # ==========================================================

    @staticmethod
    def _save_working_image(
        patient,
        output_directory: Path,
    ):

        cv2.imwrite(
            str(output_directory / "working_image.png"),
            patient.working_image,
        )

    # ==========================================================
    # Working Mask
    # ==========================================================

    @staticmethod
    def _save_working_mask(
        patient,
        output_directory: Path,
    ):

        cv2.imwrite(
            str(output_directory / "working_mask.png"),
            patient.working_mask,
        )

    # ==========================================================
    # Tissue Mask
    # ==========================================================

    @staticmethod
    def _save_tissue_mask(
        patient,
        output_directory: Path,
    ):

        if patient.tissue_binary_mask is None:
            return

        mask = (
            patient.tissue_binary_mask.astype(np.uint8)
            * 255
        )

        cv2.imwrite(
            str(output_directory / "tissue_mask.png"),
            mask,
        )

    # ==========================================================
    # Patch Overlay
    # ==========================================================

    @staticmethod
    def _save_patch_overlay(
        patient,
        output_directory: Path,
    ):

        image = patient.working_image.copy()

        for patch in patient.patches:

            coordinate = patch.coordinate

            x = coordinate.x
            y = coordinate.y
            w = coordinate.width
            h = coordinate.height

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        cv2.imwrite(
            str(output_directory / "patch_overlay.png"),
            image,
        )

    # ==========================================================
    # Image Patches
    # ==========================================================

    @staticmethod
    def _save_patch_images(
        patient,
        output_directory: Path,
    ):

        image_directory = (
            output_directory /
            "image_patches"
        )

        image_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for patch in patient.patches:

            filename = (
                image_directory /
                patch.metadata.image_filename
            )

            cv2.imwrite(
                str(filename),
                patch.image_patch,
            )

    # ==========================================================
    # Annotation Patches
    # ==========================================================

    @staticmethod
    def _save_annotation_patches(
        patient,
        output_directory: Path,
    ):

        annotation_directory = (
            output_directory /
            "annotation_patches"
        )

        annotation_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for patch in patient.patches:

            filename = (
                annotation_directory /
                patch.metadata.annotation_filename
            )

            cv2.imwrite(
                str(filename),
                patch.annotation_patch,
            )