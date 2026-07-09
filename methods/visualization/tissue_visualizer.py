from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data.patient import Patient


class TissueVisualizer:
    """
    Creates visualizations for tissue detection.
    """

    @staticmethod
    def save(
        patient: Patient,
        output_dir: str | Path,
    ) -> None:

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # image = patient.image.copy()
        image = patient.working_image.copy()

        # --------------------------------------------
        # Original Image
        # --------------------------------------------

        cv2.imwrite(
            str(output_dir / "01_original.png"),
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            ),
        )

        # --------------------------------------------
        # Binary Mask
        # --------------------------------------------

        cv2.imwrite(
            str(output_dir / "02_binary_mask.png"),
            patient.tissue_binary_mask,
        )

        # --------------------------------------------
        # Overlay
        # --------------------------------------------

        overlay = image.copy()

        overlay[
            patient.tissue_binary_mask > 0
        ] = (
            0.6
            * overlay[
                patient.tissue_binary_mask > 0
            ]
            + 0.4 * np.array([255, 0, 0])
        ).astype(np.uint8)

        cv2.imwrite(
            str(output_dir / "03_overlay.png"),
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR,
            ),
        )

        # --------------------------------------------
        # Bounding Boxes
        # --------------------------------------------

        boxes = image.copy()

        for region in patient.tissue_regions:

            cv2.rectangle(
                boxes,
                (region.x, region.y),
                (
                    region.x + region.width,
                    region.y + region.height,
                ),
                (0, 255, 0),
                2,
            )

        cv2.imwrite(
            str(output_dir / "04_bounding_boxes.png"),
            cv2.cvtColor(
                boxes,
                cv2.COLOR_RGB2BGR,
            ),
        )

        # --------------------------------------------
        # Connected Components
        # --------------------------------------------

        colored = np.zeros(
            (
                patient.tissue_binary_mask.shape[0],
                patient.tissue_binary_mask.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        rng = np.random.default_rng(42)

        for region in patient.tissue_regions:

            color = tuple(
                int(v)
                for v in rng.integers(
                    0,
                    255,
                    3,
                )
            )

            colored[
                region.mask.astype(bool)
            ] = color

        cv2.imwrite(
            str(output_dir / "05_regions.png"),
            cv2.cvtColor(
                colored,
                cv2.COLOR_RGB2BGR,
            ),
        )