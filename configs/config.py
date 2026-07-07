# Module 1
# configs/config.py

# This is the foundation of the entire project.

# It will:

# define every command-line argument,
# validate arguments,
# create required folders,
# return one configuration object (args) used throughout the project.

# The rest of the modules will never define their own arguments.
"""
config.py

Central configuration module for the HistoGraphWSL preprocessing pipeline.

Responsibilities
----------------
1. Parse command-line arguments.
2. Validate user inputs.
3. Create required output directories.
4. Return a unified configuration object.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


SUPPORTED_EXTENSIONS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
)


SUPPORTED_TISSUE_METHODS = (
    "otsu",
    "double_pass",
)


def positive_int(value: str) -> int:
    """
    Ensure integer is positive.
    """
    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{ivalue} is not a positive integer."
        )

    return ivalue


def positive_float(value: str) -> float:
    """
    Ensure float is positive.
    """

    fvalue = float(value)

    if fvalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{fvalue} must be greater than zero."
        )

    return fvalue


def probability(value: str) -> float:
    """
    Validate probability.
    """

    p = float(value)

    if p < 0 or p > 1:
        raise argparse.ArgumentTypeError(
            "Probability must lie in [0,1]."
        )

    return p


def build_parser() -> argparse.ArgumentParser:
    """
    Create argument parser.
    """

    parser = argparse.ArgumentParser(
        description="HistoGraphWSL Preprocessing Pipeline"
    )

    #######################################################
    # Dataset
    #######################################################

    parser.add_argument(
        "--dataset",
        type=str,
        default="BCSS",
        help="Dataset name."
    )

    parser.add_argument(
        "--wsi_dir",
        type=str,
        default="datasets/BCSS/imgs",
        help="Directory containing whole slide images."
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        default="datasets/BCSS/masks",
        help="Directory containing segmentation masks."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Output directory."
    )

    #######################################################
    # Patch extraction
    #######################################################

    parser.add_argument(
        "--magnification",
        type=int,
        default=10,
        help="Magnification level."
    )

    parser.add_argument(
        "--patch_size",
        type=positive_int,
        default=224,
        help="Patch size."
    )

    parser.add_argument(
        "--overlap",
        type=probability,
        default=0.50,
        help="Patch overlap."
    )

    #######################################################
    # Tissue detection
    #######################################################

    parser.add_argument(
        "--tissue_detector",
        default="otsu",
        choices=SUPPORTED_TISSUE_METHODS,
    )

    parser.add_argument(
        "--tissue_threshold",
        type=probability,
        default=0.75,
        help="Minimum tissue percentage."
    )

    #######################################################
    # Graph
    #######################################################

    parser.add_argument(
        "--graph_radius",
        type=positive_float,
        default=2.0,
    )

    #######################################################
    # Dataloader
    #######################################################

    parser.add_argument(
        "--batch_size",
        type=positive_int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=positive_int,
        default=4,
    )

    #######################################################
    # Randomness
    #######################################################

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    #######################################################
    # Logging
    #######################################################

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ]
    )

    #######################################################
    # Save options
    #######################################################

    parser.add_argument(
        "--save_patches",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_masks",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_graph",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_metadata",
        action="store_true",
        default=True,
    )

    return parser


def validate_paths(args: argparse.Namespace) -> None:
    """
    Validate dataset paths.
    """

    wsi_dir = Path(args.wsi_dir)

    if not wsi_dir.exists():
        raise FileNotFoundError(
            f"WSI directory not found:\n{wsi_dir}"
        )

    mask_dir = Path(args.mask_dir)

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask directory not found:\n{mask_dir}"
        )


def create_output_dirs(output_root: Path) -> None:
    """
    Create project output folders.
    """

    folders = [
        "patches",
        "masks",
        "labels",
        "graphs",
        "metadata",
        "visualizations",
        "logs",
    ]

    for folder in folders:
        (output_root / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


def get_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate configuration.
    """

    parser = build_parser()

    args = parser.parse_args(argv)

    validate_paths(args)

    create_output_dirs(Path(args.output_dir))

    return args