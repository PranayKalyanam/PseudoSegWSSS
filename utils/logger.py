"""
logger.py

Central logging utility for the HistoGraphWSL preprocessing pipeline.


Features of this module
-------------------------

1. Centralized logging across the entire project.
2. Simultaneous console and file logging.
3. Automatic creation of the log directory.
4. Prevention of duplicate log messages when the logger is initialized multiple times.
5. Configurable logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).
6. UTF-8 encoded log files for compatibility across platforms.
7. Proper cleanup of file handlers to avoid resource leaks.

Example
-------
from utils.logger import get_logger

logger = get_logger("PatchExtractor")

logger.info("Patch extraction started.")
logger.warning("Mask not found.")
logger.error("Unable to open WSI.")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(name)s] "
    "%(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _create_formatter() -> logging.Formatter:
    """
    Create the formatter used by both console and file handlers.

    Returns
    -------
    logging.Formatter
    """

    return logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT
    )


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Create and configure a project logger.

    Parameters
    ----------
    name : str
        Name of the logger.

    log_dir : str, optional
        Directory where log files will be stored.

    level : str
        Logging level.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    formatter = _create_formatter()

    #
    # Console Handler
    #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))

    logger.addHandler(console_handler)

    #
    # File Handler
    #
    if log_dir is not None:

        log_directory = Path(log_dir)

        log_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logfile = log_directory / f"{name}.log"

        file_handler = logging.FileHandler(
            logfile,
            mode="a",
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        file_handler.setLevel(
            getattr(logging, level.upper())
        )

        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def change_log_level(
    logger: logging.Logger,
    level: str
) -> None:
    """
    Change logging level during runtime.

    Parameters
    ----------
    logger : logging.Logger

    level : str
    """

    numeric_level = getattr(logging, level.upper())

    logger.setLevel(numeric_level)

    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def close_logger(
    logger: logging.Logger
) -> None:
    """
    Close all logger handlers.

    This prevents file descriptor leaks when running
    long preprocessing jobs.

    Parameters
    ----------
    logger : logging.Logger
    """

    handlers = logger.handlers[:]

    for handler in handlers:

        handler.close()

        logger.removeHandler(handler)