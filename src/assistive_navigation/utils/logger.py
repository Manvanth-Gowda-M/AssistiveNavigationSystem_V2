"""
Logger Module
=============
Configures and provides a consistent logging interface for all modules.

Phase: 1 (skeleton — full implementation in Phase 13)
Status: Functional basic logger

Usage:
    from assistive_navigation.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Camera opened successfully.")
    logger.warning("Low confidence detection: class=toothbrush, conf=0.38")
    logger.error("Model file not found: models/yolo11n.onnx")
"""

import logging
import logging.handlers
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a named logger with console output.

    Args:
        name: Logger name (use __name__ to get the module's name).
        level: Log level string: DEBUG, INFO, WARNING, ERROR.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler — writes to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Full file logging (rotating) will be configured in Phase 13
# when the config system is fully integrated at startup.
