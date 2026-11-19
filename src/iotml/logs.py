"""One logger per script run: timestamped lines to the console and to logs/<name>_<time>.log."""

from __future__ import annotations

import logging
import time
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_dir: Path) -> logging.Logger:
    """Create a logger that writes to the console and to a new file in ``log_dir``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"{name}_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
