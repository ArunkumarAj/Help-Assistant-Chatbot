"""Centralized logging setup."""
import logging
import os

from core.config import settings


def setup_logging(log_file_path: str | None = None) -> None:
    """Configure logging to file. Creates log directory if needed."""
    path = log_file_path or settings.log_file_path
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
