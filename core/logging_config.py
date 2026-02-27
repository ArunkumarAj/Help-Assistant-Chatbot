"""Centralized logging setup. Logs go to both app.log and terminal."""
import logging
import os
import sys

from core.config import settings


def setup_logging(log_file_path: str | None = None) -> None:
    """Configure logging to file and console. Creates log directory if needed. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return
    path = log_file_path or settings.log_file_path
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    level = logging.INFO
    # File handler (app.log)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    # Console handler (terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt))
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
