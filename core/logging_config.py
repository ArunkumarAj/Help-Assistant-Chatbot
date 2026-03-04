"""
Centralized logging setup: logs go to both a file (app.log) and the terminal.

Call setup_logging() once at app startup. Creating the log directory and
adding handlers is idempotent (skipped if root logger already has handlers).
"""
import logging
import os
import sys

from core.config import settings


def setup_logging(log_file_path: str | None = None) -> None:
    """
    Configure the root logger to write to a file and to stdout.
    Creates the log directory if needed. Safe to call multiple times; after the
    first call, handlers are left as-is.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    path = log_file_path or settings.log_file_path
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    level = logging.INFO

    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
