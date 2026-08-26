"""Shared utilities for project setup, reproducibility, and model persistence."""

from __future__ import annotations

#=========================
# Remove Warnings
#=========================
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

#===========================
#===========================

import logging
import os
import platform
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from config import CONFIG


LOGGER = logging.getLogger(__name__)
_HANDLER_MARKER = "medical_ai_helper_handler"


def _create_directory(directory: Path) -> Path:
    """Create one directory, reporting permission and path errors clearly."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        LOGGER.exception("Permission denied while creating directory: %s", directory)
        raise PermissionError(f"Permission denied while creating: {directory}") from error
    except OSError as error:
        LOGGER.exception("Unable to create directory: %s", directory)
        raise OSError(f"Unable to create directory: {directory}") from error

    if not directory.is_dir():
        raise NotADirectoryError(f"Configured path is not a directory: {directory}")
    return directory


def _as_keras_path(filepath: str | Path, *, must_exist: bool) -> Path:
    """Validate and resolve a Keras model filepath.

    Args:
        filepath: Target model path.
        must_exist: Whether the model file must already exist.

    Raises:
        ValueError: If the path is invalid or does not use the ``.keras`` suffix.
        FileNotFoundError: If a required model file does not exist.
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError("filepath must be a string or pathlib.Path.")

    path = Path(filepath).expanduser()
    if path.name in {"", "."}:
        raise ValueError("filepath must name a .keras model file.")
    if path.suffix.lower() != ".keras":
        raise ValueError("Only the native Keras '.keras' model format is supported.")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Keras model file was not found: {path}")
    return path


def create_directories() -> dict[str, Path]:
    """Create and return the configured project artifact directories.

    Returns:
        Mapping of artifact purpose to its created directory path.
    """
    directories = {
        "models": CONFIG.models_dir,
        "saved_models": CONFIG.saved_models_dir,
        "checkpoints": CONFIG.checkpoints_dir,
        "logs": CONFIG.logs_dir,
        "outputs": CONFIG.outputs_dir,
    }
    created = {name: _create_directory(path) for name, path in directories.items()}
    LOGGER.info("Project directories are ready: %s", created)
    return created


def setup_logging(level: str | int = logging.INFO) -> logging.Logger:
    """Configure application logging to both the console and a dated log file.

    Existing handlers created by this function are reused, avoiding duplicate
    console/file messages when the setup is called more than once.

    Args:
        level: Standard logging level name or numeric level.

    Returns:
        The configured project logger.
    """
    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        if not isinstance(numeric_level, int):
            raise ValueError(f"Unsupported logging level: {level}")
    elif isinstance(level, int):
        numeric_level = level
    else:
        raise TypeError("level must be a logging level name or integer.")

    logs_dir = _create_directory(CONFIG.logs_dir)
    log_path = logs_dir / f"{date.today():%Y-%m-%d}_training.log"
    logger = logging.getLogger("medical_ai")
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARKER, True)

    existing_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, _HANDLER_MARKER, False)
    ]
    if not existing_handlers:
        for handler in handlers:
            logger.addHandler(handler)
    else:
        for handler in handlers:
            handler.close()
        for handler in existing_handlers:
            handler.setLevel(numeric_level)

    logger.info("Logging configured. File output: %s", log_path)
    return logger


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds for reproducibility.

    Args:
        seed: Non-negative integer seed shared by all supported libraries.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    LOGGER.info("Random seed set to %d for Python, NumPy, and TensorFlow.", seed)


def detect_gpu() -> list[str]:
    """Detect GPUs and print TensorFlow/CUDA runtime information.

    Returns:
        Names of the visible TensorFlow GPU devices. An empty list means CPU
        execution will be used.
    """
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        gpu_names = [device.name for device in gpu_devices]
        cuda_available = tf.test.is_built_with_cuda() and bool(gpu_devices)
    except (RuntimeError, tf.errors.OpError) as error:
        LOGGER.exception("GPU detection failed.")
        raise RuntimeError("Unable to inspect TensorFlow GPU devices.") from error

    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"Number of GPUs: {len(gpu_names)}")
    print(f"GPU names: {gpu_names or 'None'}")

    if gpu_names:
        LOGGER.info("Detected %d GPU(s): %s", len(gpu_names), gpu_names)
    else:
        LOGGER.info("No GPU detected; TensorFlow will use the CPU.")
    return gpu_names


def save_model(
    model: tf.keras.Model,
    filepath: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a Keras model safely in native ``.keras`` format.

    Args:
        model: Compiled or uncompiled Keras model to persist.
        filepath: Destination ending in ``.keras``.
        overwrite: Whether an existing model file may be replaced.

    Returns:
        The saved model path.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be an instance of tf.keras.Model.")

    path = _as_keras_path(filepath, must_exist=False)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Model already exists at {path}. Pass overwrite=True to replace it."
        )
    _create_directory(path.parent)

    try:
        model.save(str(path), overwrite=overwrite)
    except (OSError, ValueError, tf.errors.OpError) as error:
        LOGGER.exception("Failed to save model to %s", path)
        raise RuntimeError(f"Unable to save Keras model to: {path}") from error

    LOGGER.info("Model saved successfully: %s", path)
    return path


def load_model(
    filepath: str | Path,
    custom_objects: dict[str, Any] | None = None,
) -> tf.keras.Model:
    """Load a native Keras ``.keras`` model after validating its filepath."""
    path = _as_keras_path(filepath, must_exist=True)
    custom: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
    }
    if custom_objects:
        custom.update(custom_objects)
    try:
        model = tf.keras.models.load_model(str(path), custom_objects=custom)
    except (OSError, ValueError, TypeError, tf.errors.OpError) as error:
        LOGGER.exception("Failed to load model from %s", path)
        raise RuntimeError(f"Unable to load Keras model from: {path}") from error

    LOGGER.info("Model loaded successfully: %s", path)
    return model


def get_project_info() -> dict[str, Any]:
    """Return runtime and project details useful for logs and experiment records."""
    return {
        "project_root": str(CONFIG.project_root),
        "tensorflow_version": tf.__version__,
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "current_working_directory": str(Path.cwd()),
    }
