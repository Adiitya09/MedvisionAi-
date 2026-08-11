"""Reusable Keras training callbacks for medical image classifiers."""

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
import re
from pathlib import Path

import tensorflow as tf

from config import CONFIG


LOGGER = logging.getLogger(__name__)
_SAFE_FILE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_config_value(attribute: str) -> object:
    """Return a required configuration value or raise a clear exception."""
    try:
        value = getattr(CONFIG, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"Missing required configuration: CONFIG.{attribute}."
        ) from error

    if value is None:
        raise ValueError(f"CONFIG.{attribute} must not be None.")
    return value


def _validate_file_component(value: str, field_name: str) -> str:
    """Validate a name that will be incorporated into an output filename."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    normalised = value.strip().lower().replace(" ", "_")
    if not _SAFE_FILE_COMPONENT.fullmatch(normalised):
        raise ValueError(
            f"{field_name} may contain only letters, numbers, underscores, and hyphens."
        )
    return normalised


def _create_directory(directory: Path) -> Path:
    """Create an artifact directory and surface filesystem failures clearly."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        raise PermissionError(f"Permission denied while creating directory: {directory}") from error
    except OSError as error:
        raise OSError(f"Unable to create directory: {directory}") from error

    if not directory.is_dir():
        raise NotADirectoryError(f"Configured path is not a directory: {directory}")
    return directory


def _validate_callback_settings() -> tuple[int, int, float, float]:
    """Read and validate callback values supplied by ``config.py``."""
    early_stopping_patience = _require_config_value("early_stopping_patience")
    reduce_lr_patience = _require_config_value("reduce_lr_patience")
    reduce_lr_factor = _require_config_value("reduce_lr_factor")
    min_learning_rate = _require_config_value("min_learning_rate")

    if (
        isinstance(early_stopping_patience, bool)
        or not isinstance(early_stopping_patience, int)
        or early_stopping_patience < 0
    ):
        raise ValueError("CONFIG.early_stopping_patience must be a non-negative integer.")
    if (
        isinstance(reduce_lr_patience, bool)
        or not isinstance(reduce_lr_patience, int)
        or reduce_lr_patience < 0
    ):
        raise ValueError("CONFIG.reduce_lr_patience must be a non-negative integer.")
    if not isinstance(reduce_lr_factor, (int, float)) or not 0 < reduce_lr_factor < 1:
        raise ValueError("CONFIG.reduce_lr_factor must be a number between zero and one.")
    if not isinstance(min_learning_rate, (int, float)) or min_learning_rate < 0:
        raise ValueError("CONFIG.min_learning_rate must be a non-negative number.")

    return (
        early_stopping_patience,
        reduce_lr_patience,
        float(reduce_lr_factor),
        float(min_learning_rate),
    )


def get_callbacks(dataset_name: str, model_name: str) -> list[tf.keras.callbacks.Callback]:
    """Create the standard callbacks for one training run.

    Args:
        dataset_name: Supported domain identifier: ``skin``, ``eye``, or ``oral``.
        model_name: Architecture identifier included in the checkpoint filename.

    Returns:
        Early stopping, checkpoint, learning-rate, TensorBoard, and CSV callbacks.

    Raises:
        ValueError: If supplied names or callback settings are invalid.
        OSError: If a configured output directory cannot be created or accessed.
    """
    dataset_key = _validate_file_component(dataset_name, "dataset_name")
    # This also verifies that the domain is declared in config.py.
    CONFIG.get_dataset(dataset_key)
    model_key = _validate_file_component(model_name, "model_name")
    early_patience, reduce_patience, reduce_factor, min_learning_rate = (
        _validate_callback_settings()
    )

    checkpoints_dir = _create_directory(Path(_require_config_value("checkpoints_dir")))
    logs_dir = _create_directory(Path(_require_config_value("logs_dir")))
    outputs_dir = _create_directory(Path(_require_config_value("outputs_dir")))

    checkpoint_path = checkpoints_dir / f"{dataset_key}_{model_key}_best.keras"
    tensorboard_path = _create_directory(logs_dir / dataset_key)
    csv_path = outputs_dir / f"{dataset_key}_training_log.csv"

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_patience,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_factor,
            patience=reduce_patience,
            min_lr=min_learning_rate,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_path),
            histogram_freq=1,
            write_graph=True,
            profile_batch=0,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(csv_path),
            separator=",",
            append=False,
        ),
    ]

    LOGGER.info("Created %d callbacks for %s/%s training.", len(callbacks), dataset_key, model_key)
    LOGGER.info("ModelCheckpoint path: %s", checkpoint_path)
    LOGGER.info("TensorBoard path: %s", tensorboard_path)
    LOGGER.info("CSV training log path: %s", csv_path)
    return callbacks
