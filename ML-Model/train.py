"""Train one selected medical-image classification model at a time."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import logging
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from config import CONFIG
from utils.callbacks import get_callbacks
from utils.dataset_loader import load_train_dataset, load_validation_dataset
from utils.helpers import (
    create_directories,
    detect_gpu,
    save_model,
    set_random_seed,
    setup_logging,
)
from utils import model_builder
from utils.model_builder import print_model_summary
from utils.visualization import plot_training_history


LOGGER = logging.getLogger(__name__)
_REQUIRED_CONFIG_ATTRIBUTES = (
    "random_seed",
    "epochs",
    "batch_size",
    "initial_learning_rate",
    "verbose",
)


def _require_config_value(attribute: str) -> Any:
    """Return a required setting from ``CONFIG`` or raise a clear error."""
    try:
        value = getattr(CONFIG, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"Missing required configuration: CONFIG.{attribute}."
        ) from error
    if value is None:
        raise ValueError(f"CONFIG.{attribute} must not be None.")
    return value


def _validate_configuration(
    dataset_name: str,
    model_name: str,
) -> tuple[str, str, int]:
    """Validate CLI selections and required training settings from config.py."""
    for attribute in _REQUIRED_CONFIG_ATTRIBUTES:
        _require_config_value(attribute)

    dataset_name = dataset_name.strip().lower()
    model_name = model_name.strip().lower()
    epochs = _require_config_value("epochs")
    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty dataset identifier.")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("CONFIG.epochs must be a positive integer.")
    if not isinstance(_require_config_value("batch_size"), int):
        raise ValueError("CONFIG.batch_size must be an integer.")

    # get_dataset validates the selected skin, eye, or oral domain.
    CONFIG.get_dataset(dataset_name)
    CONFIG.validate_dataset(dataset_name)
    return dataset_name, model_name, epochs


def _final_history_value(history: tf.keras.callbacks.History, metric_name: str) -> float:
    """Return the final value of a Keras history metric."""
    values = history.history.get(metric_name)
    if not values:
        raise ValueError(f"Training history does not contain '{metric_name}'.")
    return float(values[-1])


class _ModelBuilderConfigProxy:
    """Supply model-builder compile defaults without changing ``config.py``.

    ``model_builder.build_model`` currently compiles during construction and
    expects these optional settings. The training pipeline recompiles the
    returned model immediately using its own configurable defaults below.
    """

    def __getattr__(self, attribute: str) -> Any:
        defaults: dict[str, Any] = {
            "optimizer": "adam",
            "loss_function": "sparse_categorical_crossentropy",
            "metrics": ("accuracy",),
        }
        if attribute in defaults:
            return defaults[attribute]
        return getattr(CONFIG, attribute)


def _build_model(model_name: str, num_classes: int) -> tf.keras.Model:
    """Build a transfer model without requiring optional config compile fields."""
    original_config = model_builder.CONFIG
    model_builder.CONFIG = _ModelBuilderConfigProxy()
    try:
        return model_builder.build_model(model_name, num_classes)
    finally:
        model_builder.CONFIG = original_config


def _optional_config_value(attribute: str, default: Any) -> Any:
    """Read an optional ``CONFIG`` setting while retaining a safe default."""
    return getattr(CONFIG, attribute, default)


def _compile_model(model: tf.keras.Model) -> None:
    """Compile a model with defaults or optional future config overrides.

    Defaults are Adam with ``CONFIG.initial_learning_rate``, sparse categorical
    cross-entropy, and accuracy/precision/recall metrics.
    """
    learning_rate = float(_require_config_value("initial_learning_rate"))
    configured_optimizer = _optional_config_value("optimizer", None)
    if configured_optimizer is None:
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
    else:
        try:
            optimizer = tf.keras.optimizers.get(configured_optimizer)
        except (TypeError, ValueError) as error:
            raise ValueError("CONFIG.optimizer is not a valid Keras optimizer.") from error

    loss = _optional_config_value(
        "loss_function",
        _optional_config_value("loss", "sparse_categorical_crossentropy"),
    )
    metrics = _optional_config_value(
        "metrics",
        [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    if not isinstance(metrics, (list, tuple)) or not metrics:
        raise ValueError("CONFIG.metrics must be a non-empty list or tuple when provided.")

    try:
        model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))
    except (TypeError, ValueError, tf.errors.OpError) as error:
        raise RuntimeError("Unable to compile the model with the selected settings.") from error
    LOGGER.info(
        "Model compiled | optimizer: %s | learning rate: %s | loss: %s | metrics: %s",
        optimizer.__class__.__name__,
        learning_rate,
        loss,
        [getattr(metric, "name", metric) for metric in metrics],
    )


def train(dataset_name: str, model_name: str) -> dict[str, float | Path | str]:
    """Run the complete configured training pipeline for one disease dataset.

    Returns:
        Training metadata, final validation metrics, duration, and model path.

    Raises:
        FileNotFoundError: If a configured dataset split or model path is missing.
        ValueError: If configuration or training artifacts are invalid.
        RuntimeError: If TensorFlow cannot train or save the model.
    """
    CONFIG.create_project_directories()
    create_directories()
    logger = setup_logging()
    dataset_name, model_name, epochs = _validate_configuration(dataset_name, model_name)
    logger.info("Training dataset selected: %s", dataset_name)
    logger.info("Model selected: %s", model_name)
    logger.info(
        "Training settings | image size: %s | epochs: %d | batch size: %s | "
        "learning rate: %s",
        CONFIG.image_size,
        epochs,
        _require_config_value("batch_size"),
        _require_config_value("initial_learning_rate"),
    )

    set_random_seed(int(_require_config_value("random_seed")))
    detect_gpu()
    logger.info(
        "Dataset paths | train: %s | validation: %s",
        CONFIG.split_dir(dataset_name, "train"),
        CONFIG.split_dir(dataset_name, "validation"),
    )
    train_dataset = load_train_dataset(dataset_name)
    validation_dataset = load_validation_dataset(dataset_name)
    class_names = list(getattr(train_dataset, "class_names", []))
    validation_class_names = list(getattr(validation_dataset, "class_names", []))
    if len(class_names) < 2:
        raise ValueError("Training dataset must contain at least two detected classes.")
    if class_names != validation_class_names:
        raise ValueError(
            "Training and validation class names differ. Ensure their class folders match."
        )
    logger.info("Detected %d classes: %s", len(class_names), class_names)

    model = _build_model(model_name, num_classes=len(class_names))
    _compile_model(model)
    print_model_summary(model)
    callbacks = get_callbacks(dataset_name, model_name)
    logger.info("Configured checkpoint path: %s", CONFIG.checkpoint_path_for(dataset_name))

    logger.info("Training started.")
    started_at = perf_counter()
    try:
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            # The tf.data loader already shuffles the training split.
            shuffle=False,
            verbose=int(_require_config_value("verbose")),
        )
    except (tf.errors.OpError, ValueError, RuntimeError) as error:
        logger.exception("TensorFlow training failed.")
        raise RuntimeError("Model training did not complete successfully.") from error
    training_time = perf_counter() - started_at

    # EarlyStopping restores the lowest-validation-loss weights before this save.
    model_path = CONFIG.model_path_for(dataset_name)
    try:
        save_model(model, model_path, overwrite=True)
    except (OSError, ValueError, RuntimeError) as error:
        logger.exception("Unable to save trained model.")
        raise RuntimeError(f"Unable to save final model to: {model_path}") from error

    plot_path = plot_training_history(history, dataset_name, model_name)
    final_accuracy = _final_history_value(history, "val_accuracy")
    final_loss = _final_history_value(history, "val_loss")
    results: dict[str, float | Path | str] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "training_time_seconds": training_time,
        "model_path": model_path,
        "history_plot_path": plot_path,
    }
    logger.info(
        "Training completed | validation accuracy: %.4f | validation loss: %.4f | "
        "training time: %.2f seconds",
        final_accuracy,
        final_loss,
        training_time,
    )
    logger.info("Final model saved: %s", model_path)
    logger.info("Training plot saved: %s", plot_path)
    print(
        f"Final validation accuracy: {final_accuracy:.4f}\n"
        f"Final validation loss: {final_loss:.4f}\n"
        f"Training time: {training_time:.2f} seconds\n"
        f"Model saved to: {model_path}"
    )
    return results


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the selected dataset and transfer-learning architecture."""
    parser = argparse.ArgumentParser(
        description="Train one medical image classifier using project configuration."
    )
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        default="skin",
        help="Disease dataset to train (default: skin).",
    )
    parser.add_argument(
        "--model",
        choices=("mobilenetv2", "efficientnetb0", "resnet50"),
        default=CONFIG.backbone,
        help="Transfer-learning backbone (default: CONFIG.backbone).",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run parsed training selections and return an appropriate exit code."""
    try:
        args = _parse_arguments(arguments)
        train(args.dataset, args.model)
    except KeyboardInterrupt:
        LOGGER.warning("Training cancelled by user.")
        print("Training cancelled by user.")
        return 130
    except (AttributeError, FileNotFoundError, PermissionError, ValueError) as error:
        LOGGER.error("Configuration or filesystem error: %s", error)
        print(f"Training configuration error: {error}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError, tf.errors.OpError) as error:
        LOGGER.exception("Training pipeline failed.")
        print(f"Training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
