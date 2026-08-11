"""Publication-ready Matplotlib visualizations for model training and evaluation."""

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from config import CONFIG


LOGGER = logging.getLogger(__name__)
_SAFE_FILE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_FIGURE_SIZE = (12, 8)
_PLOT_DPI = 300


def _figure_size() -> tuple[float, float]:
    """Read the optional figure size from config, with a documented fallback."""
    figure_size = getattr(CONFIG, "figure_size", _DEFAULT_FIGURE_SIZE)
    if (
        not isinstance(figure_size, tuple)
        or len(figure_size) != 2
        or any(not isinstance(value, (int, float)) or value <= 0 for value in figure_size)
    ):
        raise ValueError("CONFIG.figure_size must be a two-item positive numeric tuple.")
    return float(figure_size[0]), float(figure_size[1])


def _safe_name(value: str, field_name: str) -> str:
    """Validate a value that will be included in an output filename."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    name = value.strip().lower().replace(" ", "_")
    if not _SAFE_FILE_COMPONENT.fullmatch(name):
        raise ValueError(
            f"{field_name} may contain only letters, numbers, underscores, and hyphens."
        )
    return name


def _output_path(filename: str) -> Path:
    """Create the configured output directory and return a safe figure path."""
    try:
        CONFIG.outputs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        LOGGER.exception("Permission denied while creating outputs directory.")
        raise PermissionError(
            f"Permission denied while creating output directory: {CONFIG.outputs_dir}"
        ) from error
    except OSError as error:
        LOGGER.exception("Unable to create outputs directory.")
        raise OSError(f"Unable to create output directory: {CONFIG.outputs_dir}") from error
    return CONFIG.outputs_dir / filename


def _save_figure(figure: plt.Figure, filename: str) -> Path:
    """Apply common layout, save a high-resolution figure, and close it."""
    path = _output_path(filename)
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=_PLOT_DPI, bbox_inches="tight")
    except (OSError, ValueError) as error:
        LOGGER.exception("Unable to save figure: %s", path)
        raise RuntimeError(f"Unable to save figure to: {path}") from error
    finally:
        plt.close(figure)
    LOGGER.info("Figure saved: %s", path)
    return path


def _history_values(history: Any, key: str) -> list[float]:
    """Extract a non-empty metric sequence from Keras History or a mapping."""
    values = getattr(history, "history", history)
    if not isinstance(values, Mapping) or key not in values:
        raise ValueError(f"history must contain a '{key}' metric sequence.")
    sequence = list(values[key])
    if not sequence:
        raise ValueError(f"history metric '{key}' must not be empty.")
    return [float(value) for value in sequence]


def _label_indices(labels: Any) -> np.ndarray:
    """Convert sparse or one-hot labels to a one-dimensional integer array."""
    values = np.asarray(labels)
    if values.size == 0:
        raise ValueError("Labels must not be empty.")
    if values.ndim == 1:
        indices = values
    elif values.ndim == 2 and values.shape[1] == 1:
        indices = values.ravel()
    elif values.ndim == 2:
        indices = np.argmax(values, axis=1)
    else:
        raise ValueError("Labels must be sparse vectors or one-hot matrices.")
    if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
        raise ValueError("Labels must contain finite integer class indices.")
    return indices.astype(np.int64, copy=False)


def _validate_probability_inputs(
    true_labels: Any,
    prediction_probabilities: Any,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels, class names, and multi-class probability dimensions."""
    y_true = _label_indices(true_labels)
    probabilities = np.asarray(prediction_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("prediction_probabilities must be a non-empty two-dimensional array.")
    if probabilities.shape[0] != len(y_true):
        raise ValueError("true_labels and prediction_probabilities have different lengths.")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Number of probability columns must equal number of class_names.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("prediction_probabilities must contain only finite values.")
    if np.any(y_true < 0) or np.any(y_true >= probabilities.shape[1]):
        raise ValueError("Labels must reference valid probability columns.")
    return y_true, probabilities


def plot_training_history(history: Any, dataset_name: str, model_name: str) -> Path:
    """Plot training/validation accuracy and loss from a Keras History object."""
    dataset_key = _safe_name(dataset_name, "dataset_name")
    model_key = _safe_name(model_name, "model_name")
    accuracy = _history_values(history, "accuracy")
    validation_accuracy = _history_values(history, "val_accuracy")
    loss = _history_values(history, "loss")
    validation_loss = _history_values(history, "val_loss")
    if len({len(accuracy), len(validation_accuracy), len(loss), len(validation_loss)}) != 1:
        raise ValueError("All history metric sequences must have the same epoch count.")

    LOGGER.info("Creating training-history plot for %s/%s.", dataset_key, model_key)
    figure, axes = plt.subplots(1, 2, figsize=_figure_size())
    epochs = np.arange(1, len(accuracy) + 1)
    axes[0].plot(epochs, accuracy, label="Training Accuracy", marker="o")
    axes[0].plot(epochs, validation_accuracy, label="Validation Accuracy", marker="o")
    axes[0].set(title="Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(epochs, loss, label="Training Loss", marker="o")
    axes[1].plot(epochs, validation_loss, label="Validation Loss", marker="o")
    axes[1].set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()
    return _save_figure(figure, f"{dataset_key}_{model_key}_training_history.png")


def plot_confusion_matrix(
    matrix: Any,
    class_names: Sequence[str],
    dataset_name: str,
    model_name: str,
) -> Path:
    """Plot an annotated multi-class confusion matrix."""
    dataset_key = _safe_name(dataset_name, "dataset_name")
    model_key = _safe_name(model_name, "model_name")
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
        raise ValueError("confusion_matrix must be a non-empty square two-dimensional array.")
    if values.shape[0] != len(class_names):
        raise ValueError("class_names length must match confusion_matrix dimensions.")

    LOGGER.info("Creating confusion-matrix plot for %s/%s.", dataset_key, model_key)
    figure, axis = plt.subplots(figsize=_figure_size())
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Images")
    positions = np.arange(len(class_names))
    axis.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=positions,
        yticks=positions,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = values.max() / 2 if values.size else 0
    for row, column in np.ndindex(values.shape):
        axis.text(
            column,
            row,
            str(values[row, column]),
            ha="center",
            va="center",
            color="white" if values[row, column] > threshold else "black",
        )
    return _save_figure(figure, f"{dataset_key}_{model_key}_confusion_matrix.png")


def plot_roc_curve(
    true_labels: Any,
    prediction_probabilities: Any,
    class_names: Sequence[str],
    dataset_name: str = "evaluation",
    model_name: str = "model",
) -> Path:
    """Plot per-class, macro-average, and micro-average One-vs-Rest ROC curves."""
    y_true, probabilities = _validate_probability_inputs(
        true_labels, prediction_probabilities, class_names
    )
    dataset_key = _safe_name(dataset_name, "dataset_name")
    model_key = _safe_name(model_name, "model_name")
    binary_labels = np.eye(len(class_names), dtype=int)[y_true]
    if any(np.unique(binary_labels[:, index]).size != 2 for index in range(len(class_names))):
        raise ValueError("ROC curves require both positive and negative samples per class.")
    figure, axis = plt.subplots(figsize=_figure_size())
    all_fpr = np.linspace(0, 1, 101)
    mean_tpr = np.zeros_like(all_fpr)

    try:
        for index, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(binary_labels[:, index], probabilities[:, index])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            axis.plot(fpr, tpr, label=f"{class_name} (AUC = {auc(fpr, tpr):.3f})")
        mean_tpr /= len(class_names)
        micro_fpr, micro_tpr, _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
    except ValueError as error:
        raise ValueError("ROC curves require both positive and negative samples per class.") from error

    axis.plot(all_fpr, mean_tpr, "--", linewidth=2, label=f"Macro Average (AUC = {auc(all_fpr, mean_tpr):.3f})")
    axis.plot(micro_fpr, micro_tpr, ":", linewidth=2, label=f"Micro Average (AUC = {auc(micro_fpr, micro_tpr):.3f})")
    axis.plot([0, 1], [0, 1], "k--", label="Chance")
    axis.set(title="Multi-class ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(loc="lower right", fontsize="small")
    return _save_figure(figure, f"{dataset_key}_{model_key}_roc_curve.png")


def plot_precision_recall_curve(
    true_labels: Any,
    prediction_probabilities: Any,
    class_names: Sequence[str],
    dataset_name: str = "evaluation",
    model_name: str = "model",
) -> Path:
    """Plot per-class One-vs-Rest precision-recall curves."""
    y_true, probabilities = _validate_probability_inputs(
        true_labels, prediction_probabilities, class_names
    )
    dataset_key = _safe_name(dataset_name, "dataset_name")
    model_key = _safe_name(model_name, "model_name")
    figure, axis = plt.subplots(figsize=_figure_size())
    for index, class_name in enumerate(class_names):
        binary_labels = (y_true == index).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, probabilities[:, index])
        axis.plot(recall, precision, label=class_name)
    axis.set(title="Multi-class Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(loc="best", fontsize="small")
    return _save_figure(figure, f"{dataset_key}_{model_key}_precision_recall_curve.png")


def plot_class_distribution(
    dataset: tf.data.Dataset,
    dataset_name: str = "dataset",
) -> Path:
    """Plot image counts per class for a batched dataset-loader dataset."""
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError("dataset must be an instance of tf.data.Dataset.")
    class_names = list(getattr(dataset, "class_names", []))
    if not class_names:
        raise ValueError("dataset must expose non-empty class_names metadata.")

    counts = np.zeros(len(class_names), dtype=int)
    try:
        for _, labels in dataset:
            indices = _label_indices(labels)
            if np.any(indices >= len(class_names)):
                raise ValueError("A dataset label exceeds the available class names.")
            counts += np.bincount(indices, minlength=len(class_names))
    except (tf.errors.OpError, TypeError, ValueError) as error:
        LOGGER.exception("Unable to calculate class distribution.")
        raise RuntimeError("Unable to calculate dataset class distribution.") from error
    if not counts.sum():
        raise ValueError("dataset is empty; class distribution cannot be plotted.")

    dataset_key = _safe_name(dataset_name, "dataset_name")
    figure, axis = plt.subplots(figsize=_figure_size())
    bars = axis.bar(class_names, counts)
    axis.set(title="Class Distribution", xlabel="Class", ylabel="Number of Images")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    axis.bar_label(bars, padding=3)
    return _save_figure(figure, f"{dataset_key}_class_distribution.png")


def plot_sample_images(
    dataset: tf.data.Dataset,
    dataset_name: str = "dataset",
) -> Path:
    """Plot up to 16 randomly selected dataset images with their class labels."""
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError("dataset must be an instance of tf.data.Dataset.")
    class_names = list(getattr(dataset, "class_names", []))
    if not class_names:
        raise ValueError("dataset must expose non-empty class_names metadata.")

    images, labels = [], []
    for batch_images, batch_labels in dataset:
        images.extend(np.asarray(batch_images))
        labels.extend(_label_indices(batch_labels))
    if not images:
        raise ValueError("dataset is empty; sample images cannot be plotted.")

    sample_count = min(16, len(images))
    seed = getattr(CONFIG, "random_seed", None)
    sample_indices = np.random.default_rng(seed).choice(len(images), sample_count, replace=False)
    figure, axes = plt.subplots(4, 4, figsize=_figure_size())
    for axis in axes.ravel():
        axis.axis("off")
    for axis, index in zip(axes.ravel(), sample_indices, strict=False):
        label = int(labels[index])
        if label < 0 or label >= len(class_names):
            raise ValueError("A dataset label exceeds the available class names.")
        axis.imshow(np.asarray(images[index]).astype(np.uint8))
        axis.set_title(class_names[label], fontsize=9)
        axis.axis("off")
    figure.suptitle("Random Dataset Samples")
    dataset_key = _safe_name(dataset_name, "dataset_name")
    return _save_figure(figure, f"{dataset_key}_sample_images.png")


def plot_model_comparison(results: Mapping[str, Mapping[str, float]]) -> Path:
    """Plot six key metrics for MobileNetV2, EfficientNetB0, and ResNet50.

    ``results`` maps each model name to accuracy, precision, recall, F1 score,
    training time, and inference time values.
    """
    required_metrics = (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time",
        "inference_time",
    )
    if not results:
        raise ValueError("results must contain at least one model.")
    model_names = list(results)
    for model_name, result in results.items():
        missing = set(required_metrics) - set(result)
        if missing:
            raise ValueError(f"Results for {model_name} are missing: {sorted(missing)}.")

    figure, axes = plt.subplots(2, 3, figsize=_figure_size())
    for axis, metric_name in zip(axes.ravel(), required_metrics, strict=True):
        values = [float(results[model_name][metric_name]) for model_name in model_names]
        bars = axis.bar(model_names, values)
        axis.set(title=metric_name.replace("_", " ").title(), ylabel=metric_name.replace("_", " ").title())
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        axis.bar_label(bars, fmt="%.3f", padding=3)
        plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    LOGGER.info("Creating model-comparison plot for: %s", model_names)
    return _save_figure(figure, "model_comparison.png")
