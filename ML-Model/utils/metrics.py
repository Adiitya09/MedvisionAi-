"""Reusable multi-class evaluation utilities for TensorFlow/Keras models."""

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
from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score


LOGGER = logging.getLogger(__name__)
_VALID_AVERAGES = frozenset({"macro", "micro", "weighted"})


def _labels_to_indices(labels: Any) -> np.ndarray:
    """Convert sparse or one-hot encoded labels to a one-dimensional array."""
    values = np.asarray(labels)
    if values.size == 0:
        raise ValueError("Labels must not be empty.")
    if values.ndim == 1:
        indices = values
    elif values.ndim == 2 and values.shape[1] == 1:
        indices = values.ravel()
    elif values.ndim == 2 and values.shape[1] > 1:
        indices = np.argmax(values, axis=1)
    else:
        raise ValueError("Labels must be sparse vectors or two-dimensional one-hot arrays.")

    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
            raise ValueError("Labels must contain finite integer class indices.")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0):
        raise ValueError("Labels must contain non-negative class indices.")
    return indices


def _validate_label_pairs(
    true_labels: Any,
    predicted_labels: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equally sized label arrays and return integer class indices."""
    y_true = _labels_to_indices(true_labels)
    y_pred = _labels_to_indices(predicted_labels)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("true_labels and predicted_labels must have the same length.")
    return y_true, y_pred


def _validate_probabilities(
    true_labels: Any,
    probabilities: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate probability scores against true class indices."""
    y_true = _labels_to_indices(true_labels)
    y_probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_probabilities.ndim != 2 or y_probabilities.shape[1] < 2:
        raise ValueError(
            "prediction_probabilities must have shape (samples, at least two classes)."
        )
    if y_probabilities.shape[0] != y_true.shape[0]:
        raise ValueError("true_labels and prediction_probabilities must have the same length.")
    if not np.all(np.isfinite(y_probabilities)):
        raise ValueError("prediction_probabilities must contain only finite values.")
    if np.max(y_true) >= y_probabilities.shape[1]:
        raise ValueError("A true label does not have a matching probability column.")
    return y_true, y_probabilities


def _validate_average(average: str) -> str:
    """Validate a Scikit-learn multi-class averaging strategy."""
    if average not in _VALID_AVERAGES:
        supported = ", ".join(sorted(_VALID_AVERAGES))
        raise ValueError(f"average must be one of: {supported}.")
    return average


def predict_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict labels and probabilities for every batch in a TensorFlow dataset.

    Args:
        model: A Keras multi-class classifier with softmax probability output.
        dataset: Batched ``(images, labels)`` TensorFlow dataset.

    Returns:
        True labels, predicted labels, and class probabilities respectively.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be an instance of tf.keras.Model.")
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError("dataset must be an instance of tf.data.Dataset.")

    true_label_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    LOGGER.info("Generating predictions for evaluation.")

    try:
        for batch in dataset:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("dataset must yield (images, labels) batches.")
            images, labels = batch
            probabilities = np.asarray(model(images, training=False))
            true_label_batches.append(_labels_to_indices(labels))
            probability_batches.append(probabilities)
    except Exception as error:
        LOGGER.exception("Prediction failed during dataset evaluation.")
        raise RuntimeError("Unable to generate predictions for the supplied dataset.") from error

    if not true_label_batches:
        raise ValueError("dataset is empty; evaluation requires at least one batch.")

    try:
        true_labels = np.concatenate(true_label_batches)
        prediction_probabilities = np.concatenate(probability_batches)
    except ValueError as error:
        raise ValueError("Prediction batches have incompatible dimensions.") from error
    true_labels, prediction_probabilities = _validate_probabilities(
        true_labels,
        prediction_probabilities,
    )
    predicted_labels = np.argmax(prediction_probabilities, axis=1).astype(np.int64)
    return true_labels, predicted_labels, prediction_probabilities


def calculate_accuracy(true_labels: Any, predicted_labels: Any) -> float:
    """Calculate classification accuracy."""
    y_true, y_pred = _validate_label_pairs(true_labels, predicted_labels)
    return float(sklearn_accuracy_score(y_true, y_pred))


def calculate_precision(
    true_labels: Any,
    predicted_labels: Any,
    average: str = "macro",
) -> float:
    """Calculate macro, micro, or weighted precision for multi-class labels."""
    y_true, y_pred = _validate_label_pairs(true_labels, predicted_labels)
    return float(
        sklearn_precision_score(
            y_true,
            y_pred,
            average=_validate_average(average),
            zero_division=0,
        )
    )


def calculate_recall(
    true_labels: Any,
    predicted_labels: Any,
    average: str = "macro",
) -> float:
    """Calculate macro, micro, or weighted recall for multi-class labels."""
    y_true, y_pred = _validate_label_pairs(true_labels, predicted_labels)
    return float(
        sklearn_recall_score(
            y_true,
            y_pred,
            average=_validate_average(average),
            zero_division=0,
        )
    )


def calculate_f1_score(
    true_labels: Any,
    predicted_labels: Any,
    average: str = "macro",
) -> float:
    """Calculate macro, micro, or weighted F1 score for multi-class labels."""
    y_true, y_pred = _validate_label_pairs(true_labels, predicted_labels)
    return float(
        sklearn_f1_score(
            y_true,
            y_pred,
            average=_validate_average(average),
            zero_division=0,
        )
    )


def classification_report(
    true_labels: Any,
    predicted_labels: Any,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return Scikit-learn's detailed classification report as a dictionary."""
    y_true, y_pred = _validate_label_pairs(true_labels, predicted_labels)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    if class_names is not None and len(class_names) <= int(labels.max()):
        raise ValueError("class_names does not contain every encountered class label.")

    target_names = (
        [class_names[label] for label in labels] if class_names is not None else None
    )
    return sklearn_classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix(true_labels: Any, predicted_labels: Any) -> np.ndarray:
    """Return the multi-class confusion matrix as a NumPy array."""
    y_true, y_pred = _validate_label_pairs(true_labels, predicted_labels)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    return np.asarray(sklearn_confusion_matrix(y_true, y_pred, labels=labels))


def roc_auc_score(
    true_labels: Any,
    prediction_probabilities: Any,
    average: str = "macro",
) -> float:
    """Calculate One-vs-Rest ROC AUC for binary or multi-class probabilities."""
    y_true, y_probabilities = _validate_probabilities(true_labels, prediction_probabilities)
    average = _validate_average(average)
    try:
        if y_probabilities.shape[1] == 2:
            return float(sklearn_roc_auc_score(y_true, y_probabilities[:, 1]))
        return float(
            sklearn_roc_auc_score(
                y_true,
                y_probabilities,
                multi_class="ovr",
                average=average,
                labels=np.arange(y_probabilities.shape[1]),
            )
        )
    except ValueError as error:
        raise ValueError(
            "ROC AUC requires valid probabilities and at least one true sample "
            "from every evaluated class."
        ) from error


def precision_recall_curve(
    true_labels: Any,
    prediction_probabilities: Any,
) -> dict[int, dict[str, np.ndarray]]:
    """Return One-vs-Rest precision-recall arrays for each class.

    The result maps each class index to ``precision``, ``recall``, and
    ``thresholds`` NumPy arrays, ready for plotting.
    """
    y_true, y_probabilities = _validate_probabilities(true_labels, prediction_probabilities)
    curves: dict[int, dict[str, np.ndarray]] = {}
    for class_index in range(y_probabilities.shape[1]):
        binary_labels = (y_true == class_index).astype(np.int64)
        precision, recall, thresholds = sklearn_precision_recall_curve(
            binary_labels,
            y_probabilities[:, class_index],
        )
        curves[class_index] = {
            "precision": np.asarray(precision),
            "recall": np.asarray(recall),
            "thresholds": np.asarray(thresholds),
        }
    return curves


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Sequence[str] | None = None,
    average: str = "macro",
) -> dict[str, Any]:
    """Evaluate a model and return the requested multi-class metric bundle."""
    LOGGER.info("Model evaluation started.")
    true_labels, predicted_labels, prediction_probabilities = predict_dataset(model, dataset)
    metrics = {
        "accuracy": calculate_accuracy(true_labels, predicted_labels),
        "precision": calculate_precision(true_labels, predicted_labels, average),
        "recall": calculate_recall(true_labels, predicted_labels, average),
        "f1_score": calculate_f1_score(true_labels, predicted_labels, average),
        "roc_auc": roc_auc_score(true_labels, prediction_probabilities, average),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            class_names,
        ),
    }
    LOGGER.info(
        "Model evaluation completed | accuracy: %.4f | precision: %.4f | "
        "recall: %.4f | f1: %.4f | roc_auc: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics
