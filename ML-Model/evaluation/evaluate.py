"""Evaluate a trained MedvisionAI classifier against one complete test split.

Run from any working directory:
``python path/to/evaluation/evaluate.py --dataset skin``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


# Enable direct execution without assuming the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config import CONFIG
from utils.dataset_loader import SUPPORTED_IMAGE_SUFFIXES, load_test_dataset
from utils.helpers import create_directories, load_model, setup_logging
from utils.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    classification_report,
    confusion_matrix,
    predict_dataset,
)
from utils.visualization import plot_confusion_matrix


LOGGER = logging.getLogger(__name__)


def _load_class_names(dataset_name: str) -> list[str]:
    """Load the configured JSON class-index mapping for a dataset domain."""
    mapping_path = CONFIG.class_names_path_for(dataset_name)
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Class-name mapping was not found: {mapping_path}. "
            "Export the training dataset's class_names to this configured path."
        )
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read class-name mapping: {mapping_path}") from error

    if isinstance(mapping, list):
        names = mapping
    elif isinstance(mapping, Mapping):
        try:
            names = [mapping[str(index)] for index in range(len(mapping))]
        except KeyError as error:
            raise ValueError(
                "Class-name mapping keys must be consecutive indices starting at zero."
            ) from error
    else:
        raise ValueError("Class-name mapping must be a JSON list or indexed JSON object.")

    if len(names) < 2 or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Class-name mapping must contain at least two non-empty class names.")
    return list(names)


def _validate_model_output(model: tf.keras.Model, class_names: Sequence[str]) -> None:
    """Ensure model probability outputs match the configured class mapping."""
    output_shape = model.output_shape
    if isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError("The model must expose one two-dimensional probability output.")
    if output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output count does not match the class mapping "
            f"({output_shape[-1]} outputs, {len(class_names)} class names)."
        )


def _test_image_paths(dataset_name: str, expected_labels: np.ndarray) -> list[Path]:
    """Return test images in the same stable class/file ordering as Keras loader.

    The dataset loader uses Keras' directory loader with alphabetical inferred
    classes and no test shuffle. The label check below prevents writing a CSV
    with an incorrect image-to-prediction pairing if that ordering changes.
    """
    test_directory = CONFIG.split_dir(dataset_name, "test")
    if not test_directory.is_dir():
        raise FileNotFoundError(f"Test dataset directory was not found: {test_directory}")

    class_directories = sorted(
        directory for directory in test_directory.iterdir() if directory.is_dir()
    )
    paths: list[Path] = []
    inferred_labels: list[int] = []
    for class_index, class_directory in enumerate(class_directories):
        class_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        paths.extend(class_paths)
        inferred_labels.extend([class_index] * len(class_paths))

    if len(paths) != len(expected_labels):
        raise RuntimeError(
            "The discovered test image count does not match generated predictions."
        )
    if not np.array_equal(np.asarray(inferred_labels, dtype=np.int64), expected_labels):
        raise RuntimeError(
            "Test image ordering does not match the dataset-loader label ordering; "
            "predictions.csv was not written to avoid incorrect image associations."
        )
    return paths


def _json_safe(value: Any) -> Any:
    """Convert NumPy values recursively to JSON-serializable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON results with consistent error reporting."""
    try:
        path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unable to write JSON output: {path}") from error


def _report_text(report: Mapping[str, Any]) -> str:
    """Format a Scikit-learn dictionary report as readable plain text."""
    lines = ["Class                         Precision    Recall  F1-score   Support"]
    for name, values in report.items():
        if not isinstance(values, Mapping):
            continue
        lines.append(
            f"{name:<28} {values.get('precision', 0.0):>9.4f} "
            f"{values.get('recall', 0.0):>9.4f} {values.get('f1-score', 0.0):>9.4f} "
            f"{values.get('support', 0):>9.0f}"
        )
    accuracy = report.get("accuracy")
    if isinstance(accuracy, (int, float)):
        lines.extend(("", f"Accuracy: {accuracy:.4f}"))
    return "\n".join(lines) + "\n"


def _write_predictions(
    path: Path,
    image_paths: Sequence[Path],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Write one complete-test-set prediction row for every source image."""
    try:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "image_path",
                    "true_class",
                    "predicted_class",
                    "predicted_probability",
                    "correct",
                ),
            )
            writer.writeheader()
            for image_path, true_label, predicted_label, probability in zip(
                image_paths,
                true_labels,
                predicted_labels,
                probabilities[np.arange(len(predicted_labels)), predicted_labels],
                strict=True,
            ):
                writer.writerow(
                    {
                        "image_path": str(image_path),
                        "true_class": class_names[int(true_label)],
                        "predicted_class": class_names[int(predicted_label)],
                        "predicted_probability": f"{float(probability):.8f}",
                        "correct": bool(true_label == predicted_label),
                    }
                )
    except OSError as error:
        raise RuntimeError(f"Unable to write prediction CSV: {path}") from error


def evaluate_model(dataset_name: str) -> dict[str, Any]:
    """Evaluate the configured model on one complete, unmodified test split.

    Args:
        dataset_name: One configured disease domain: skin, eye, or oral.

    Returns:
        Test metrics and the paths of generated evaluation artifacts.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    dataset_key = dataset_name.strip().lower()
    CONFIG.get_dataset(dataset_key)
    create_directories()
    logger = setup_logging()
    test_directory = CONFIG.split_dir(dataset_key, "test")
    model_path = CONFIG.model_path_for(dataset_key)
    class_mapping_path = CONFIG.class_names_path_for(dataset_key)
    output_directory = CONFIG.outputs_dir / "evaluation" / dataset_key

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(f"Unable to create evaluation output directory: {output_directory}") from error

    logger.info(
        "Evaluation started | dataset: %s | test path: %s | model: %s | checkpoint: %s",
        dataset_key,
        test_directory,
        model_path,
        CONFIG.checkpoint_path_for(dataset_key),
    )
    if not test_directory.is_dir():
        raise FileNotFoundError(f"Test dataset directory was not found: {test_directory}")
    class_names = _load_class_names(dataset_key)
    logger.info("Loaded class-name mapping: %s", class_mapping_path)
    model = load_model(model_path)
    _validate_model_output(model, class_names)
    test_dataset = load_test_dataset(dataset_key)
    if list(getattr(test_dataset, "class_names", [])) != class_names:
        raise ValueError(
            "Test dataset class directory order does not match the saved class-name mapping."
        )

    try:
        evaluation_values = model.evaluate(
            test_dataset,
            verbose=CONFIG.verbose,
            return_dict=True,
        )
        true_labels, predicted_labels, probabilities = predict_dataset(model, test_dataset)
    except (tf.errors.OpError, TypeError, ValueError, RuntimeError) as error:
        logger.exception("Test-set evaluation or prediction failed.")
        raise RuntimeError("Unable to evaluate the configured model on the test dataset.") from error

    if "loss" not in evaluation_values:
        raise RuntimeError("The loaded model did not return a test loss during evaluation.")
    report = classification_report(true_labels, predicted_labels, class_names)
    matrix = confusion_matrix(true_labels, predicted_labels)
    metrics: dict[str, Any] = {
        "dataset": dataset_key,
        "model_path": str(model_path),
        "class_mapping_path": str(class_mapping_path),
        "test_loss": float(evaluation_values["loss"]),
        "test_accuracy": calculate_accuracy(true_labels, predicted_labels),
        "precision": calculate_precision(true_labels, predicted_labels, average="macro"),
        "recall": calculate_recall(true_labels, predicted_labels, average="macro"),
        "f1_score": calculate_f1_score(true_labels, predicted_labels, average="macro"),
        "macro_f1": calculate_f1_score(true_labels, predicted_labels, average="macro"),
        "weighted_f1": calculate_f1_score(true_labels, predicted_labels, average="weighted"),
        "test_samples": int(len(true_labels)),
    }

    metrics_path = output_directory / "metrics.json"
    report_json_path = output_directory / "classification_report.json"
    report_text_path = output_directory / "classification_report.txt"
    predictions_path = output_directory / "predictions.csv"
    matrix_path = output_directory / "confusion_matrix.png"
    _write_json(metrics_path, metrics)
    _write_json(report_json_path, report)
    try:
        report_text_path.write_text(_report_text(report), encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Unable to write classification report: {report_text_path}") from error

    source_matrix_path = plot_confusion_matrix(
        matrix,
        class_names,
        dataset_key,
        model_path.stem,
    )
    try:
        source_matrix_path.replace(matrix_path)
    except OSError as error:
        raise RuntimeError(f"Unable to place confusion matrix at: {matrix_path}") from error

    image_paths = _test_image_paths(dataset_key, true_labels)
    _write_predictions(
        predictions_path,
        image_paths,
        true_labels,
        predicted_labels,
        probabilities,
        class_names,
    )
    results = {
        **metrics,
        "metrics_path": str(metrics_path),
        "classification_report_json_path": str(report_json_path),
        "classification_report_text_path": str(report_text_path),
        "confusion_matrix_path": str(matrix_path),
        "predictions_path": str(predictions_path),
    }
    logger.info(
        "Evaluation completed | loss: %.4f | accuracy: %.4f | precision: %.4f | "
        "recall: %.4f | macro F1: %.4f | weighted F1: %.4f",
        results["test_loss"],
        results["test_accuracy"],
        results["precision"],
        results["recall"],
        results["macro_f1"],
        results["weighted_f1"],
    )
    return results


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the single supported command-line dataset selection."""
    parser = argparse.ArgumentParser(description="Evaluate one trained disease classifier.")
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        required=True,
        help="Disease domain whose configured model and test split are evaluated.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run evaluation and return a conventional process exit code."""
    try:
        args = _parse_arguments(arguments)
        results = evaluate_model(args.dataset)
    except KeyboardInterrupt:
        LOGGER.warning("Evaluation cancelled by user.")
        return 130
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        LOGGER.error("Evaluation validation error: %s", error)
        print(f"Evaluation validation error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        LOGGER.exception("Evaluation failed.")
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Test loss: {results['test_loss']:.4f}\n"
        f"Test accuracy: {results['test_accuracy']:.4f}\n"
        f"Macro F1: {results['macro_f1']:.4f}\n"
        f"Weighted F1: {results['weighted_f1']:.4f}\n"
        f"Evaluation outputs: {CONFIG.outputs_dir / 'evaluation' / args.dataset}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
