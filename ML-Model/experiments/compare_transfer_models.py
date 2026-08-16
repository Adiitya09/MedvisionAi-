"""Run a controlled transfer-learning backbone comparison for one dataset.

The primary selection metric is validation macro F1. Test metrics are reported
only after each independent model run and are never used to rank models.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


# Permit direct execution from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config import CONFIG
from utils import model_builder
from utils.callbacks import get_callbacks
from utils.dataset_loader import (
    load_test_dataset,
    load_train_dataset,
    load_validation_dataset,
)
from utils.helpers import (
    create_directories,
    detect_gpu,
    save_model,
    set_random_seed,
    setup_logging,
)
from utils.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    predict_dataset,
)
from utils.visualization import plot_model_comparison


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODELS = ("mobilenetv2", "efficientnetb0", "resnet50")
DEFAULT_COMPARISON_EPOCHS = 5
PRIMARY_METRIC = "validation_macro_f1"


class _ModelBuilderConfigProxy:
    """Provide train.py-compatible optional compile settings to model_builder."""

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
    """Build a configured model without modifying config.py or model_builder.py."""
    original_config = model_builder.CONFIG
    model_builder.CONFIG = _ModelBuilderConfigProxy()
    try:
        return model_builder.build_model(model_name, num_classes)
    finally:
        model_builder.CONFIG = original_config


def _parameter_count(weights: Sequence[tf.Variable]) -> int:
    """Return the scalar count for a Keras weight collection."""
    return int(sum(tf.keras.backend.count_params(weight) for weight in weights))


def _validate_class_names(*datasets: tf.data.Dataset) -> list[str]:
    """Ensure every split exposes the same non-empty inferred class ordering."""
    class_name_sets = [list(getattr(dataset, "class_names", [])) for dataset in datasets]
    if not class_name_sets[0] or len(class_name_sets[0]) < 2:
        raise ValueError("The dataset must contain at least two inferred class directories.")
    if any(names != class_name_sets[0] for names in class_name_sets[1:]):
        raise ValueError(
            "Train, validation, and test class directory order must be identical."
        )
    return class_name_sets[0]


def _configure_unique_callbacks(
    dataset_name: str,
    model_name: str,
    output_directory: Path,
) -> list[tf.keras.callbacks.Callback]:
    """Reuse standard callbacks while giving each comparison run unique logs."""
    callbacks = get_callbacks(dataset_name, f"comparison_{model_name}")
    for callback in callbacks:
        if isinstance(callback, tf.keras.callbacks.CSVLogger):
            callback.filename = str(output_directory / f"{model_name}_training.csv")
        elif isinstance(callback, tf.keras.callbacks.TensorBoard):
            callback.log_dir = str(CONFIG.logs_dir / "comparisons" / dataset_name / model_name)
    return callbacks


def _evaluate_split(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> dict[str, float]:
    """Evaluate one split using the same reusable metric functions for all models."""
    try:
        values = model.evaluate(dataset, verbose=0, return_dict=True)
        true_labels, predicted_labels, _ = predict_dataset(model, dataset)
    except (tf.errors.OpError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Model evaluation failed for the selected split.") from error
    if "loss" not in values:
        raise RuntimeError("Model evaluation did not return a loss value.")
    return {
        "loss": float(values["loss"]),
        "accuracy": calculate_accuracy(true_labels, predicted_labels),
        "precision": calculate_precision(true_labels, predicted_labels, average="macro"),
        "recall": calculate_recall(true_labels, predicted_labels, average="macro"),
        "macro_f1": calculate_f1_score(true_labels, predicted_labels, average="macro"),
        "weighted_f1": calculate_f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
        ),
    }


def _write_results(
    results: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write the sorted comparison results as both CSV and JSON."""
    csv_path = output_directory / "comparison_results.csv"
    json_path = output_directory / "comparison_results.json"
    if not results:
        raise ValueError("Comparison results must not be empty.")
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError("Unable to save comparison results.") from error
    return csv_path, json_path


def _save_comparison_plot(
    results: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> Path:
    """Reuse the project comparison plot and place it in this experiment folder."""
    plot_input = {
        str(result["model_name"]): {
            "accuracy": float(result["test_accuracy"]),
            "precision": float(result["precision"]),
            "recall": float(result["recall"]),
            "f1_score": float(result["macro_f1"]),
            "training_time": float(result["training_time_seconds"]),
            "inference_time": float(result["test_inference_time_seconds"]),
        }
        for result in results
    }
    source_path = plot_model_comparison(plot_input)
    target_path = output_directory / "comparison_plot.png"
    try:
        source_path.replace(target_path)
    except OSError as error:
        raise RuntimeError(f"Unable to save comparison plot: {target_path}") from error
    return target_path


def compare_models(
    dataset_name: str,
    model_names: Sequence[str],
    epochs: int,
) -> list[dict[str, Any]]:
    """Train and fairly compare selected backbones for one disease dataset.

    Models are ranked by validation macro F1, with validation accuracy as a
    deterministic tie breaker. Test metrics are saved for final reporting only.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    dataset_key = dataset_name.strip().lower()
    CONFIG.get_dataset(dataset_key)
    CONFIG.validate_dataset(dataset_key)
    selected_models = list(dict.fromkeys(name.strip().lower() for name in model_names))
    if not selected_models or any(name not in SUPPORTED_MODELS for name in selected_models):
        raise ValueError(f"models must be selected from: {', '.join(SUPPORTED_MODELS)}.")

    create_directories()
    logger = setup_logging()
    detect_gpu()
    output_directory = CONFIG.outputs_dir / "comparisons" / dataset_key
    model_directory = CONFIG.models_dir / "comparisons" / dataset_key
    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Comparison started | dataset: %s | models: %s | epochs: %d | "
        "primary metric: %s",
        dataset_key,
        selected_models,
        epochs,
        PRIMARY_METRIC,
    )
    logger.info(
        "Fixed settings | image size: %s | batch size: %d | seed: %d | "
        "learning rate: %s",
        CONFIG.image_size,
        CONFIG.batch_size,
        CONFIG.random_seed,
        CONFIG.initial_learning_rate,
    )

    results: list[dict[str, Any]] = []
    for model_name in selected_models:
        logger.info("Starting independent %s run.", model_name)
        set_random_seed(CONFIG.random_seed)
        train_dataset = load_train_dataset(dataset_key)
        validation_dataset = load_validation_dataset(dataset_key)
        test_dataset = load_test_dataset(dataset_key)
        class_names = _validate_class_names(train_dataset, validation_dataset, test_dataset)

        try:
            model = _build_model(model_name, len(class_names))
            callbacks = _configure_unique_callbacks(dataset_key, model_name, output_directory)
            started_at = perf_counter()
            model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=callbacks,
                shuffle=False,
                verbose=CONFIG.verbose,
            )
            training_time = perf_counter() - started_at
        except (tf.errors.OpError, TypeError, ValueError, RuntimeError) as error:
            logger.exception("Training failed for %s.", model_name)
            raise RuntimeError(f"Training failed for '{model_name}'.") from error

        validation_metrics = _evaluate_split(model, validation_dataset)
        inference_started_at = perf_counter()
        test_metrics = _evaluate_split(model, test_dataset)
        test_inference_time = perf_counter() - inference_started_at
        model_path = model_directory / f"{model_name}.keras"
        save_model(model, model_path, overwrite=True)
        result = {
            "model_name": model_name,
            "dataset": dataset_key,
            "parameter_count": int(model.count_params()),
            "trainable_parameters": _parameter_count(model.trainable_weights),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "training_time_seconds": training_time,
            "test_inference_time_seconds": test_inference_time,
            "model_path": str(model_path),
            "checkpoint_path": str(
                CONFIG.checkpoints_dir / f"{dataset_key}_comparison_{model_name}_best.keras"
            ),
        }
        results.append(result)
        logger.info(
            "%s completed | validation macro F1: %.4f | test accuracy: %.4f",
            model_name,
            result["validation_macro_f1"],
            result["test_accuracy"],
        )
        tf.keras.backend.clear_session()

    results.sort(
        key=lambda result: (result["validation_macro_f1"], result["validation_accuracy"]),
        reverse=True,
    )
    best_model = results[0]
    for rank, result in enumerate(results, start=1):
        result["rank_by_validation_macro_f1"] = rank
        result["is_best_model"] = rank == 1
    csv_path, json_path = _write_results(results, output_directory)
    plot_path = _save_comparison_plot(results, output_directory)
    logger.info(
        "Comparison completed | best model: %s | validation macro F1: %.4f",
        best_model["model_name"],
        best_model["validation_macro_f1"],
    )
    logger.info("Results saved | CSV: %s | JSON: %s | plot: %s", csv_path, json_path, plot_path)
    return results


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a comparison experiment request without executing it on import."""
    parser = argparse.ArgumentParser(
        description="Fairly compare transfer-learning backbones for one disease dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        required=True,
        help="Disease dataset to use for the controlled experiment.",
    )
    parser.add_argument(
        "--models",
        choices=SUPPORTED_MODELS,
        nargs="+",
        default=list(SUPPORTED_MODELS),
        help="Backbones to compare (default: all supported backbones).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_COMPARISON_EPOCHS,
        help=f"Training epochs per model (default: {DEFAULT_COMPARISON_EPOCHS}).",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run an explicitly requested comparison experiment."""
    try:
        args = _parse_arguments(arguments)
        results = compare_models(args.dataset, args.models, args.epochs)
    except KeyboardInterrupt:
        LOGGER.warning("Comparison cancelled by user.")
        return 130
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        LOGGER.error("Comparison validation error: %s", error)
        print(f"Comparison validation error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        LOGGER.exception("Comparison failed.")
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1

    best_model = results[0]
    print(
        f"Best model by validation macro F1: {best_model['model_name']}\n"
        f"Validation macro F1: {best_model['validation_macro_f1']:.4f}\n"
        f"Results: {CONFIG.outputs_dir / 'comparisons' / args.dataset}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
