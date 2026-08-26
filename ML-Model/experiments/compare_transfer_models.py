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


def _save_comparison_plots(
    results: Sequence[Mapping[str, Any]],
    output_directory: Path,
    dataset_name: str,
) -> dict[str, Path]:
    """Generate all required comparison visualizations for the evaluated backbones."""
    model_names = [str(r["model_name"]) for r in results]
    macro_f1s = [float(r["validation_macro_f1"]) for r in results]
    val_accuracies = [float(r["best_val_accuracy"]) for r in results]
    training_times = [float(r["training_time_seconds"]) for r in results]
    param_counts = [int(r["parameter_count"]) for r in results]

    import matplotlib.pyplot as plt

    saved_plots: dict[str, Path] = {}
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    # 1. Validation Macro F1 Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(model_names, macro_f1s, color=colors[: len(model_names)])
    ax.set_title(f"Validation Macro F1 Score - {dataset_name.title()} Domain", fontsize=14, pad=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
    plot_path = output_directory / "validation_macro_f1_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_plots["macro_f1"] = plot_path

    # 2. Validation Accuracy Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(model_names, val_accuracies, color=colors[: len(model_names)])
    ax.set_title(f"Validation Accuracy - {dataset_name.title()} Domain", fontsize=14, pad=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
    plot_path = output_directory / "validation_accuracy_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_plots["accuracy"] = plot_path

    # 3. Training Time Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(model_names, training_times, color=colors[: len(model_names)])
    ax.set_title(f"Training Time (Seconds) - {dataset_name.title()} Domain", fontsize=14, pad=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=11)
    plot_path = output_directory / "training_time_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_plots["training_time"] = plot_path

    # 4. Parameter Count Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(model_names, [p / 1e6 for p in param_counts], color=colors[: len(model_names)])
    ax.set_title(f"Total Parameters (Millions) - {dataset_name.title()} Domain", fontsize=14, pad=12)
    ax.set_ylabel("Parameters (Millions)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.2f M", padding=3, fontsize=11)
    plot_path = output_directory / "parameter_count_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_plots["parameters"] = plot_path

    # 5. Combined 6-panel summary plot using project visualization utility
    plot_input = {
        str(result["model_name"]): {
            "accuracy": float(result["best_val_accuracy"]),
            "precision": float(result["validation_macro_precision"]),
            "recall": float(result["validation_macro_recall"]),
            "f1_score": float(result["validation_macro_f1"]),
            "training_time": float(result["training_time_seconds"]),
            "inference_time": float(result.get("validation_inference_time_seconds", 0.0)),
        }
        for result in results
    }
    source_path = plot_model_comparison(plot_input)
    target_path = output_directory / "comparison_plot.png"
    try:
        source_path.replace(target_path)
        saved_plots["combined"] = target_path
    except OSError:
        pass

    return saved_plots


def compare_models(
    dataset_name: str,
    model_names: Sequence[str],
    epochs: int,
) -> list[dict[str, Any]]:
    """Train and fairly compare selected backbones for one disease dataset.

    Models are ranked by validation macro F1, with validation accuracy as a
    deterministic tie breaker. Test split is strictly isolated and untouched.
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
        logger.info("Starting independent %s run on %s dataset.", model_name, dataset_key)
        set_random_seed(CONFIG.random_seed)
        train_dataset = load_train_dataset(dataset_key)
        validation_dataset = load_validation_dataset(dataset_key)
        test_dataset = load_test_dataset(dataset_key)
        class_names = _validate_class_names(train_dataset, validation_dataset, test_dataset)

        try:
            model = _build_model(model_name, len(class_names))
            callbacks = _configure_unique_callbacks(dataset_key, model_name, output_directory)
            started_at = perf_counter()
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=callbacks,
                shuffle=False,
                verbose=CONFIG.verbose,
            )
            training_time = perf_counter() - started_at
        except (tf.errors.OpError, TypeError, ValueError, RuntimeError) as error:
            logger.exception("Training failed for %s on %s.", model_name, dataset_key)
            raise RuntimeError(f"Training failed for '{model_name}' on '{dataset_key}'.") from error

        # Evaluate validation set with restored best weights from EarlyStopping
        val_eval_start = perf_counter()
        validation_metrics = _evaluate_split(model, validation_dataset)
        val_inference_time = perf_counter() - val_eval_start

        # Extract training progress history metrics
        train_accuracies = history.history.get("accuracy", [0.0])
        val_accuracies_hist = history.history.get("val_accuracy", [0.0])
        val_losses_hist = history.history.get("val_loss", [float("inf")])
        training_accuracy = float(train_accuracies[-1])
        best_val_accuracy = float(max(val_accuracies_hist))
        best_epoch = int(np.argmin(val_losses_hist) + 1)

        model_path = model_directory / f"{model_name}.keras"
        save_model(model, model_path, overwrite=True)

        result = {
            "dataset": dataset_key,
            "model_name": model_name,
            "parameter_count": int(model.count_params()),
            "trainable_parameters": _parameter_count(model.trainable_weights),
            "training_accuracy": training_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_macro_precision": validation_metrics["precision"],
            "validation_macro_recall": validation_metrics["recall"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_weighted_f1": validation_metrics["weighted_f1"],
            "best_epoch": best_epoch,
            "training_time_seconds": training_time,
            "validation_inference_time_seconds": val_inference_time,
            "model_path": str(model_path),
            "checkpoint_path": str(
                CONFIG.checkpoints_dir / f"{dataset_key}_comparison_{model_name}_best.keras"
            ),
        }
        results.append(result)
        logger.info(
            "%s completed | Best Val Acc: %.4f | Val Macro F1: %.4f | Best Epoch: %d | Time: %.1fs",
            model_name,
            result["best_val_accuracy"],
            result["validation_macro_f1"],
            result["best_epoch"],
            result["training_time_seconds"],
        )
        tf.keras.backend.clear_session()

    results.sort(
        key=lambda r: (r["validation_macro_f1"], r["best_val_accuracy"]),
        reverse=True,
    )
    best_model = results[0]
    for rank, result in enumerate(results, start=1):
        result["rank_by_validation_macro_f1"] = rank
        result["is_best_model"] = rank == 1

    csv_path, json_path = _write_results(results, output_directory)
    saved_plots = _save_comparison_plots(results, output_directory, dataset_key)
    logger.info(
        "Comparison completed for %s | winner: %s (Val Macro F1: %.4f)",
        dataset_key,
        best_model["model_name"],
        best_model["validation_macro_f1"],
    )
    logger.info("Results saved | CSV: %s | JSON: %s | plots: %s", csv_path, json_path, list(saved_plots.values()))
    return results


def run_all_comparisons(
    model_names: Sequence[str] = SUPPORTED_MODELS,
    epochs: int = DEFAULT_COMPARISON_EPOCHS,
) -> dict[str, list[dict[str, Any]]]:
    """Run model comparison sequentially across Skin, Eye, and Oral datasets."""
    all_results: dict[str, list[dict[str, Any]]] = {}
    datasets = ("skin", "eye", "oral")
    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"RUNNING COMPARISON FOR DOMAIN: {dataset_name.upper()}")
        print(f"{'=' * 60}\n")
        all_results[dataset_name] = compare_models(dataset_name, model_names, epochs)
    return all_results


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a comparison experiment request without executing it on import."""
    parser = argparse.ArgumentParser(
        description="Fairly compare transfer-learning backbones for medical image datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral", "all"),
        default="all",
        help="Disease dataset to use for comparison (default: all).",
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
        if args.dataset == "all":
            all_results = run_all_comparisons(args.models, args.epochs)
            print("\n" + "=" * 60)
            print("ALL TRANSFER LEARNING COMPARISONS COMPLETED")
            print("=" * 60)
            for ds, res in all_results.items():
                winner = res[0]
                print(f"Dataset: {ds.upper():<6} | Winner: {winner['model_name']:<15} | Val Macro F1: {winner['validation_macro_f1']:.4f} | Best Val Acc: {winner['best_val_accuracy']:.4f}")
        else:
            results = compare_models(args.dataset, args.models, args.epochs)
            best_model = results[0]
            print(
                f"\nBest model by validation macro F1: {best_model['model_name']}\n"
                f"Validation macro F1: {best_model['validation_macro_f1']:.4f}\n"
                f"Best validation accuracy: {best_model['best_val_accuracy']:.4f}\n"
                f"Results directory: {CONFIG.outputs_dir / 'comparisons' / args.dataset}"
            )
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
