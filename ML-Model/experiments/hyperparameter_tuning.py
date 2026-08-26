"""Run controlled, validation-only hyperparameter tuning and fine-tuning.

Each trial evaluates a distinct configuration (learning rate, dropout, batch size,
or top-layer backbone fine-tuning). Models are ranked exclusively by validation
macro F1 score; the test split is deliberately never loaded here.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

# Permit direct execution from Windows, Google Colab, and Lightning AI.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from config import CONFIG
from utils import callbacks as callbacks_module
from utils import dataset_loader, model_builder
from utils.callbacks import get_callbacks
from utils.dataset_loader import load_train_dataset, load_validation_dataset
from utils.helpers import create_directories, detect_gpu, set_random_seed, setup_logging
from utils.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    predict_dataset,
)
from utils.model_builder import unfreeze_backbone
from utils.visualization import plot_model_comparison


LOGGER = logging.getLogger(__name__)
SUPPORTED_BACKBONES = ("mobilenetv2", "efficientnetb0", "resnet50")
SELECTED_DOMAIN_BACKBONES: dict[str, str] = {
    "skin": "resnet50",
    "eye": "resnet50",
    "oral": "efficientnetb0",
}
DEFAULT_EPOCHS = 4
DEFAULT_MAX_TRIALS = 5
PRIMARY_METRIC = "validation_macro_f1"


class _ConfigProxy:
    """Temporarily override selected immutable project settings for one trial."""

    def __init__(self, overrides: Mapping[str, Any]) -> None:
        self._overrides = dict(overrides)

    def __getattr__(self, attribute: str) -> Any:
        defaults: dict[str, Any] = {
            "optimizer": "adam",
            "loss_function": "sparse_categorical_crossentropy",
            "metrics": ("accuracy",),
            "early_stopping_patience": 3,
            "reduce_lr_patience": 2,
            "reduce_lr_factor": 0.5,
            "min_learning_rate": 1e-7,
        }
        if attribute in self._overrides:
            return self._overrides[attribute]
        if attribute in defaults:
            return defaults[attribute]
        return getattr(CONFIG, attribute)


@contextmanager
def _trial_configuration(overrides: Mapping[str, Any]) -> Iterator[None]:
    """Apply trial settings to existing utilities without mutating ``CONFIG``."""
    proxy = _ConfigProxy(overrides)
    original_model_config = model_builder.CONFIG
    original_dataset_config = dataset_loader.CONFIG
    original_callbacks_config = callbacks_module.CONFIG
    model_builder.CONFIG = proxy
    dataset_loader.CONFIG = proxy
    callbacks_module.CONFIG = proxy
    try:
        yield
    finally:
        model_builder.CONFIG = original_model_config
        dataset_loader.CONFIG = original_dataset_config
        callbacks_module.CONFIG = original_callbacks_config


def _trial_candidates() -> list[dict[str, Any]]:
    """Create the 5 controlled trial candidates for hyperparameter and fine-tuning."""
    baseline_optimizer = getattr(CONFIG, "optimizer", "adam")
    return [
        {
            "label": "baseline_frozen",
            "description": "Baseline transfer learning with frozen backbone (lr=1e-4, dropout=0.30)",
            "initial_learning_rate": 1e-4,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": 0.30,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "lower_learning_rate",
            "description": "Lower initial learning rate for smoother optimization (lr=5e-5, dropout=0.30)",
            "initial_learning_rate": 5e-5,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": 0.30,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "higher_learning_rate",
            "description": "Higher initial learning rate for faster adaptation (lr=3e-4, dropout=0.30)",
            "initial_learning_rate": 3e-4,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": 0.30,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "higher_dropout",
            "description": "Increased dropout regularization to mitigate overfitting (lr=1e-4, dropout=0.45)",
            "initial_learning_rate": 1e-4,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": 0.45,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "fine_tune_top_layers",
            "description": "Unfreeze top 15 backbone layers with low learning rate (lr=1e-5, dropout=0.30)",
            "initial_learning_rate": 1e-5,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": 0.30,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 15,
        },
    ]


def _validate_class_names(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> list[str]:
    """Confirm identical inferred class ordering across the allowed splits."""
    train_class_names = list(getattr(train_dataset, "class_names", []))
    validation_class_names = list(getattr(validation_dataset, "class_names", []))
    if len(train_class_names) < 2:
        raise ValueError("Training data must contain at least two class directories.")
    if train_class_names != validation_class_names:
        raise ValueError("Train and validation class directory order must match exactly.")
    return train_class_names


def _unique_callbacks(
    dataset_name: str,
    backbone: str,
    trial_number: int,
    output_directory: Path,
) -> list[tf.keras.callbacks.Callback]:
    """Reuse standard callbacks while isolating trial checkpoint/log artifacts."""
    trial_name = f"tuning_{backbone}_trial_{trial_number}"
    callbacks = get_callbacks(dataset_name, trial_name)
    for callback in callbacks:
        if isinstance(callback, tf.keras.callbacks.CSVLogger):
            callback.filename = str(output_directory / f"trial_{trial_number}_training.csv")
        elif isinstance(callback, tf.keras.callbacks.TensorBoard):
            callback.log_dir = str(
                CONFIG.logs_dir / "tuning" / dataset_name / backbone / f"trial_{trial_number}"
            )
        elif isinstance(callback, tf.keras.callbacks.EarlyStopping):
            callback.patience = 3
        elif isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau):
            callback.patience = 2
            callback.factor = 0.5
    return callbacks


def _recompile_after_unfreezing(model: tf.keras.Model, learning_rate: float) -> None:
    """Recompile after changing trainable layers so fine tuning takes effect."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _save_tuning_plots(
    results: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> list[Path]:
    """Save individual and combined visualization plots for the tuning results."""
    saved_plots: list[Path] = []
    labels = [f"T{r['trial_number']}: {r['trial_label']}" for r in results]
    macro_f1s = [float(r["validation_macro_f1"]) for r in results]
    accuracies = [float(r["best_validation_accuracy"]) for r in results]
    training_times = [float(r["training_time_seconds"]) for r in results]

    # 1. Validation Macro F1 Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, macro_f1s, color="#4C72B0")
    ax.set_title("Validation Macro F1 Score by Tuning Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Macro F1")
    ax.set_ylim(0, max(macro_f1s) * 1.15 if max(macro_f1s) > 0 else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    f1_plot = output_directory / "tuning_macro_f1_comparison.png"
    fig.savefig(f1_plot, dpi=150)
    plt.close(fig)
    saved_plots.append(f1_plot)

    # 2. Validation Accuracy Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accuracies, color="#55A868")
    ax.set_title("Best Validation Accuracy by Tuning Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Accuracy")
    ax.set_ylim(0, max(accuracies) * 1.15 if max(accuracies) > 0 else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    acc_plot = output_directory / "tuning_accuracy_comparison.png"
    fig.savefig(acc_plot, dpi=150)
    plt.close(fig)
    saved_plots.append(acc_plot)

    # 3. Training Time Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, training_times, color="#C44E52")
    ax.set_title("Training Time (s) by Tuning Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    time_plot = output_directory / "tuning_training_time_comparison.png"
    fig.savefig(time_plot, dpi=150)
    plt.close(fig)
    saved_plots.append(time_plot)

    # 4. Standard 6-panel Comparison Plot
    plot_input = {
        f"Trial {result['trial_number']}": {
            "accuracy": float(result["best_validation_accuracy"]),
            "precision": float(result["validation_macro_precision"]),
            "recall": float(result["validation_macro_recall"]),
            "f1_score": float(result["validation_macro_f1"]),
            "training_time": float(result["training_time_seconds"]),
            "inference_time": float(result.get("validation_inference_time_seconds", 0.0)),
        }
        for result in results
    }
    source_path = plot_model_comparison(plot_input)
    target_path = output_directory / "tuning_history.png"
    try:
        if source_path.resolve() != target_path.resolve():
            target_path.write_bytes(source_path.read_bytes())
    except Exception:
        pass
    saved_plots.append(target_path)

    return saved_plots


def _write_results(
    results: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Save complete JSON trial histories and CSV trial summaries."""
    csv_path = output_directory / "tuning_results.csv"
    json_path = output_directory / "tuning_results.json"
    if not results:
        raise ValueError("Tuning results must not be empty.")
    summary_rows = [{key: value for key, value in row.items() if key != "history"} for row in results]
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError("Unable to save tuning results.") from error
    return csv_path, json_path


def tune_hyperparameters(
    dataset_name: str,
    backbone: str | None = None,
    epochs: int = DEFAULT_EPOCHS,
    max_trials: int = DEFAULT_MAX_TRIALS,
) -> list[dict[str, Any]]:
    """Run controlled one-factor-at-a-time tuning without ever loading test data."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    
    dataset_key = dataset_name.strip().lower()
    selected_backbone = backbone or SELECTED_DOMAIN_BACKBONES.get(dataset_key, CONFIG.backbone)
    if selected_backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{selected_backbone}'.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer.")

    CONFIG.get_dataset(dataset_key)
    for split_name in ("train", "validation"):
        split_path = CONFIG.split_dir(dataset_key, split_name)
        if not split_path.is_dir():
            raise FileNotFoundError(f"{split_name.title()} dataset directory was not found: {split_path}")

    create_directories()
    logger = setup_logging()
    detect_gpu()
    output_directory = CONFIG.outputs_dir / "tuning" / dataset_key
    output_directory.mkdir(parents=True, exist_ok=True)
    models_tuning_dir = CONFIG.models_dir / "tuning" / dataset_key
    models_tuning_dir.mkdir(parents=True, exist_ok=True)

    candidates = _trial_candidates()[:max_trials]
    logger.info(
        "Tuning started | dataset: %s | backbone: %s | epochs: %d | trials: %d | "
        "primary metric: %s | test split: not loaded",
        dataset_key,
        selected_backbone,
        epochs,
        len(candidates),
        PRIMARY_METRIC,
    )

    results: list[dict[str, Any]] = []
    for trial_number, candidate in enumerate(candidates, start=1):
        logger.info(
            "Starting trial %d/%d: %s (%s)",
            trial_number,
            len(candidates),
            candidate["label"],
            candidate["description"],
        )
        set_random_seed(CONFIG.random_seed)
        checkpoint_file = (
            CONFIG.checkpoints_dir
            / f"{dataset_key}_tuning_{selected_backbone}_trial_{trial_number}_best.keras"
        )
        with _trial_configuration(candidate):
            train_dataset = load_train_dataset(dataset_key)
            validation_dataset = load_validation_dataset(dataset_key)
            class_names = _validate_class_names(train_dataset, validation_dataset)
            try:
                model = model_builder.build_model(selected_backbone, len(class_names))
                fine_tune_layers = int(candidate["fine_tune_layers"])
                if fine_tune_layers:
                    unfreeze_backbone(model, fine_tune_layers)
                    _recompile_after_unfreezing(model, float(candidate["initial_learning_rate"]))
                
                param_count = int(sum(tf.keras.backend.count_params(w) for w in model.weights))
                trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))

                callbacks = _unique_callbacks(
                    dataset_key,
                    selected_backbone,
                    trial_number,
                    output_directory,
                )
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

                eval_start = perf_counter()
                true_labels, predicted_labels, _ = predict_dataset(model, validation_dataset)
                val_inference_time = perf_counter() - eval_start
            except (tf.errors.OpError, TypeError, ValueError, RuntimeError) as error:
                logger.exception("Tuning trial %d failed.", trial_number)
                raise RuntimeError(f"Tuning trial {trial_number} failed.") from error

        val_macro_f1 = calculate_f1_score(true_labels, predicted_labels, average="macro")
        val_weighted_f1 = calculate_f1_score(true_labels, predicted_labels, average="weighted")
        val_macro_prec = calculate_precision(true_labels, predicted_labels, average="macro")
        val_macro_rec = calculate_recall(true_labels, predicted_labels, average="macro")
        best_val_acc = max(float(v) for v in history.history["val_accuracy"])
        best_val_loss = min(float(v) for v in history.history["val_loss"])
        best_epoch = int(history.history["val_loss"].index(best_val_loss) + 1)
        final_train_acc = float(history.history["accuracy"][-1])

        trial_result: dict[str, Any] = {
            "trial_number": trial_number,
            "trial_label": candidate["label"],
            "description": candidate["description"],
            "dataset": dataset_key,
            "backbone": selected_backbone,
            "learning_rate": float(candidate["initial_learning_rate"]),
            "batch_size": int(candidate["batch_size"]),
            "dropout": float(candidate["dropout_rate"]),
            "optimizer": str(candidate["optimizer"]),
            "fine_tune_layers": int(candidate["fine_tune_layers"]),
            "fine_tuning_strategy": (
                "frozen" if not candidate["fine_tune_layers"] else f"unfreeze_top_{candidate['fine_tune_layers']}_layers"
            ),
            "parameter_count": param_count,
            "trainable_parameters": trainable_params,
            "training_accuracy": final_train_acc,
            "best_validation_accuracy": best_val_acc,
            "best_validation_loss": best_val_loss,
            "validation_macro_precision": val_macro_prec,
            "validation_macro_recall": val_macro_rec,
            "validation_macro_f1": val_macro_f1,
            "validation_weighted_f1": val_weighted_f1,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time,
            "validation_inference_time_seconds": val_inference_time,
            "checkpoint_path": str(checkpoint_file),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }
        results.append(trial_result)
        logger.info(
            "Trial %d completed | label: %s | Val Macro F1: %.4f | Best Val Acc: %.4f | Best Val Loss: %.4f | Time: %.1fs",
            trial_number,
            candidate["label"],
            val_macro_f1,
            best_val_acc,
            best_val_loss,
            training_time,
        )
        tf.keras.backend.clear_session()

    results.sort(
        key=lambda result: (result["validation_macro_f1"], result["best_validation_accuracy"]),
        reverse=True,
    )
    for rank, result in enumerate(results, start=1):
        result["rank_by_validation_macro_f1"] = rank
        result["is_best_configuration"] = rank == 1

    best_result = results[0]
    csv_path, json_path = _write_results(results, output_directory)
    best_path = output_directory / "best_hyperparameters.json"
    try:
        best_path.write_text(json.dumps(best_result, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unable to save best hyperparameters: {best_path}") from error
    
    # Save the best tuned model separately in models/tuning/<dataset>/
    best_checkpoint = Path(best_result["checkpoint_path"])
    if best_checkpoint.is_file():
        tuned_model_dest = models_tuning_dir / f"{selected_backbone}_best_tuned.keras"
        try:
            tuned_model_dest.write_bytes(best_checkpoint.read_bytes())
            logger.info("Saved best tuned model to %s", tuned_model_dest)
        except Exception as error:
            logger.warning("Could not copy best tuned model: %s", error)

    plot_paths = _save_tuning_plots(results, output_directory)
    logger.info(
        "Tuning completed for %s | best trial: %d (%s) | validation macro F1: %.4f | "
        "CSV: %s | JSON: %s | best config: %s | plots: %s",
        dataset_key,
        best_result["trial_number"],
        best_result["trial_label"],
        best_result["validation_macro_f1"],
        csv_path,
        json_path,
        best_path,
        plot_paths,
    )
    return results


def run_all_tuning(
    epochs: int = DEFAULT_EPOCHS,
    max_trials: int = DEFAULT_MAX_TRIALS,
) -> dict[str, list[dict[str, Any]]]:
    """Run sequential tuning across all 3 selected domain architectures."""
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Starting Sequential Multi-Domain Hyperparameter & Fine-Tuning Run")
    logger.info("Order: 1. Skin (ResNet50) -> 2. Eye (ResNet50) -> 3. Oral (EfficientNetB0)")
    logger.info("=" * 70)

    dataset_order = ["skin", "eye", "oral"]
    all_results: dict[str, list[dict[str, Any]]] = {}
    summary_records: list[dict[str, Any]] = []

    for dataset_key in dataset_order:
        backbone = SELECTED_DOMAIN_BACKBONES[dataset_key]
        logger.info("\n>>> Beginning tuning for dataset: %s (Backbone: %s) <<<", dataset_key.upper(), backbone)
        domain_results = tune_hyperparameters(
            dataset_name=dataset_key,
            backbone=backbone,
            epochs=epochs,
            max_trials=max_trials,
        )
        all_results[dataset_key] = domain_results
        best = domain_results[0]
        summary_records.append({
            "dataset": dataset_key,
            "winning_backbone": backbone,
            "best_trial_number": best["trial_number"],
            "best_trial_label": best["trial_label"],
            "best_learning_rate": best["learning_rate"],
            "best_dropout": best["dropout"],
            "best_fine_tune_layers": best["fine_tune_layers"],
            "best_validation_macro_f1": best["validation_macro_f1"],
            "best_validation_accuracy": best["best_validation_accuracy"],
            "best_validation_loss": best["best_validation_loss"],
            "training_time_seconds": best["training_time_seconds"],
        })

    # Save cross-domain overall summary
    tuning_root = CONFIG.outputs_dir / "tuning"
    overall_csv = tuning_root / "overall_tuning_summary.csv"
    overall_json = tuning_root / "overall_tuning_summary.json"
    with overall_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    overall_json.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("ALL DOMAIN TUNING COMPLETED SUCCESSFULLY!")
    logger.info("Overall summary saved to %s and %s", overall_csv, overall_json)
    logger.info("=" * 70)
    return all_results


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a conservative tuning run without executing it on import."""
    parser = argparse.ArgumentParser(
        description="Tune controlled medical-image model hyperparameters."
    )
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral", "all"),
        required=True,
        help="Disease dataset used for train/validation-only tuning (or 'all' for sequential run).",
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default=None,
        help="Backbone to tune (defaults to the selected winning backbone for the domain).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Epochs per trial (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help=f"Maximum controlled trials (default: {DEFAULT_MAX_TRIALS}).",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run an explicitly requested tuning experiment."""
    try:
        args = _parse_arguments(arguments)
        if args.dataset == "all":
            all_results = run_all_tuning(
                epochs=args.epochs,
                max_trials=args.max_trials,
            )
            print("\nSequential tuning completed for all domains!")
            for domain, domain_results in all_results.items():
                best = domain_results[0]
                print(
                    f"- {domain.upper()} (Backbone: {SELECTED_DOMAIN_BACKBONES[domain]}): "
                    f"Best Trial {best['trial_number']} ({best['trial_label']}) | "
                    f"Val Macro F1: {best['validation_macro_f1']:.4f} | "
                    f"Best Val Acc: {best['best_validation_accuracy']:.4f}"
                )
            return 0

        results = tune_hyperparameters(
            dataset_name=args.dataset,
            backbone=args.backbone,
            epochs=args.epochs,
            max_trials=args.max_trials,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Hyperparameter tuning cancelled by user.")
        return 130
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        LOGGER.error("Tuning validation error: %s", error)
        print(f"Tuning validation error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        LOGGER.exception("Tuning failed.")
        print(f"Tuning failed: {error}", file=sys.stderr)
        return 1

    best_result = results[0]
    print(
        f"Best trial by validation macro F1: {best_result['trial_number']} ({best_result['trial_label']})\n"
        f"Validation macro F1: {best_result['validation_macro_f1']:.4f}\n"
        f"Best validation accuracy: {best_result['best_validation_accuracy']:.4f}\n"
        f"Results: {CONFIG.outputs_dir / 'tuning' / args.dataset}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
