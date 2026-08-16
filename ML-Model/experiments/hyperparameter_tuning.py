"""Run a conservative, validation-only hyperparameter tuning experiment.

Each trial changes one setting from the baseline. Models are ranked exclusively
by validation macro F1; the test split is deliberately never loaded here.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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

import tensorflow as tf

from config import CONFIG
from utils import callbacks as callbacks_module
from utils import dataset_loader, model_builder
from utils.callbacks import get_callbacks
from utils.dataset_loader import load_train_dataset, load_validation_dataset
from utils.helpers import create_directories, detect_gpu, set_random_seed, setup_logging
from utils.metrics import calculate_f1_score, predict_dataset
from utils.model_builder import unfreeze_backbone
from utils.visualization import plot_model_comparison


LOGGER = logging.getLogger(__name__)
SUPPORTED_BACKBONES = ("mobilenetv2", "efficientnetb0", "resnet50")
DEFAULT_EPOCHS = 3
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
    """Create a compact one-factor-at-a-time search around project defaults."""
    baseline_optimizer = getattr(CONFIG, "optimizer", "adam")
    alternative_optimizer = "rmsprop" if str(baseline_optimizer).lower() == "adam" else "adam"
    reduced_batch_size = max(8, CONFIG.batch_size // 2)
    increased_dropout = min(0.60, CONFIG.dropout_rate + 0.15)
    return [
        {
            "label": "baseline",
            "initial_learning_rate": CONFIG.initial_learning_rate,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": CONFIG.dropout_rate,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "learning_rate_alternative",
            "initial_learning_rate": CONFIG.initial_learning_rate * 3,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": CONFIG.dropout_rate,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "batch_size_alternative",
            "initial_learning_rate": CONFIG.initial_learning_rate,
            "batch_size": reduced_batch_size,
            "dropout_rate": CONFIG.dropout_rate,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "dropout_alternative",
            "initial_learning_rate": CONFIG.initial_learning_rate,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": increased_dropout,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "optimizer_alternative",
            "initial_learning_rate": CONFIG.initial_learning_rate,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": CONFIG.dropout_rate,
            "optimizer": alternative_optimizer,
            "fine_tune_layers": 0,
        },
        {
            "label": "fine_tune_last_10",
            "initial_learning_rate": CONFIG.initial_learning_rate,
            "batch_size": CONFIG.batch_size,
            "dropout_rate": CONFIG.dropout_rate,
            "optimizer": baseline_optimizer,
            "fine_tune_layers": 10,
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
    return callbacks


def _recompile_after_unfreezing(model: tf.keras.Model) -> None:
    """Recompile after changing trainable layers so fine tuning takes effect."""
    optimizer_config = tf.keras.optimizers.serialize(model.optimizer)
    optimizer = tf.keras.optimizers.deserialize(optimizer_config)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


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


def _save_history_plot(
    results: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> Path:
    """Reuse the comparison visualization to show validation trial performance."""
    plot_input = {
        f"Trial {result['trial_number']}": {
            "accuracy": float(result["best_validation_accuracy"]),
            "precision": float(result["validation_macro_f1"]),
            "recall": float(result["validation_macro_f1"]),
            "f1_score": float(result["validation_macro_f1"]),
            "training_time": float(result["training_time_seconds"]),
            "inference_time": 0.0,
        }
        for result in results
    }
    source_path = plot_model_comparison(plot_input)
    target_path = output_directory / "tuning_history.png"
    try:
        source_path.replace(target_path)
    except OSError as error:
        raise RuntimeError(f"Unable to save tuning history plot: {target_path}") from error
    return target_path


def tune_hyperparameters(
    dataset_name: str,
    backbone: str,
    epochs: int,
    max_trials: int,
) -> list[dict[str, Any]]:
    """Run controlled one-factor-at-a-time tuning without ever loading test data."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer.")

    dataset_key = dataset_name.strip().lower()
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
    candidates = _trial_candidates()[:max_trials]
    logger.info(
        "Tuning started | dataset: %s | backbone: %s | epochs: %d | trials: %d | "
        "primary metric: %s | test split: not loaded",
        dataset_key,
        backbone,
        epochs,
        len(candidates),
        PRIMARY_METRIC,
    )

    results: list[dict[str, Any]] = []
    for trial_number, candidate in enumerate(candidates, start=1):
        logger.info("Starting trial %d: %s", trial_number, candidate["label"])
        set_random_seed(CONFIG.random_seed)
        with _trial_configuration(candidate):
            train_dataset = load_train_dataset(dataset_key)
            validation_dataset = load_validation_dataset(dataset_key)
            class_names = _validate_class_names(train_dataset, validation_dataset)
            try:
                model = model_builder.build_model(backbone, len(class_names))
                fine_tune_layers = int(candidate["fine_tune_layers"])
                if fine_tune_layers:
                    unfreeze_backbone(model, fine_tune_layers)
                    _recompile_after_unfreezing(model)
                callbacks = _unique_callbacks(
                    dataset_key,
                    backbone,
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
                true_labels, predicted_labels, _ = predict_dataset(model, validation_dataset)
            except (tf.errors.OpError, TypeError, ValueError, RuntimeError) as error:
                logger.exception("Tuning trial %d failed.", trial_number)
                raise RuntimeError(f"Tuning trial {trial_number} failed.") from error

        validation_macro_f1 = calculate_f1_score(
            true_labels,
            predicted_labels,
            average="macro",
        )
        trial_result: dict[str, Any] = {
            "trial_number": trial_number,
            "trial_label": candidate["label"],
            "dataset": dataset_key,
            "backbone": backbone,
            "learning_rate": float(candidate["initial_learning_rate"]),
            "batch_size": int(candidate["batch_size"]),
            "dropout": float(candidate["dropout_rate"]),
            "optimizer": str(candidate["optimizer"]),
            "fine_tune_layers": int(candidate["fine_tune_layers"]),
            "fine_tuning_strategy": (
                "frozen" if not candidate["fine_tune_layers"] else "unfreeze_final_layers"
            ),
            "best_validation_accuracy": max(float(value) for value in history.history["val_accuracy"]),
            "best_validation_loss": min(float(value) for value in history.history["val_loss"]),
            "validation_macro_f1": validation_macro_f1,
            "training_time_seconds": training_time,
            "checkpoint_path": str(
                CONFIG.checkpoints_dir
                / f"{dataset_key}_tuning_{backbone}_trial_{trial_number}_best.keras"
            ),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }
        results.append(trial_result)
        logger.info(
            "Trial %d completed | validation macro F1: %.4f | best validation accuracy: %.4f",
            trial_number,
            validation_macro_f1,
            trial_result["best_validation_accuracy"],
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
    plot_path = _save_history_plot(results, output_directory)
    logger.info(
        "Tuning completed | best trial: %d | validation macro F1: %.4f | "
        "CSV: %s | JSON: %s | best config: %s | plot: %s",
        best_result["trial_number"],
        best_result["validation_macro_f1"],
        csv_path,
        json_path,
        best_path,
        plot_path,
    )
    return results


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a conservative tuning run without executing it on import."""
    parser = argparse.ArgumentParser(
        description="Tune controlled medical-image model hyperparameters."
    )
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        required=True,
        help="Disease dataset used for train/validation-only tuning.",
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default=CONFIG.backbone,
        help="Backbone to tune (default: CONFIG.backbone).",
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
        results = tune_hyperparameters(
            args.dataset,
            args.backbone,
            args.epochs,
            args.max_trials,
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
        f"Best trial by validation macro F1: {best_result['trial_number']}\n"
        f"Validation macro F1: {best_result['validation_macro_f1']:.4f}\n"
        f"Results: {CONFIG.outputs_dir / 'tuning' / args.dataset}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
