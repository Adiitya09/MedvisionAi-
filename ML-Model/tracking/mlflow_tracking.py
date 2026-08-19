"""MLflow experiment tracking for the disease-classification training pipeline.

Use ``MLflowExperimentTracker`` as a context manager and include its callback
in ``model.fit``.  Compare completed runs from the command line:

    python tracking/mlflow_tracking.py --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# Support direct execution without assuming a project-root working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

from config import CONFIG, ProjectConfig


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure useful console logging for tracking operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def gpu_telemetry() -> dict[str, float]:
    """Read NVIDIA GPU utilization and memory use if ``nvidia-smi`` is present."""
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
        values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        return {
            "gpu_utilization_percent": values[0],
            "gpu_memory_used_mb": values[1],
            "gpu_memory_total_mb": values[2],
        }
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return {}


def collect_validation_metrics(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> dict[str, float]:
    """Calculate macro precision, recall, and F1 on a labelled validation set."""
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        predicted_labels.extend(np.argmax(probabilities, axis=1).tolist())
        true_labels.extend(np.asarray(labels).reshape(-1).astype(int).tolist())
    if not true_labels:
        raise ValueError("Cannot calculate validation metrics from an empty dataset.")
    return {
        "val_precision_macro": float(
            precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "val_recall_macro": float(
            recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "val_f1_macro": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
    }


def _extract_learning_rate(optimizer: tf.keras.optimizers.Optimizer | None) -> float:
    """Extract a numeric learning rate safely across Keras 2, Keras 3, and schedules."""
    if optimizer is None:
        return 0.0
    lr = getattr(optimizer, "learning_rate", 0.0)
    if callable(lr):
        try:
            iterations = getattr(optimizer, "iterations", 0)
            return float(lr(iterations))
        except Exception:
            return 0.0
    if hasattr(lr, "numpy"):
        return float(lr.numpy())
    try:
        return float(lr)
    except (TypeError, ValueError):
        return 0.0


class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    """Log Keras metrics, learning rate, validation F1, and GPU use per epoch."""

    def __init__(self, validation_dataset: tf.data.Dataset | None = None) -> None:
        super().__init__()
        self.validation_dataset = validation_dataset

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Persist one complete epoch record in the currently active MLflow run."""
        metrics = {
            key: float(value)
            for key, value in (logs or {}).items()
            if np.isscalar(value) and np.isfinite(value)
        }
        optimizer = getattr(self.model, "optimizer", None)
        metrics["learning_rate"] = _extract_learning_rate(optimizer)
        metrics.update(gpu_telemetry())
        if self.validation_dataset is not None:
            metrics.update(collect_validation_metrics(self.model, self.validation_dataset))
        mlflow.log_metrics(metrics, step=epoch + 1)


class MLflowExperimentTracker:
    """Manage an MLflow run around TensorFlow training and artifact logging.

    Example:
        with MLflowExperimentTracker("skin_efficientnetb0", params, validation_ds) as tracker:
            history = model.fit(train_ds, callbacks=[tracker.callback], ...)
            tracker.log_trained_model(model, history, class_names)
    """

    def __init__(
        self,
        experiment_name: str,
        parameters: dict[str, Any],
        validation_dataset: tf.data.Dataset | None = None,
        config: ProjectConfig = CONFIG,
    ) -> None:
        self.experiment_name = experiment_name
        self.parameters = parameters
        self.validation_dataset = validation_dataset
        self.config = config
        self._start_time: float | None = None
        self._run: mlflow.ActiveRun | None = None

    @property
    def callback(self) -> MLflowMetricsCallback:
        """Return a fresh Keras callback for this active experiment."""
        return MLflowMetricsCallback(self.validation_dataset)

    @property
    def tracking_uri(self) -> str:
        """Use a project-local MLflow file store, without hard-coded paths."""
        return (self.config.outputs_dir / "mlruns").resolve().as_uri()

    def __enter__(self) -> "MLflowExperimentTracker":
        self.config.outputs_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run()
        self._start_time = time.perf_counter()
        serializable_parameters = {
            key: str(value) if isinstance(value, (Path, tuple, list, dict)) else value
            for key, value in self.parameters.items()
        }
        mlflow.log_params(serializable_parameters)
        policy = getattr(tf.keras.mixed_precision, "global_policy", lambda: None)()
        policy_name = getattr(policy, "name", str(policy)) if policy is not None else "float32"
        mlflow.log_params(
            {
                "tensorflow_version": tf.__version__,
                "mixed_precision_policy": policy_name,
                "gpu_available": bool(tf.config.list_physical_devices("GPU")),
            }
        )
        mlflow.log_metrics(gpu_telemetry(), step=0)
        LOGGER.info("Started MLflow run %s", self._run.info.run_id)
        return self

    def log_trained_model(
        self,
        model: tf.keras.Model,
        history: tf.keras.callbacks.History | None = None,
        class_names: Iterable[str] | None = None,
    ) -> None:
        """Store the final Keras model, class mapping, history, and timing data."""
        if self._run is None:
            raise RuntimeError("Start the tracker context before logging a model.")
        elapsed_seconds = time.perf_counter() - (self._start_time or time.perf_counter())
        mlflow.log_metric("training_time_seconds", elapsed_seconds)
        mlflow.log_metrics(gpu_telemetry())
        with tempfile.TemporaryDirectory(prefix="mlflow_keras_") as temporary_directory:
            temporary_path = Path(temporary_directory)
            model_path = temporary_path / "model.keras"
            model.save(model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")
            if history is not None:
                history_path = temporary_path / "history.json"
                history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
                mlflow.log_artifact(str(history_path), artifact_path="training")
            if class_names is not None:
                classes_path = temporary_path / "class_names.json"
                classes_path.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")
                mlflow.log_artifact(str(classes_path), artifact_path="model")

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._run is None:
            return
        elapsed_seconds = time.perf_counter() - (self._start_time or time.perf_counter())
        mlflow.log_metric("training_time_seconds", elapsed_seconds)
        mlflow.log_metrics(gpu_telemetry())
        if exc_type is not None:
            mlflow.set_tag("run_status", "failed")
            mlflow.set_tag("failure", str(exc_value))
        else:
            mlflow.set_tag("run_status", "completed")
        mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        self._run = None


def compare_runs(experiment_name: str, config: ProjectConfig = CONFIG) -> Path:
    """Export a sortable run comparison table for an MLflow experiment."""
    mlflow.set_tracking_uri((config.outputs_dir / "mlruns").resolve().as_uri())
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named '{experiment_name}' was found.")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_f1_macro DESC", "metrics.val_accuracy DESC"],
    )
    columns = [
        "run_id", "status", "start_time", "metrics.val_accuracy", "metrics.val_loss",
        "metrics.val_precision_macro", "metrics.val_recall_macro", "metrics.val_f1_macro",
        "metrics.training_time_seconds", "params.optimizer", "params.learning_rate",
        "params.epochs",
    ]
    available_columns = [column for column in columns if column in runs.columns]
    output_path = config.outputs_dir / "mlflow_run_comparison.csv"
    runs.loc[:, available_columns].to_csv(output_path, index=False)
    LOGGER.info("Saved comparison of %d runs to %s", len(runs), output_path)
    return output_path


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Export a run comparison CSV.")
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        default=None,
        help="Optional disease domain identifier for the MLflow experiment.",
    )
    parser.add_argument("--experiment", default=CONFIG.backbone, help="MLflow experiment name.")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_arguments(arguments)
    if not args.compare:
        print("Use --compare to export a run comparison CSV, or import MLflowExperimentTracker into training workflows.")
        return 0
    experiment_name = args.dataset or args.experiment
    try:
        compare_runs(experiment_name)
    except Exception as error:
        LOGGER.exception("MLflow run comparison failed: %s", error)
        print(f"MLflow run comparison error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
