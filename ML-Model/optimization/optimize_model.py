"""Convert, quantize, evaluate, and benchmark a trained Keras classifier.

Example:
    python optimize_model.py --model models/efficientnetb0_best_tuned.keras

Outputs are written to ``outputs/optimization``. INT8 conversion needs the
configured training split to build a representative calibration dataset.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import gc
import json
import logging
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CONFIG, ProjectConfig

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def dataset_for_evaluation(
    directory: Path, config: ProjectConfig, batch_size: int = 32
) -> tf.data.Dataset:
    """Load labelled images without augmentation for fair accuracy comparisons."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=config.image_size,
        batch_size=batch_size,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)


def representative_dataset(config: ProjectConfig) -> Callable[[], object]:
    """Return a bounded image generator used to calibrate INT8 quantization."""
    calibration_data = dataset_for_evaluation(config.train_dir, config, batch_size=1).take(200)

    def generator() -> object:
        for images, _ in calibration_data:
            yield [tf.cast(images, tf.float32)]

    return generator


def convert_tflite(
    model: tf.keras.Model,
    destination: Path,
    optimization: str,
    config: ProjectConfig,
) -> Path:
    """Convert a Keras model to float32, float16, or strict full-INT8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif optimization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(config)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif optimization != "fp32":
        raise ValueError(f"Unsupported TFLite optimization: {optimization}")
    destination.write_bytes(converter.convert())
    LOGGER.info("Saved %s TFLite model: %s", optimization.upper(), destination)
    return destination


def convert_onnx(model: tf.keras.Model, destination: Path) -> Path | None:
    """Export ONNX when optional tf2onnx is installed; otherwise log guidance."""
    try:
        import tf2onnx  # type: ignore[import-not-found]
    except ImportError:
        LOGGER.warning("Skipping ONNX. Install optional dependency: pip install tf2onnx")
        return None
    try:
        tf2onnx.convert.from_keras(model, output_path=str(destination))
        LOGGER.info("Saved ONNX model: %s", destination)
        return destination
    except (RuntimeError, ValueError, tf.errors.OpError) as error:
        LOGGER.exception("ONNX conversion failed: %s", error)
        return None


def _quantize_input(array: np.ndarray, detail: dict[str, object]) -> np.ndarray:
    """Convert floating image values to the interpreter's declared input dtype."""
    dtype = detail["dtype"]
    if dtype == np.float32:
        return array.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if not scale:
        raise RuntimeError("Quantized TFLite model is missing input quantization data.")
    limits = np.iinfo(dtype)
    quantized = np.round(array / scale + zero_point)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize_output(array: np.ndarray, detail: dict[str, object]) -> np.ndarray:
    """Convert quantized prediction values back to floating-point probabilities."""
    scale, zero_point = detail["quantization"]
    if scale:
        return (array.astype(np.float32) - zero_point) * scale
    return array.astype(np.float32)


def benchmark_tflite(
    model_path: Path, dataset: tf.data.Dataset, warmup_runs: int = 10
) -> dict[str, float]:
    """Measure TFLite accuracy and single-image latency on the test set."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    correct = total = 0
    latencies: list[float] = []
    for images, labels in dataset:
        for image, label in zip(images.numpy(), labels.numpy(), strict=True):
            input_data = _quantize_input(image[np.newaxis, ...], input_detail)
            interpreter.set_tensor(input_detail["index"], input_data)
            start = time.perf_counter()
            interpreter.invoke()
            elapsed_ms = (time.perf_counter() - start) * 1_000
            output = _dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
            prediction = int(np.argmax(output[0]))
            correct += prediction == int(label)
            total += 1
            if total > warmup_runs:
                latencies.append(elapsed_ms)
    if total == 0:
        raise ValueError("The test dataset contains no images.")
    return {
        "accuracy": correct / total,
        "inference_time_ms": float(np.mean(latencies or [0.0])),
        "samples": float(total),
    }


def benchmark_keras(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, float]:
    """Measure baseline Keras accuracy and single-image latency on the same data."""
    correct = total = 0
    latencies: list[float] = []
    for images, labels in dataset:
        for image, label in zip(images.numpy(), labels.numpy(), strict=True):
            image_batch = image[np.newaxis, ...]
            start = time.perf_counter()
            output = model(image_batch, training=False).numpy()[0]
            elapsed_ms = (time.perf_counter() - start) * 1_000
            correct += int(np.argmax(output)) == int(label)
            total += 1
            if total > 10:
                latencies.append(elapsed_ms)
    return {
        "accuracy": correct / total,
        "inference_time_ms": float(np.mean(latencies or [0.0])),
        "samples": float(total),
    }


def process_memory_mb() -> float | None:
    """Return current process RSS when the optional psutil package is present."""
    try:
        import os
        import psutil  # type: ignore[import-not-found]

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except ImportError:
        LOGGER.warning("Memory measurement skipped. Install optional dependency: psutil")
        return None


def add_artifact_record(
    name: str, path: Path, metrics: dict[str, float], memory_mb: float | None
) -> dict[str, object]:
    """Create one uniform row for comparisons and reports."""
    return {
        "model": name,
        "path": str(path.resolve()),
        "model_size_mb": path.stat().st_size / (1024**2),
        "memory_usage_mb": memory_mb,
        **metrics,
    }


def save_charts(records: list[dict[str, object]], destination: Path) -> None:
    """Save a clear three-panel comparison of size, latency, and accuracy."""
    labels = [str(record["model"]) for record in records]
    metrics = [
        ("model_size_mb", "Model size (MB)"),
        ("inference_time_ms", "Inference time (ms/image)"),
        ("accuracy", "Test accuracy"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#2563eb", "#7c3aed", "#059669"]
    for axis, (key, title), color in zip(axes, metrics, colors, strict=True):
        values = [float(record[key]) for record in records]
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title, fontweight="bold")
        axis.tick_params(axis="x", rotation=22)
        if key == "accuracy":
            axis.set_ylim(0, 1)
            axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=3)
        else:
            axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=CONFIG.models_dir / f"{CONFIG.backbone}_best_tuned.keras",
        help="Path to the trained Keras model.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all conversion, quantization, evaluation, and reporting steps."""
    configure_logging()
    arguments = parse_arguments()
    CONFIG.validate_dataset_structure()
    output_dir = CONFIG.outputs_dir / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.model.is_file():
        raise FileNotFoundError(f"Model does not exist: {arguments.model}")

    model = tf.keras.models.load_model(arguments.model, compile=False)
    test_data = dataset_for_evaluation(CONFIG.test_dir, CONFIG)
    records = [
        add_artifact_record(
            "Keras FP32", arguments.model, benchmark_keras(model, test_data), process_memory_mb()
        )
    ]
    for format_name in ("fp32", "fp16", "int8"):
        artifact = convert_tflite(model, output_dir / f"model_{format_name}.tflite", format_name, CONFIG)
        records.append(
            add_artifact_record(
                f"TFLite {format_name.upper()}", artifact,
                benchmark_tflite(artifact, test_data), process_memory_mb()
            )
        )
        gc.collect()
    convert_onnx(model, output_dir / "model.onnx")

    report_path = output_dir / "optimization_comparison.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "optimization_comparison.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    chart_path = output_dir / "optimization_comparison.png"
    save_charts(records, chart_path)
    LOGGER.info("Saved comparison report: %s", report_path)
    LOGGER.info("Saved comparison chart: %s", chart_path)


if __name__ == "__main__":
    main()
