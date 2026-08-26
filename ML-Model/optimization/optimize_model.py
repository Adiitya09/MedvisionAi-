"""Convert, quantize, evaluate, and benchmark trained MedvisionAI models.

Supported formats:
1. Keras FP32 (Original .keras baseline)
2. TensorFlow Lite FP32
3. TensorFlow Lite FP16
4. TensorFlow Lite INT8 (Full INT8 quantization with representative dataset calibration)
5. ONNX (Open Neural Network Exchange via tf2onnx & onnxruntime)

Outputs:
- outputs/optimization/<dataset>/model_fp32.tflite
- outputs/optimization/<dataset>/model_fp16.tflite
- outputs/optimization/<dataset>/model_int8.tflite
- outputs/optimization/<dataset>/model.onnx
- outputs/optimization/<dataset>/optimization_comparison.json
- outputs/optimization/<dataset>/optimization_comparison.csv
- outputs/optimization/<dataset>/optimization_comparison.png
- outputs/optimization/overall_optimization_summary.json
- outputs/optimization/overall_optimization_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

tf.get_logger().setLevel(logging.ERROR)

from config import CONFIG, ProjectConfig
from utils.helpers import load_model, setup_logging

LOGGER = logging.getLogger(__name__)


def dataset_for_evaluation(
    directory: Path, config: ProjectConfig, batch_size: int = 32
) -> tf.data.Dataset:
    """Load labelled images without augmentation for fair benchmarking."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=config.image_size,
        batch_size=batch_size,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)


def representative_dataset(train_dir: Path, config: ProjectConfig) -> Callable[[], object]:
    """Return a bounded image generator used to calibrate INT8 quantization."""
    calibration_data = dataset_for_evaluation(train_dir, config, batch_size=1).take(50)

    def generator() -> object:
        for images, _ in calibration_data:
            yield [tf.cast(images, tf.float32)]

    return generator


def convert_tflite(
    model: tf.keras.Model,
    destination: Path,
    optimization: str,
    train_dir: Path,
    config: ProjectConfig,
) -> tuple[Path, str, str | None]:
    """Convert a Keras model to float32, float16, or INT8 TFLite."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if optimization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset(train_dir, config)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif optimization != "fp32":
            raise ValueError(f"Unsupported TFLite optimization: {optimization}")

        tflite_model = converter.convert()
        destination.write_bytes(tflite_model)
        LOGGER.info(
            "Saved %s TFLite model: %s (%.2f MB)",
            optimization.upper(),
            destination,
            len(tflite_model) / (1024 * 1024),
        )
        return destination, "SUCCESS", None
    except Exception as error:
        LOGGER.exception("TFLite %s conversion failed: %s", optimization, error)
        return destination, "FAILED", str(error)


def convert_onnx(model: tf.keras.Model, destination: Path) -> tuple[Path | None, str, str | None]:
    """Export ONNX format via tf2onnx."""
    try:
        import tf2onnx
    except ImportError as e:
        msg = f"tf2onnx is not installed: {e}"
        LOGGER.warning(msg)
        return None, "FAILED", msg

    try:
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(destination))
        LOGGER.info(
            "Saved ONNX model: %s (%.2f MB)",
            destination,
            destination.stat().st_size / (1024 * 1024),
        )
        return destination, "SUCCESS", None
    except Exception as error:
        LOGGER.exception("ONNX conversion failed: %s", error)
        return None, "FAILED", str(error)


def _quantize_input(array: np.ndarray, detail: dict[str, object]) -> np.ndarray:
    """Convert floating image values to the interpreter's declared input dtype."""
    dtype = detail["dtype"]
    if dtype == np.float32:
        return array.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if not scale:
        return array.astype(dtype)
    limits = np.iinfo(dtype)
    quantized = np.round(array / scale + zero_point)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize_output(array: np.ndarray, detail: dict[str, object]) -> np.ndarray:
    """Convert quantized prediction values back to floating-point probabilities."""
    scale, zero_point = detail["quantization"]
    if scale:
        return (array.astype(np.float32) - zero_point) * scale
    return array.astype(np.float32)


def benchmark_keras_full(
    model: tf.keras.Model, dataset: tf.data.Dataset, warmup_runs: int = 5, timed_runs: int = 30
) -> dict[str, Any]:
    """Evaluate Keras model on the complete test split and measure single-image latency."""
    y_true: list[int] = []
    for _, labels in dataset:
        y_true.extend(labels.numpy().tolist())

    probs = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probs, axis=1).tolist()

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Single-image latency benchmark
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    for _ in range(warmup_runs):
        _ = model(dummy, training=False)
    latencies: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = model(dummy, training=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_time_ms": float(np.mean(latencies or [0.0])),
        "samples": len(y_true),
        "predictions": y_pred,
        "true_labels": y_true,
    }


def benchmark_tflite_full(
    model_path: Path,
    dataset: tf.data.Dataset,
    keras_preds: list[int],
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> dict[str, Any]:
    """Evaluate TFLite model on the complete test split and measure single-image latency."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    input_shape = str(list(input_detail["shape"]))
    output_shape = str(list(output_detail["shape"]))

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataset:
        images_np = images.numpy()
        labels_np = labels.numpy()
        for i in range(len(labels_np)):
            img = images_np[i : i + 1]
            input_data = _quantize_input(img, input_detail)
            interpreter.set_tensor(input_detail["index"], input_data)
            interpreter.invoke()
            output = _dequantize_output(
                interpreter.get_tensor(output_detail["index"]), output_detail
            )
            pred = int(np.argmax(output[0]))
            y_true.append(int(labels_np[i]))
            y_pred.append(pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    agreement = (
        float(np.mean(np.array(y_pred) == np.array(keras_preds))) if keras_preds else 1.0
    )

    # Single-image latency benchmark
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_quant = _quantize_input(dummy, input_detail)
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_detail["index"], dummy_quant)
        interpreter.invoke()
    latencies: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], dummy_quant)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_time_ms": float(np.mean(latencies or [0.0])),
        "samples": len(y_true),
        "prediction_agreement": agreement,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "predictions": y_pred,
    }


def benchmark_onnx_full(
    model_path: Path,
    dataset: tf.data.Dataset,
    keras_preds: list[int],
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> dict[str, Any]:
    """Evaluate ONNX model on the complete test split and measure single-image latency."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name
    output_name = output_meta.name

    input_shape = str(input_meta.shape)
    output_shape = str(output_meta.shape)

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataset:
        images_np = images.numpy().astype(np.float32)
        labels_np = labels.numpy()
        output = session.run([output_name], {input_name: images_np})[0]
        preds = np.argmax(output, axis=1).tolist()
        y_pred.extend(preds)
        y_true.extend(labels_np.tolist())

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    agreement = (
        float(np.mean(np.array(y_pred) == np.array(keras_preds))) if keras_preds else 1.0
    )

    # Single-image latency benchmark
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    for _ in range(warmup_runs):
        _ = session.run([output_name], {input_name: dummy})
    latencies: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_time_ms": float(np.mean(latencies or [0.0])),
        "samples": len(y_true),
        "prediction_agreement": agreement,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "predictions": y_pred,
    }


def process_memory_mb() -> float | None:
    """Return process resident memory in MB."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return None


def save_charts(records: list[dict[str, Any]], destination: Path) -> None:
    """Save 4-panel comparison of size, latency, accuracy, and macro F1."""
    valid_records = [r for r in records if r["status"] == "SUCCESS"]
    if not valid_records:
        return

    labels = [str(r["format"]) for r in valid_records]
    metrics = [
        ("size_mb", "Model Size (MB)", "#2563eb", "%.2f MB"),
        ("inference_time_ms", "Single-Image Latency (ms)", "#7c3aed", "%.2f ms"),
        ("accuracy", "Test Accuracy", "#059669", "%.2%"),
        ("macro_f1", "Macro F1 Score", "#d97706", "%.4f"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, (key, title, color, fmt) in zip(axes.ravel(), metrics, strict=True):
        values = [float(r.get(key, 0.0)) for r in valid_records]
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title, fontweight="bold", fontsize=12)
        axis.tick_params(axis="x", rotation=20)
        if "Accuracy" in title or "F1" in title:
            axis.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            if fmt == "%.4f":
                lbl = f"{val:.4f}"
            elif fmt == "%.2f MB":
                lbl = f"{val:.2f} MB"
            elif fmt == "%.2f ms":
                lbl = f"{val:.2f} ms"
            else:
                lbl = f"{val:.2%}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                lbl,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.grid(axis="y", alpha=0.25, linestyle="--")

    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def optimize_domain(
    dataset_name: str,
    model_path: Path | None = None,
    config: ProjectConfig = CONFIG,
) -> list[dict[str, Any]]:
    """Convert and benchmark all formats for one dataset domain."""
    dataset_key = dataset_name.strip().lower()
    config.get_dataset(dataset_key)
    config.validate_dataset(dataset_key)

    resolved_model_path = (
        Path(model_path) if model_path is not None else config.model_path_for(dataset_key)
    )
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    train_directory = config.split_dir(dataset_key, "train")
    test_directory = config.split_dir(dataset_key, "test")

    output_dir = config.outputs_dir / "optimization" / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "\n=================================================================\n"
        "Optimization started | domain: %s | model: %s | output: %s\n"
        "=================================================================",
        dataset_key.upper(),
        resolved_model_path.name,
        output_dir,
    )

    model = load_model(resolved_model_path)
    test_data = dataset_for_evaluation(test_directory, config)

    # 1. Benchmark Keras baseline
    LOGGER.info("[%s] Benchmarking Keras FP32 baseline on test set...", dataset_key.upper())
    keras_bench = benchmark_keras_full(model, test_data)
    keras_preds = keras_bench["predictions"]
    LOGGER.info(
        "[%s] Keras FP32: Accuracy=%.4f, Macro F1=%.4f, Latency=%.2f ms",
        dataset_key.upper(),
        keras_bench["accuracy"],
        keras_bench["macro_f1"],
        keras_bench["inference_time_ms"],
    )

    records: list[dict[str, Any]] = [
        {
            "dataset": dataset_key,
            "format": "Keras FP32",
            "model_file": resolved_model_path.name,
            "path": str(resolved_model_path.resolve()),
            "status": "SUCCESS",
            "error": None,
            "size_mb": float(resolved_model_path.stat().st_size / (1024**2)),
            "compression_ratio": 1.0,
            "input_shape": str(list(model.input_shape)),
            "output_shape": str(list(model.output_shape)),
            "inference_time_ms": keras_bench["inference_time_ms"],
            "accuracy": keras_bench["accuracy"],
            "macro_f1": keras_bench["macro_f1"],
            "weighted_f1": keras_bench["weighted_f1"],
            "prediction_agreement": 1.0,
            "memory_usage_mb": process_memory_mb(),
            "samples": keras_bench["samples"],
        }
    ]

    # 2. TFLite Conversions & Benchmarking (FP32, FP16, INT8)
    for format_name in ("fp32", "fp16", "int8"):
        tflite_path = output_dir / f"model_{format_name}.tflite"
        LOGGER.info("[%s] Converting to TFLite %s...", dataset_key.upper(), format_name.upper())
        artifact_path, status, error = convert_tflite(
            model,
            tflite_path,
            format_name,
            train_directory,
            config,
        )

        if status == "SUCCESS" and artifact_path.is_file():
            LOGGER.info("[%s] Benchmarking TFLite %s on test set...", dataset_key.upper(), format_name.upper())
            bench = benchmark_tflite_full(artifact_path, test_data, keras_preds)
            LOGGER.info(
                "[%s] TFLite %s: Accuracy=%.4f, Macro F1=%.4f, Agreement=%.2f%%, Latency=%.2f ms, Size=%.2f MB",
                dataset_key.upper(),
                format_name.upper(),
                bench["accuracy"],
                bench["macro_f1"],
                bench["prediction_agreement"] * 100,
                bench["inference_time_ms"],
                artifact_path.stat().st_size / (1024**2),
            )
            records.append(
                {
                    "dataset": dataset_key,
                    "format": f"TFLite {format_name.upper()}",
                    "model_file": artifact_path.name,
                    "path": str(artifact_path.resolve()),
                    "status": "SUCCESS",
                    "error": None,
                    "size_mb": float(artifact_path.stat().st_size / (1024**2)),
                    "compression_ratio": float(
                        records[0]["size_mb"] / (artifact_path.stat().st_size / (1024**2))
                    ),
                    "input_shape": bench["input_shape"],
                    "output_shape": bench["output_shape"],
                    "inference_time_ms": bench["inference_time_ms"],
                    "accuracy": bench["accuracy"],
                    "macro_f1": bench["macro_f1"],
                    "weighted_f1": bench["weighted_f1"],
                    "prediction_agreement": bench["prediction_agreement"],
                    "memory_usage_mb": process_memory_mb(),
                    "samples": bench["samples"],
                }
            )
        else:
            records.append(
                {
                    "dataset": dataset_key,
                    "format": f"TFLite {format_name.upper()}",
                    "model_file": tflite_path.name,
                    "path": str(tflite_path.resolve()),
                    "status": "FAILED",
                    "error": error,
                    "size_mb": 0.0,
                    "compression_ratio": 0.0,
                    "input_shape": "N/A",
                    "output_shape": "N/A",
                    "inference_time_ms": 0.0,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "weighted_f1": 0.0,
                    "prediction_agreement": 0.0,
                    "memory_usage_mb": process_memory_mb(),
                    "samples": 0,
                }
            )
        gc.collect()

    # 3. ONNX Conversion & Benchmarking
    onnx_path = output_dir / "model.onnx"
    LOGGER.info("[%s] Converting to ONNX...", dataset_key.upper())
    onnx_artifact, onnx_status, onnx_error = convert_onnx(model, onnx_path)
    if onnx_status == "SUCCESS" and onnx_artifact and onnx_artifact.is_file():
        LOGGER.info("[%s] Benchmarking ONNX model on test set...", dataset_key.upper())
        onnx_bench = benchmark_onnx_full(onnx_artifact, test_data, keras_preds)
        LOGGER.info(
            "[%s] ONNX: Accuracy=%.4f, Macro F1=%.4f, Agreement=%.2f%%, Latency=%.2f ms, Size=%.2f MB",
            dataset_key.upper(),
            onnx_bench["accuracy"],
            onnx_bench["macro_f1"],
            onnx_bench["prediction_agreement"] * 100,
            onnx_bench["inference_time_ms"],
            onnx_artifact.stat().st_size / (1024**2),
        )
        records.append(
            {
                "dataset": dataset_key,
                "format": "ONNX",
                "model_file": onnx_artifact.name,
                "path": str(onnx_artifact.resolve()),
                "status": "SUCCESS",
                "error": None,
                "size_mb": float(onnx_artifact.stat().st_size / (1024**2)),
                "compression_ratio": float(
                    records[0]["size_mb"] / (onnx_artifact.stat().st_size / (1024**2))
                ),
                "input_shape": onnx_bench["input_shape"],
                "output_shape": onnx_bench["output_shape"],
                "inference_time_ms": onnx_bench["inference_time_ms"],
                "accuracy": onnx_bench["accuracy"],
                "macro_f1": onnx_bench["macro_f1"],
                "weighted_f1": onnx_bench["weighted_f1"],
                "prediction_agreement": onnx_bench["prediction_agreement"],
                "memory_usage_mb": process_memory_mb(),
                "samples": onnx_bench["samples"],
            }
        )
    else:
        records.append(
            {
                "dataset": dataset_key,
                "format": "ONNX",
                "model_file": onnx_path.name,
                "path": str(onnx_path.resolve()),
                "status": "FAILED",
                "error": onnx_error,
                "size_mb": 0.0,
                "compression_ratio": 0.0,
                "input_shape": "N/A",
                "output_shape": "N/A",
                "inference_time_ms": 0.0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "prediction_agreement": 0.0,
                "memory_usage_mb": process_memory_mb(),
                "samples": 0,
            }
        )

    # Save CSV
    report_path = output_dir / "optimization_comparison.csv"
    csv_fields = [
        "dataset",
        "format",
        "model_file",
        "status",
        "size_mb",
        "compression_ratio",
        "inference_time_ms",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "prediction_agreement",
        "input_shape",
        "output_shape",
        "memory_usage_mb",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Save JSON
    json_path = output_dir / "optimization_comparison.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Save Charts
    chart_path = output_dir / "optimization_comparison.png"
    save_charts(records, chart_path)

    LOGGER.info("[%s] Saved comparison report: %s", dataset_key.upper(), report_path)
    LOGGER.info("[%s] Saved comparison JSON: %s", dataset_key.upper(), json_path)
    LOGGER.info("[%s] Saved comparison chart: %s", dataset_key.upper(), chart_path)

    return records


def run_all_optimizations() -> list[dict[str, Any]]:
    """Run optimization pipeline for all 3 selected models."""
    logger = setup_logging()
    logger.info("=================================================================")
    logger.info("STARTING MODEL OPTIMIZATION PIPELINE ACROSS ALL DOMAINS")
    logger.info("=================================================================")

    target_specs = [
        {
            "dataset": "skin",
            "model_path": PROJECT_ROOT / "models" / "skin_model.keras",
        },
        {
            "dataset": "eye",
            "model_path": PROJECT_ROOT / "models" / "eye_model.keras",
        },
        {
            "dataset": "oral",
            "model_path": PROJECT_ROOT / "models" / "tuning" / "oral" / "efficientnetb0_best_tuned.keras",
        },
    ]

    all_records: list[dict[str, Any]] = []

    for spec in target_specs:
        domain_records = optimize_domain(
            dataset_name=spec["dataset"],
            model_path=spec["model_path"],
            config=CONFIG,
        )
        all_records.extend(domain_records)

    # Save overall summary
    opt_base_dir = CONFIG.outputs_dir / "optimization"
    summary_json = opt_base_dir / "overall_optimization_summary.json"
    summary_json.write_text(json.dumps(all_records, indent=2), encoding="utf-8")

    summary_csv = opt_base_dir / "overall_optimization_summary.csv"
    csv_fields = [
        "dataset",
        "format",
        "model_file",
        "status",
        "size_mb",
        "compression_ratio",
        "inference_time_ms",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "prediction_agreement",
        "input_shape",
        "output_shape",
        "error",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    # Print summary table
    print("\n" + "=" * 135)
    print("MODEL OPTIMIZATION SUMMARY TABLE")
    print("=" * 135)
    print(
        f"{'Dataset':<8} | {'Format':<14} | {'Size (MB)':<10} | {'Latency (ms)':<14} | {'Accuracy':<10} | {'Macro F1':<10} | {'Agreement':<12} | {'Status':<8}"
    )
    print("-" * 135)
    for r in all_records:
        size_str = f"{r['size_mb']:.2f} MB" if r["status"] == "SUCCESS" else "N/A"
        lat_str = f"{r['inference_time_ms']:.2f} ms" if r["status"] == "SUCCESS" else "N/A"
        acc_str = f"{r['accuracy']:.4f}" if r["status"] == "SUCCESS" else "N/A"
        f1_str = f"{r['macro_f1']:.4f}" if r["status"] == "SUCCESS" else "N/A"
        agree_str = f"{r['prediction_agreement']*100:.2f}%" if r["status"] == "SUCCESS" else "N/A"
        print(
            f"{r['dataset'].upper():<8} | {r['format']:<14} | {size_str:<10} | {lat_str:<14} | {acc_str:<10} | {f1_str:<10} | {agree_str:<12} | {r['status']:<8}"
        )
    print("=" * 135)

    return all_records


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral", "all"),
        default="all",
        help="Disease domain whose configured model and dataset splits are optimized (default: all).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the trained Keras model (default: domain selected model).",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run all conversion, quantization, evaluation, and reporting steps."""
    try:
        args = parse_arguments(arguments)
        if args.dataset == "all":
            run_all_optimizations()
        else:
            optimize_domain(args.dataset, model_path=args.model)
    except KeyboardInterrupt:
        LOGGER.warning("Optimization cancelled by user.")
        return 130
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("Optimization validation error: %s", error)
        print(f"Optimization validation error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        LOGGER.exception("Optimization failed.")
        print(f"Optimization failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
