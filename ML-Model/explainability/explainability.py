"""Generate Grad-CAM visual explanations for one trained MedvisionAI model.

Grad-CAM highlights image features that influenced a model output. It is not
medical proof of causality and must not be interpreted as a diagnosis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np


# Support direct execution without assuming a project-root working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config import CONFIG
from inference.predict import (
    _load_and_prepare_image,
    _load_class_names,
    _validate_image_path,
    _validate_model_shapes,
)
from utils.helpers import load_model, setup_logging


LOGGER = logging.getLogger(__name__)


def _call_for_inference(layer: tf.keras.layers.Layer, inputs: tf.Tensor) -> tf.Tensor:
    """Call a Keras layer with inference mode when it accepts ``training``."""
    try:
        return layer(inputs, training=False)
    except TypeError:
        return layer(inputs)


def _find_target_layer(model: tf.keras.Model) -> tuple[tf.keras.Model, tf.keras.layers.Layer]:
    """Find the final spatial convolutional layer inside the trained backbone.

    This works across MobileNetV2, EfficientNetB0, and ResNet50 because it
    searches layer types and output shapes instead of architecture-specific
    names.
    """
    try:
        backbone = model.get_layer("backbone")
    except ValueError as error:
        raise ValueError(
            "Grad-CAM requires a model with the standardized 'backbone' layer. "
            "Inspect model.summary() to identify a compatible convolutional model."
        ) from error
    if not isinstance(backbone, tf.keras.Model):
        raise ValueError("The model's 'backbone' layer is not a Keras model.")

    convolution_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in reversed(backbone.layers):
        output_shape = getattr(layer.output, "shape", ())
        if isinstance(layer, convolution_types) and len(output_shape) == 4:
            return backbone, layer
    raise ValueError(
        "No spatial convolutional feature layer was found in the backbone. "
        "Inspect model.summary() and ensure the saved model uses MobileNetV2, "
        "EfficientNetB0, or ResNet50 with include_top=False."
    )


def _build_gradcam_model(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    target_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    """Create a GradientTape-compatible model yielding feature maps and scores."""
    try:
        feature_model = tf.keras.Model(
            backbone.input,
            [target_layer.output, backbone.output],
            name="gradcam_feature_extractor",
        )
        backbone_index = model.layers.index(backbone)
    except (AttributeError, ValueError) as error:
        raise ValueError("Unable to construct a Grad-CAM feature extractor.") from error

    inputs = model.input
    features = inputs
    for layer in model.layers[1:backbone_index]:
        features = _call_for_inference(layer, features)
    convolution_outputs, features = _call_for_inference(feature_model, features)
    for layer in model.layers[backbone_index + 1 :]:
        features = _call_for_inference(layer, features)
    return tf.keras.Model(inputs, [convolution_outputs, features], name="gradcam_model")


def _create_heatmap(
    gradcam_model: tf.keras.Model,
    image_batch: tf.Tensor,
    target_class_index: int,
) -> np.ndarray:
    """Generate a normalized Grad-CAM heatmap for one target class."""
    try:
        with tf.GradientTape() as tape:
            convolution_outputs, predictions = gradcam_model(image_batch, training=False)
            class_score = predictions[:, target_class_index]
        gradients = tape.gradient(class_score, convolution_outputs)
    except (tf.errors.OpError, TypeError, ValueError) as error:
        raise RuntimeError("GradientTape could not compute the Grad-CAM heatmap.") from error

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable for the selected layer.")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(convolution_outputs[0] * pooled_gradients, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    if float(maximum) == 0.0:
        raise RuntimeError("Grad-CAM heatmap is empty for the predicted class.")
    heatmap = heatmap / maximum
    resized_heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        CONFIG.image_size,
        method="bilinear",
    )[..., 0]
    return np.asarray(resized_heatmap, dtype=np.float32)


def _save_visuals(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    dataset_name: str,
) -> tuple[Path, Path, Path]:
    """Save the resized image, colored heatmap, and transparent overlay."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = CONFIG.outputs_dir / "explainability" / dataset_name
    original_path = output_directory / f"{timestamp}_original.png"
    heatmap_path = output_directory / f"{timestamp}_heatmap.png"
    overlay_path = output_directory / f"{timestamp}_overlay.png"
    color_map = matplotlib.colormaps["jet"]
    colored_heatmap = np.asarray(color_map(heatmap)[..., :3] * 255, dtype=np.uint8)
    overlay = np.asarray(
        np.clip(0.60 * original_image + 0.40 * colored_heatmap, 0, 255),
        dtype=np.uint8,
    )
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        tf.keras.utils.save_img(original_path, original_image)
        tf.keras.utils.save_img(heatmap_path, colored_heatmap)
        tf.keras.utils.save_img(overlay_path, overlay)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to save Grad-CAM images in: {output_directory}") from error
    return original_path, heatmap_path, overlay_path


def generate_gradcam(image_path: str, dataset_name: str) -> dict[str, Any]:
    """Generate a Grad-CAM explanation for the predicted class of one image.

    Args:
        image_path: Supported RGB medical image path.
        dataset_name: One configured domain: skin, eye, or oral.

    Returns:
        Prediction metadata, selected convolutional layer, and saved artifact
        paths. The heatmap explains model features, not medical causality.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    dataset_key = dataset_name.strip().lower()
    CONFIG.get_dataset(dataset_key)
    resolved_image_path = _validate_image_path(image_path)
    model_path = CONFIG.model_path_for(dataset_key)
    class_names = _load_class_names(dataset_key)
    logger = setup_logging()
    logger.info(
        "Grad-CAM started | dataset: %s | image: %s | model: %s",
        dataset_key,
        resolved_image_path,
        model_path,
    )

    model = load_model(model_path)
    _validate_model_shapes(model, class_names)
    image_batch = _load_and_prepare_image(resolved_image_path)
    try:
        probabilities = np.asarray(model.predict(image_batch, verbose=0), dtype=np.float64)
    except (tf.errors.OpError, TypeError, ValueError) as error:
        raise RuntimeError("Unable to generate a prediction for Grad-CAM.") from error
    if probabilities.shape != (1, len(class_names)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid probabilities for the input image.")

    target_class_index = int(np.argmax(probabilities[0]))
    backbone, target_layer = _find_target_layer(model)
    gradcam_model = _build_gradcam_model(model, backbone, target_layer)
    heatmap = _create_heatmap(gradcam_model, image_batch, target_class_index)
    original_image = np.asarray(image_batch[0], dtype=np.uint8)
    original_path, heatmap_path, overlay_path = _save_visuals(
        original_image,
        heatmap,
        dataset_key,
    )
    result: dict[str, Any] = {
        "image_path": str(resolved_image_path),
        "dataset": dataset_key,
        "model": model_path.name,
        "predicted_class": class_names[target_class_index],
        "confidence": float(probabilities[0, target_class_index]),
        "target_class_index": target_class_index,
        "target_layer": target_layer.name,
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }
    json_path = original_path.with_name(original_path.name.replace("_original.png", "_result.json"))
    try:
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unable to save Grad-CAM metadata: {json_path}") from error
    result["result_path"] = str(json_path)
    logger.info(
        "Grad-CAM completed | class: %s | confidence: %.2f%% | target layer: %s",
        result["predicted_class"],
        result["confidence"] * 100,
        result["target_layer"],
    )
    return result


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse an explicitly requested Grad-CAM image explanation."""
    parser = argparse.ArgumentParser(description="Generate a model Grad-CAM explanation.")
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        required=True,
        help="Disease domain associated with the trained model.",
    )
    parser.add_argument("--image", required=True, help="Path to a supported image file.")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run Grad-CAM generation and return a conventional process exit code."""
    try:
        args = _parse_arguments(arguments)
        result = generate_gradcam(args.image, args.dataset)
    except KeyboardInterrupt:
        LOGGER.warning("Grad-CAM generation cancelled by user.")
        return 130
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        LOGGER.error("Grad-CAM validation error: %s", error)
        print(f"Grad-CAM validation error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        LOGGER.exception("Grad-CAM generation failed.")
        print(f"Grad-CAM generation failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Prediction: {result['predicted_class']}\n"
        f"Confidence: {result['confidence'] * 100:.2f}%\n"
        f"Target layer: {result['target_layer']}\n"
        f"Overlay: {result['overlay_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
