"""Generate publication-quality explanations for disease-image predictions.

Example:
    python explainability.py --image path/to/image.jpg --true-label pneumonia

The module supports models trained by this project and saves both a high-
resolution figure and a JSON prediction record under ``outputs/explainability``.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CONFIG, ProjectConfig


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure timestamped logging for command-line usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_class_names(path: Path) -> list[str]:
    """Read the persisted class-index mapping and validate its contents."""
    try:
        class_names = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to load class names from {path}") from error
    if not isinstance(class_names, list) or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError("Class names must be a JSON list of strings.")
    return class_names


def load_image(image_path: Path, config: ProjectConfig) -> tuple[tf.Tensor, np.ndarray]:
    """Load an RGB image for inference and a normalized copy for plotting."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        image_bytes = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, config.image_size)
    except tf.errors.OpError as error:
        raise ValueError(f"Could not decode image: {image_path}") from error

    image = tf.cast(image, tf.float32)
    batch = tf.expand_dims(image, axis=0)
    return batch, np.clip(image.numpy() / 255.0, 0.0, 1.0)


def _find_last_convolution(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Find the final convolutional layer, including inside nested backbones."""
    candidates: list[tf.keras.layers.Layer] = []

    def visit(layer: tf.keras.layers.Layer) -> None:
        if isinstance(layer, tf.keras.layers.Conv2D):
            candidates.append(layer)
        if isinstance(layer, tf.keras.Model):
            for nested_layer in layer.layers:
                visit(nested_layer)

    for model_layer in model.layers:
        visit(model_layer)
    if not candidates:
        raise ValueError("No Conv2D layer was found; Grad-CAM cannot be generated.")
    return candidates[-1]


def grad_cam(
    model: tf.keras.Model, image_batch: tf.Tensor, class_index: int
) -> np.ndarray:
    """Create a normalized Grad-CAM heatmap for one model prediction."""
    last_conv_layer = _find_last_convolution(model)
    gradient_model = tf.keras.Model(
        inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradient_model(image_batch, training=False)
        target_score = predictions[:, class_index]
    gradients = tape.gradient(target_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients for this model.")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def integrated_gradients(
    model: tf.keras.Model,
    image_batch: tf.Tensor,
    class_index: int,
    steps: int = 64,
) -> np.ndarray:
    """Compute a normalized Integrated Gradients attribution map.

    A black image is used as the baseline.  This method measures how changing
    pixels along the path from baseline to image affects the predicted class.
    """
    if steps < 2:
        raise ValueError("Integrated Gradients needs at least two interpolation steps.")
    baseline = tf.zeros_like(image_batch)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated = baseline + alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * (
        image_batch - baseline
    )

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = model(interpolated, training=False)
        target_scores = predictions[:, class_index]
    gradients = tape.gradient(target_scores, interpolated)
    if gradients is None:
        raise RuntimeError("Could not calculate Integrated Gradients for this model.")

    # Trapezoidal integration over the straight-line baseline-to-image path.
    average_gradients = (gradients[:-1] + gradients[1:]) / 2.0
    integrated = (image_batch[0] - baseline[0]) * tf.reduce_mean(
        average_gradients, axis=0
    )
    attribution = tf.reduce_sum(tf.abs(integrated), axis=-1)
    maximum = tf.reduce_max(attribution)
    attribution = tf.where(maximum > 0, attribution / maximum, attribution)
    return attribution.numpy()


def resize_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D attribution array to an image's pixel dimensions."""
    resized = tf.image.resize(heatmap[..., np.newaxis], target_size)
    return tf.squeeze(resized, axis=-1).numpy()


def save_visualization(
    original_image: np.ndarray,
    gradcam_map: np.ndarray,
    integrated_gradients_map: np.ndarray,
    predicted_label: str,
    confidence: float,
    output_path: Path,
    true_label: str | None = None,
) -> None:
    """Save an annotated, publication-quality explanation figure."""
    display_gradcam = resize_heatmap(gradcam_map, original_image.shape[:2])
    display_ig = resize_heatmap(integrated_gradients_map, original_image.shape[:2])
    overlay = np.clip(
        0.55 * original_image
        + 0.45 * plt.colormaps["jet"](display_gradcam)[..., :3],
        0.0,
        1.0,
    )
    title = f"Prediction: {predicted_label} ({confidence:.2%})"
    if true_label:
        title += f"  |  True label: {true_label}"

    plt.style.use("seaborn-v0_8-white")
    figure, axes = plt.subplots(1, 4, figsize=(18, 5.2), constrained_layout=True)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    panels = (
        (original_image, "Original Image", None),
        (display_gradcam, "Grad-CAM Heatmap", "jet"),
        (overlay, "Grad-CAM Overlay", None),
        (display_ig, "Integrated Gradients", "magma"),
    )
    for axis, (content, panel_title, colormap) in zip(axes, panels, strict=True):
        rendered = axis.imshow(content, cmap=colormap, vmin=0, vmax=1)
        axis.set_title(panel_title, fontsize=13, fontweight="semibold")
        axis.axis("off")
        if colormap:
            colorbar = figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Attribution", fontsize=10)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def explain_image(
    image_path: Path,
    model_path: Path,
    class_names_path: Path,
    true_label: str | None,
    config: ProjectConfig,
) -> tuple[Path, Path]:
    """Predict and save Grad-CAM plus Integrated Gradients explanations."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Keras model does not exist: {model_path}")
    class_names = load_class_names(class_names_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    image_batch, original_image = load_image(image_path, config)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    if predicted_index >= len(class_names):
        raise ValueError("Model output size does not match the saved class names.")

    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    gradcam_map = grad_cam(model, image_batch, predicted_index)
    ig_map = integrated_gradients(model, image_batch, predicted_index)

    output_dir = config.outputs_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{image_path.stem}_{datetime.now():%Y%m%d-%H%M%S}"
    figure_path = output_dir / f"{run_id}_explanation.png"
    metadata_path = output_dir / f"{run_id}_prediction.json"
    save_visualization(
        original_image, gradcam_map, ig_map, predicted_label, confidence, figure_path, true_label
    )
    metadata = {
        "image": str(image_path.resolve()),
        "model": str(model_path.resolve()),
        "predicted_disease": predicted_label,
        "confidence": confidence,
        "true_disease": true_label,
        "top_5_predictions": [
            {"disease": class_names[index], "confidence": float(probabilities[index])}
            for index in np.argsort(probabilities)[::-1][:5]
        ],
        "visualization": str(figure_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved explanation figure: %s", figure_path)
    LOGGER.info("Saved explanation metadata: %s", metadata_path)
    return figure_path, metadata_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for standalone inference explanations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to one image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=CONFIG.models_dir / f"{CONFIG.backbone}_best_tuned.keras",
        help="Path to a trained .keras model.",
    )
    parser.add_argument(
        "--class-names", type=Path, default=CONFIG.class_names_path,
        help="Path to the class_names.json generated during training.",
    )
    parser.add_argument(
        "--true-label", default=None, help="Known disease label, if available."
    )
    return parser.parse_args()


def main() -> None:
    """Run explainability from the command line."""
    configure_logging()
    arguments = parse_arguments()
    explain_image(
        image_path=arguments.image,
        model_path=arguments.model,
        class_names_path=arguments.class_names,
        true_label=arguments.true_label,
        config=CONFIG,
    )


if __name__ == "__main__":
    main()
