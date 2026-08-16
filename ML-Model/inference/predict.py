"""Run single-image inference with a trained MedvisionAI Keras model.

Example:
    python inference/predict.py --dataset skin --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Support direct execution from Windows, Google Colab, and Lightning AI.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config import CONFIG
from utils.helpers import load_model, setup_logging


LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _validate_image_path(image_path: str | Path) -> Path:
    """Validate a supported, readable image path and return its resolved path."""
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be a string or pathlib.Path.")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file was not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image extension '{path.suffix}'. Use: {supported}.")
    return path.resolve()


def _load_class_names(dataset_name: str) -> list[str]:
    """Load the configured class-index-to-name mapping for a disease domain.

    The mapping file may be either a JSON list ordered by class index or a JSON
    object whose keys are integer class indices.
    """
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
        class_names = mapping
    elif isinstance(mapping, dict):
        try:
            ordered_indices = range(len(mapping))
            class_names = [mapping[str(index)] for index in ordered_indices]
        except KeyError as error:
            raise ValueError(
                "Class-name mapping keys must be consecutive indices starting at zero."
            ) from error
    else:
        raise ValueError("Class-name mapping must be a JSON list or indexed JSON object.")

    if len(class_names) < 2 or not all(isinstance(name, str) and name.strip() for name in class_names):
        raise ValueError("Class-name mapping must contain at least two non-empty class names.")
    return list(class_names)


def _load_and_prepare_image(image_path: Path) -> tf.Tensor:
    """Load one RGB image and resize it to the configured model input size.

    Models trained by this project contain their preprocessing layer. Therefore
    this function returns unnormalized float32 pixel values in the 0-255 range;
    applying backbone preprocessing again here would distort predictions.
    """
    try:
        image = tf.keras.utils.load_img(
            image_path,
            color_mode="rgb",
            target_size=CONFIG.image_size,
        )
        array = tf.keras.utils.img_to_array(image, dtype="float32")
    except (OSError, ValueError, tf.errors.OpError) as error:
        raise ValueError(f"Unable to load image: {image_path}") from error

    if array.shape != CONFIG.input_shape:
        raise ValueError(
            "Prepared image has an unexpected shape: "
            f"{array.shape}; expected {CONFIG.input_shape}."
        )
    return tf.expand_dims(array, axis=0)


def _validate_model_shapes(model: tf.keras.Model, class_names: Sequence[str]) -> None:
    """Ensure the saved model input/output agree with project configuration."""
    input_shape = model.input_shape
    if isinstance(input_shape, list) or tuple(input_shape[1:]) != CONFIG.input_shape:
        raise ValueError(
            "Model input shape does not match CONFIG.input_shape "
            f"({input_shape} model input, {CONFIG.input_shape} configured input)."
        )
    output_shape = model.output_shape
    if isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError("The saved model must have one two-dimensional probability output.")
    class_count = output_shape[-1]
    if not isinstance(class_count, int) or class_count != len(class_names):
        raise ValueError(
            "Model output class count does not match the class-name mapping "
            f"({class_count} model outputs, {len(class_names)} names)."
        )


def _top_predictions(probabilities: np.ndarray, class_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return the highest three class predictions with probabilities."""
    top_count = min(3, len(class_names))
    indices = np.argsort(probabilities)[::-1][:top_count]
    return [
        {
            "class_index": int(index),
            "class_name": class_names[int(index)],
            "probability": float(probabilities[int(index)]),
        }
        for index in indices
    ]


def _save_prediction(result: dict[str, Any], dataset_name: str) -> Path:
    """Save one prediction result beneath the configured outputs directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = CONFIG.outputs_dir / "predictions" / dataset_name
    output_path = output_dir / f"prediction_{timestamp}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError as error:
        LOGGER.exception("Unable to save prediction result.")
        raise RuntimeError(f"Unable to save prediction result: {output_path}") from error
    LOGGER.info("Prediction result saved: %s", output_path)
    return output_path


def predict_image(image_path: str, dataset_name: str) -> dict[str, Any]:
    """Predict a disease class for one image using its configured trained model.

    Args:
        image_path: Path to a JPEG, PNG, WebP, or BMP medical image.
        dataset_name: Domain selection: ``skin``, ``eye``, or ``oral``.

    Returns:
        Dataset, model, top prediction, confidence, and the three most likely
        classes with their probabilities.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    dataset_key = dataset_name.strip().lower()
    CONFIG.get_dataset(dataset_key)
    resolved_image_path = _validate_image_path(image_path)
    model_path = CONFIG.model_path_for(dataset_key)
    class_names = _load_class_names(dataset_key)

    LOGGER.info(
        "Inference started | dataset: %s | image: %s | model: %s",
        dataset_key,
        resolved_image_path,
        model_path,
    )
    model = load_model(model_path)
    _validate_model_shapes(model, class_names)
    image_batch = _load_and_prepare_image(resolved_image_path)
    try:
        predictions = np.asarray(model.predict(image_batch, verbose=0), dtype=np.float64)
    except (tf.errors.OpError, TypeError, ValueError) as error:
        LOGGER.exception("Model prediction failed.")
        raise RuntimeError("Unable to run inference with the loaded model.") from error

    if predictions.shape != (1, len(class_names)) or not np.all(np.isfinite(predictions)):
        raise ValueError("Model returned invalid class probabilities for the input image.")
    probabilities = predictions[0]
    top_predictions = _top_predictions(probabilities, class_names)
    best_prediction = top_predictions[0]
    result: dict[str, Any] = {
        "image_path": str(resolved_image_path),
        "dataset": dataset_key,
        "model": model_path.name,
        "predicted_class": best_prediction["class_name"],
        "confidence": best_prediction["probability"],
        "top_predictions": top_predictions,
    }
    LOGGER.info(
        "Inference completed | predicted class: %s | confidence: %.2f%%",
        result["predicted_class"],
        result["confidence"] * 100,
    )
    result["prediction_path"] = str(_save_prediction(result, dataset_key))
    return result


def _print_prediction(result: dict[str, Any]) -> None:
    """Print a compact, human-readable prediction summary."""
    print(
        f"Dataset: {result['dataset']}\n"
        f"Model: {result['model']}\n\n"
        f"Prediction:\n{result['predicted_class']}\n\n"
        f"Confidence:\n{result['confidence'] * 100:.2f}%\n\n"
        "Top predictions:"
    )
    for position, prediction in enumerate(result["top_predictions"], start=1):
        print(
            f"{position}. {prediction['class_name']:<30} "
            f"{prediction['probability'] * 100:.2f}%"
        )


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse single-image inference options."""
    parser = argparse.ArgumentParser(description="Predict a disease class from one image.")
    parser.add_argument(
        "--dataset",
        choices=("skin", "eye", "oral"),
        required=True,
        help="Disease domain associated with the trained model.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a supported image file.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line inference flow and return a process exit status."""
    setup_logging()
    try:
        args = _parse_arguments(arguments)
        result = predict_image(args.image, args.dataset)
        _print_prediction(result)
    except KeyboardInterrupt:
        LOGGER.warning("Inference cancelled by user.")
        return 130
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        LOGGER.error("Inference validation error: %s", error)
        print(f"Inference validation error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        LOGGER.exception("Inference failed.")
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
