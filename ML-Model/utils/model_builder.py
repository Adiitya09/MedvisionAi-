"""Reusable transfer-learning model factory for medical image classification."""

from __future__ import annotations

#=========================
# Remove Warnings
#=========================
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

#===========================
#===========================

import logging
from collections.abc import Callable, Sequence
from typing import Final

import tensorflow as tf

from config import CONFIG


LOGGER = logging.getLogger(__name__)

_MODEL_ALIASES: Final[dict[str, str]] = {
    "mobilenetv2": "mobilenetv2",
    "mobile_net_v2": "mobilenetv2",
    "efficientnetb0": "efficientnetb0",
    "efficient_net_b0": "efficientnetb0",
    "resnet50": "resnet50",
    "resnet_50": "resnet50",
}


def _normalise_model_name(model_name: str) -> str:
    """Return a supported canonical model name.

    Raises:
        ValueError: If ``model_name`` does not identify a supported backbone.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("A non-empty model name is required.")

    normalised = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _MODEL_ALIASES[normalised]
    except KeyError as error:
        supported = ", ".join(("MobileNetV2", "EfficientNetB0", "ResNet50"))
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}."
        ) from error


def _require_config_value(attribute: str) -> object:
    """Return a required CONFIG setting with a clear error if it is absent."""
    try:
        value = getattr(CONFIG, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"Missing required configuration: CONFIG.{attribute}."
        ) from error

    if value is None:
        raise ValueError(f"CONFIG.{attribute} must not be None.")
    return value


def _get_model_spec(
    model_name: str,
) -> tuple[Callable[..., tf.keras.Model], Callable[[tf.Tensor], tf.Tensor]]:
    """Return the configured Keras application factory and preprocessing function."""
    applications = tf.keras.applications
    specifications = {
        "mobilenetv2": (
            applications.MobileNetV2,
            applications.mobilenet_v2.preprocess_input,
        ),
        "efficientnetb0": (
            applications.EfficientNetB0,
            applications.efficientnet.preprocess_input,
        ),
        "resnet50": (
            applications.ResNet50,
            applications.resnet50.preprocess_input,
        ),
    }
    return specifications[_normalise_model_name(model_name)]


def get_preprocessing_function(
    model_name: str | None = None,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the model-specific ImageNet preprocessing function.

    Args:
        model_name: Optional architecture name. When omitted, uses
            ``CONFIG.backbone``.
    """
    selected_name = model_name or str(_require_config_value("backbone"))
    _, preprocessing_function = _get_model_spec(selected_name)
    return preprocessing_function


def _build_augmentation_layer() -> tf.keras.Sequential:
    """Build the image augmentation pipeline applied only during training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.20),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )


def _build_optimizer() -> tf.keras.optimizers.Optimizer:
    """Build the optimizer selected through ``CONFIG.optimizer``.

    ``CONFIG.initial_learning_rate`` is used for the initial frozen-backbone
    phase. Supported values are Adam, SGD, and RMSprop (case-insensitive).
    """
    optimizer_name = str(_require_config_value("optimizer")).strip().lower()
    learning_rate = _require_config_value("initial_learning_rate")

    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("CONFIG.initial_learning_rate must be a positive number.") from error

    if learning_rate <= 0:
        raise ValueError("CONFIG.initial_learning_rate must be greater than zero.")

    optimizer_factories: dict[str, type[tf.keras.optimizers.Optimizer]] = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    try:
        return optimizer_factories[optimizer_name](learning_rate=learning_rate)
    except KeyError as error:
        supported = ", ".join(optimizer_factories)
        raise ValueError(
            f"Unsupported CONFIG.optimizer '{optimizer_name}'. Use: {supported}."
        ) from error


def _parameter_count(weights: Sequence[tf.Variable]) -> int:
    """Return the scalar parameter count for a collection of Keras weights."""
    return int(sum(tf.keras.backend.count_params(weight) for weight in weights))


def build_model(model_name: str | None, num_classes: int) -> tf.keras.Model:
    """Build and compile a frozen ImageNet transfer-learning classifier.

    Args:
        model_name: Architecture name, or ``None`` to select ``CONFIG.backbone``.
        num_classes: Number of class labels detected by the dataset loader.

    Returns:
        A compiled Keras model with a frozen named ``backbone`` layer.

    Raises:
        ValueError: If the model name, classes, or configuration is invalid.
        RuntimeError: If pretrained backbone construction fails.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer greater than or equal to 2.")

    selected_name = model_name or str(_require_config_value("backbone"))
    canonical_name = _normalise_model_name(selected_name)
    input_shape = _require_config_value("input_shape")
    dropout_rate = _require_config_value("dropout_rate")
    loss = _require_config_value("loss_function")
    metrics = _require_config_value("metrics")

    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError("CONFIG.input_shape must be a three-item (height, width, channels) tuple.")
    if input_shape[-1] != 3:
        raise ValueError("CONFIG.input_shape must specify three RGB channels.")
    if not isinstance(dropout_rate, (int, float)) or not 0 <= dropout_rate < 1:
        raise ValueError("CONFIG.dropout_rate must be a number in the range [0, 1).")
    if not isinstance(metrics, (list, tuple)) or not metrics:
        raise ValueError("CONFIG.metrics must be a non-empty list or tuple.")

    backbone_factory, preprocessing_function = _get_model_spec(canonical_name)
    try:
        base_model = backbone_factory(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    except Exception as error:
        LOGGER.exception("Unable to initialise the %s ImageNet backbone.", canonical_name)
        raise RuntimeError(
            f"Unable to initialise the '{canonical_name}' ImageNet backbone."
        ) from error

    # Wrap only after ImageNet weights have loaded; naming an application during
    # construction changes Keras' internally selected pretrained-weight file.
    backbone = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output,
        name="backbone",
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _build_augmentation_layer()(inputs)
    x = tf.keras.layers.Lambda(preprocessing_function, name="preprocessing")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(float(dropout_rate), name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{canonical_name}_classifier")

    optimizer = _build_optimizer()
    model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))
    trainable_parameters = _parameter_count(model.trainable_weights)
    LOGGER.info(
        "Built %s | input: %s | classes: %d | trainable parameters: %d | "
        "optimizer: %s | learning rate: %s",
        canonical_name,
        input_shape,
        num_classes,
        trainable_parameters,
        optimizer.__class__.__name__,
        optimizer.learning_rate.numpy(),
    )
    return model


def _get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Get the named backbone layer from a model built by ``build_model``."""
    try:
        backbone = model.get_layer("backbone")
    except (AttributeError, ValueError) as error:
        raise ValueError("Model does not contain a 'backbone' layer.") from error
    if not isinstance(backbone, tf.keras.Model):
        raise ValueError("The 'backbone' layer is not a Keras model.")
    return backbone


def freeze_backbone(model: tf.keras.Model) -> None:
    """Freeze every backbone layer before initial transfer-learning training."""
    backbone = _get_backbone(model)
    backbone.trainable = False
    LOGGER.info("Backbone '%s' is frozen.", backbone.name)


def unfreeze_backbone(model: tf.keras.Model, num_layers: int) -> None:
    """Unfreeze only the final ``num_layers`` backbone layers for fine tuning.

    Args:
        model: A model returned by :func:`build_model`.
        num_layers: Positive count of final backbone layers to make trainable.
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("num_layers must be a positive integer.")

    backbone = _get_backbone(model)
    if num_layers > len(backbone.layers):
        raise ValueError(
            f"num_layers cannot exceed the {len(backbone.layers)} backbone layers."
        )

    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    for layer in backbone.layers[-num_layers:]:
        layer.trainable = True

    LOGGER.info("Unfroze the final %d layers of backbone '%s'.", num_layers, backbone.name)


def print_model_summary(model: tf.keras.Model) -> None:
    """Print the model summary and log its total/trainable parameter counts."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be an instance of tf.keras.Model.")

    total_parameters = _parameter_count(model.weights)
    trainable_parameters = _parameter_count(model.trainable_weights)
    non_trainable_parameters = total_parameters - trainable_parameters
    model.summary()
    LOGGER.info(
        "Parameters | total: %d | trainable: %d | non-trainable: %d",
        total_parameters,
        trainable_parameters,
        non_trainable_parameters,
    )
    print(
        "Total parameters: "
        f"{total_parameters:,}\n"
        f"Trainable parameters: {trainable_parameters:,}\n"
        f"Non-trainable parameters: {non_trainable_parameters:,}"
    )
