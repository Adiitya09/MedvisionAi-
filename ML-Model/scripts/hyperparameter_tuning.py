"""Hyperparameter search and automatic best-model retraining.

Run this module after placing the dataset in the layout described in README:
    python hyperparameter_tuning.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import keras_tuner as kt
import numpy as np
import tensorflow as tf

from config import CONFIG, ProjectConfig


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure concise, timestamped console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """Make the search reproducible where TensorFlow permits it."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def enable_mixed_precision(config: ProjectConfig) -> None:
    """Enable mixed precision only when a compatible GPU is available."""
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    if config.use_mixed_precision and has_gpu:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        LOGGER.info("Mixed precision enabled.")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")


def build_augmentation(config: ProjectConfig) -> tf.keras.Sequential:
    """Return image augmentations applied only while model.fit is training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomCrop(
                int(config.image_height * config.random_crop_scale),
                int(config.image_width * config.random_crop_scale),
            ),
            tf.keras.layers.Resizing(*config.image_size),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config.rotation_factor),
            tf.keras.layers.RandomZoom(config.zoom_factor),
            tf.keras.layers.RandomBrightness(config.brightness_factor),
            tf.keras.layers.RandomContrast(config.contrast_factor),
        ],
        name="augmentation",
    )


def get_backbone(
    config: ProjectConfig,
) -> tuple[Callable, Callable]:
    """Return the selected Keras application constructor and preprocessor."""
    backbones = {
        "mobilenetv2": (
            tf.keras.applications.MobileNetV2,
            tf.keras.applications.mobilenet_v2.preprocess_input,
        ),
        "efficientnetb0": (
            tf.keras.applications.EfficientNetB0,
            tf.keras.applications.efficientnet.preprocess_input,
        ),
        "resnet50": (
            tf.keras.applications.ResNet50,
            tf.keras.applications.resnet50.preprocess_input,
        ),
    }
    try:
        return backbones[config.backbone]
    except KeyError as error:
        raise ValueError(f"Unsupported backbone: {config.backbone}") from error


def build_model(
    hyperparameters: kt.HyperParameters,
    config: ProjectConfig,
    num_classes: int,
) -> tf.keras.Model:
    """Build a transfer-learning classifier from a Keras Tuner trial."""
    backbone_factory, preprocess_input = get_backbone(config)
    base_model = backbone_factory(
        include_top=False,
        weights="imagenet",
        input_shape=config.input_shape,
    )

    frozen_layers = hyperparameters.Int(
        "frozen_layers",
        min_value=0,
        max_value=len(base_model.layers),
        step=max(1, len(base_model.layers) // 10),
    )
    for layer_index, layer in enumerate(base_model.layers):
        layer.trainable = layer_index >= frozen_layers

    activation = hyperparameters.Choice(
        "activation", values=["relu", "gelu", "swish"]
    )
    dropout = hyperparameters.Float("dropout", 0.20, 0.60, step=0.10)
    dense_units = hyperparameters.Choice("dense_units", values=[128, 256, 512, 1024])

    inputs = tf.keras.Input(shape=config.input_shape, name="image")
    x = build_augmentation(config)(inputs)
    x = tf.keras.layers.Lambda(preprocess_input, name="preprocessing")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    x = tf.keras.layers.Dense(dense_units, activation=activation, name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{config.backbone}_tuned")

    learning_rate = hyperparameters.Float(
        "learning_rate", min_value=1e-5, max_value=1e-3, sampling="log"
    )
    optimizer_name = hyperparameters.Choice(
        "optimizer", values=["adam", "adamw", "rmsprop"]
    )
    optimizer_classes = {
        "adam": tf.keras.optimizers.Adam,
        "adamw": tf.keras.optimizers.AdamW,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    model.compile(
        optimizer=optimizer_classes[optimizer_name](learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )
    return model


def create_dataset(
    directory: Path,
    class_names: list[str] | None,
    batch_size: int,
    training: bool,
    config: ProjectConfig,
) -> tf.data.Dataset:
    """Load a split as an efficient, deterministic tf.data pipeline."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=config.image_size,
        batch_size=batch_size,
        shuffle=training,
        seed=config.random_seed,
    )
    if config.cache_datasets:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(
            config.shuffle_buffer_size,
            seed=config.random_seed,
            reshuffle_each_iteration=True,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


class DatasetAwareRandomSearch(kt.RandomSearch):
    """Random search that rebuilds data pipelines for trial batch sizes."""

    def __init__(self, config: ProjectConfig, class_names: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.class_names = class_names

    def run_trial(self, trial: kt.engine.trial.Trial, *args: object, **kwargs: object) -> object:
        """Train one trial using its own batch size and epoch count."""
        batch_size = trial.hyperparameters.Choice("batch_size", values=[16, 32, 64])
        epochs = trial.hyperparameters.Int("epochs", min_value=15, max_value=50, step=5)
        train_dataset = create_dataset(
            self.config.train_dir, self.class_names, batch_size, True, self.config
        )
        validation_dataset = create_dataset(
            self.config.validation_dir, self.class_names, batch_size, False, self.config
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=self.config.early_stopping_patience,
                mode="max", restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=self.config.reduce_lr_patience,
                factor=self.config.reduce_lr_factor, min_lr=self.config.min_learning_rate,
            ),
        ]
        kwargs.update(
            {
                "x": train_dataset,
                "validation_data": validation_dataset,
                "epochs": epochs,
                "callbacks": callbacks,
                "verbose": self.config.verbose,
            }
        )
        return super().run_trial(trial, *args, **kwargs)


def save_trial_comparison(tuner: kt.Tuner, output_path: Path) -> None:
    """Write final metric values and hyperparameters for every completed trial."""
    rows: list[dict[str, object]] = []
    for trial in tuner.oracle.trials.values():
        row: dict[str, object] = {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "score_val_accuracy": trial.score,
            **trial.hyperparameters.values,
        }
        for metric_name in ("val_loss", "val_top_5_accuracy"):
            history = trial.metrics.get_history(metric_name)
            if history:
                row[metric_name] = history[-1].value[0]
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d trial comparisons to %s", len(rows), output_path)


def save_class_names(class_names: list[str], path: Path) -> None:
    """Persist the label-to-index mapping used by inference."""
    path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def retrain_best_model(
    tuner: kt.Tuner, config: ProjectConfig, class_names: list[str]
) -> tf.keras.Model:
    """Train the chosen architecture on the configured epoch budget and save it."""
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    batch_size = best_hyperparameters.get("batch_size")
    epochs = best_hyperparameters.get("epochs")
    model = tuner.hypermodel.build(best_hyperparameters)
    train_dataset = create_dataset(config.train_dir, class_names, batch_size, True, config)
    validation_dataset = create_dataset(
        config.validation_dir, class_names, batch_size, False, config
    )
    best_tuned_model_path = config.models_dir / f"{config.backbone}_best_tuned.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_tuned_model_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.logs_dir / f"tuned_{datetime.now():%Y%m%d-%H%M%S}")
        ),
    ]
    LOGGER.info("Retraining the best model for up to %d epochs.", epochs)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    best_model = tf.keras.models.load_model(best_tuned_model_path)
    best_model.export(str(config.saved_models_dir / f"{config.backbone}_best_tuned"))
    LOGGER.info("Saved best Keras model to %s", best_tuned_model_path)
    return best_model


def main() -> None:
    """Execute tuning, report the winner, and retrain it automatically."""
    configure_logging()
    CONFIG.validate_dataset_structure()
    CONFIG.create_output_directories()
    set_global_seed(CONFIG.random_seed)
    enable_mixed_precision(CONFIG)

    probe_dataset = create_dataset(
        CONFIG.train_dir, None, CONFIG.batch_size, True, CONFIG
    )
    class_names = list(probe_dataset.class_names)
    if len(class_names) < 2:
        raise ValueError("At least two disease classes are required for classification.")
    save_class_names(class_names, CONFIG.class_names_path)
    LOGGER.info("Found %d classes: %s", len(class_names), ", ".join(class_names))

    tuner = DatasetAwareRandomSearch(
        config=CONFIG,
        class_names=class_names,
        hypermodel=lambda hp: build_model(hp, CONFIG, len(class_names)),
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=20,
        executions_per_trial=1,
        directory=str(CONFIG.outputs_dir / "keras_tuner"),
        project_name=f"{CONFIG.backbone}_search",
        overwrite=False,
        seed=CONFIG.random_seed,
    )
    tuner.search()
    tuner.results_summary(num_trials=20)
    save_trial_comparison(tuner, CONFIG.outputs_dir / "tuning_trials.csv")

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    LOGGER.info("Best hyperparameters: %s", best_hyperparameters.values)
    (CONFIG.outputs_dir / "best_hyperparameters.json").write_text(
        json.dumps(best_hyperparameters.values, indent=2), encoding="utf-8"
    )
    retrain_best_model(tuner, CONFIG, class_names)


if __name__ == "__main__":
    main()
