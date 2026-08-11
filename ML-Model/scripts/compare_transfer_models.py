"""Train, evaluate, rank, and select transfer-learning backbones.

Every candidate uses the same dataset splits and ProjectConfig hyperparameters.
The winner is chosen by macro F1, then multiclass AUC, then accuracy.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CONFIG, ProjectConfig


LOGGER = logging.getLogger(__name__)
BACKBONES: dict[str, tuple[Callable, Callable]] = {
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
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
    "inceptionv3": (
        tf.keras.applications.InceptionV3,
        tf.keras.applications.inception_v3.preprocess_input,
    ),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def make_dataset(
    directory: Path, class_names: list[str] | None, training: bool, config: ProjectConfig
) -> tf.data.Dataset:
    """Build one consistent efficient tf.data pipeline for every candidate."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=training,
        seed=config.random_seed,
    )
    if config.cache_datasets:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(
            config.shuffle_buffer_size, seed=config.random_seed, reshuffle_each_iteration=True
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def augmentation(config: ProjectConfig) -> tf.keras.Sequential:
    """Return the identical training-only augmentation stack for every model."""
    crop_height = int(config.image_height * config.random_crop_scale)
    crop_width = int(config.image_width * config.random_crop_scale)
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomCrop(crop_height, crop_width),
            tf.keras.layers.Resizing(*config.image_size),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config.rotation_factor),
            tf.keras.layers.RandomZoom(config.zoom_factor),
            tf.keras.layers.RandomBrightness(config.brightness_factor),
            tf.keras.layers.RandomContrast(config.contrast_factor),
        ],
        name="augmentation",
    )


def build_model(name: str, class_count: int, config: ProjectConfig) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Create a frozen ImageNet feature extractor and uniform classifier head."""
    factory, preprocessing = BACKBONES[name]
    base_model = factory(include_top=False, weights="imagenet", input_shape=config.input_shape)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=config.input_shape, name="image")
    x = augmentation(config)(inputs)
    x = tf.keras.layers.Lambda(preprocessing, name="preprocessing")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", dtype="float32", name="predictions"
    )(x)
    model = tf.keras.Model(inputs, outputs, name=f"{name}_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.initial_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model, base_model


def make_callbacks(model_path: Path, config: ProjectConfig) -> list[tf.keras.callbacks.Callback]:
    """Create the same model-selection callbacks for all candidates."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience, min_lr=config.min_learning_rate,
        ),
    ]


def evaluate_model(model: tf.keras.Model, test_data: tf.data.Dataset) -> dict[str, float]:
    """Calculate fair, macro-averaged test metrics and inference latency."""
    true_labels: list[int] = []
    probabilities: list[np.ndarray] = []
    elapsed_seconds = 0.0
    image_count = 0
    for images, labels in test_data:
        start = time.perf_counter()
        batch_probabilities = model(images, training=False).numpy()
        elapsed_seconds += time.perf_counter() - start
        image_count += len(images)
        probabilities.append(batch_probabilities)
        true_labels.extend(labels.numpy().astype(int).tolist())
    y_true = np.asarray(true_labels)
    y_score = np.concatenate(probabilities, axis=0)
    y_pred = np.argmax(y_score, axis=1)
    try:
        auc = roc_auc_score(
            tf.keras.utils.to_categorical(y_true, y_score.shape[1]),
            y_score,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = float("nan")
        LOGGER.warning("AUC unavailable; the test split lacks a required class.")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro_ovr": float(auc),
        "inference_time_ms": (elapsed_seconds / image_count) * 1_000,
    }


def memory_usage_mb() -> float | None:
    """Return process RSS if psutil is available; otherwise return no value."""
    try:
        import os
        import psutil  # type: ignore[import-not-found]

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        return None


def fine_tune(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    model_path: Path,
    config: ProjectConfig,
) -> None:
    """Apply the same final 20%-layer fine-tuning policy to all candidates."""
    if config.fine_tune_epochs <= 0:
        return
    base_model.trainable = True
    unfreeze_from = int(len(base_model.layers) * 0.80)
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.fine_tune_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(
        train_data, validation_data=validation_data, epochs=config.fine_tune_epochs,
        callbacks=make_callbacks(model_path, config), verbose=config.verbose,
    )


def save_ranking(records: list[dict[str, object]], output_dir: Path) -> None:
    """Save ranking in CSV, JSON, and readable Markdown table formats."""
    fieldnames = list(records[0])
    with (output_dir / "model_ranking.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "model_ranking.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    header = "| " + " | ".join(fieldnames) + " |\n"
    divider = "|" + "|".join(["---"] * len(fieldnames)) + "|\n"
    body = "".join("| " + " | ".join(str(row[key]) for key in fieldnames) + " |\n" for row in records)
    (output_dir / "model_ranking.md").write_text(header + divider + body, encoding="utf-8")


def main() -> None:
    """Train all backbones, rank test performance, and export the best model."""
    configure_logging()
    CONFIG.validate_dataset_structure()
    CONFIG.create_output_directories()
    tf.keras.utils.set_random_seed(CONFIG.random_seed)
    if CONFIG.use_mixed_precision and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    probe_data = make_dataset(CONFIG.train_dir, None, True, CONFIG)
    class_names = list(probe_data.class_names)
    CONFIG.class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    train_data = make_dataset(CONFIG.train_dir, class_names, True, CONFIG)
    validation_data = make_dataset(CONFIG.validation_dir, class_names, False, CONFIG)
    test_data = make_dataset(CONFIG.test_dir, class_names, False, CONFIG)
    output_dir = CONFIG.outputs_dir / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for name in BACKBONES:
        LOGGER.info("Training %s", name)
        tf.keras.backend.clear_session()
        model_path = CONFIG.models_dir / f"{name}_comparison.keras"
        model, base_model = build_model(name, len(class_names), CONFIG)
        start = time.perf_counter()
        model.fit(
            train_data, validation_data=validation_data, epochs=CONFIG.epochs,
            callbacks=make_callbacks(model_path, CONFIG), verbose=CONFIG.verbose,
        )
        fine_tune(model, base_model, train_data, validation_data, model_path, CONFIG)
        training_time = time.perf_counter() - start
        best_model = tf.keras.models.load_model(model_path, compile=False)
        metrics = evaluate_model(best_model, test_data)
        records.append({
            "model": name,
            **metrics,
            "training_time_minutes": training_time / 60,
            "model_size_mb": model_path.stat().st_size / 1024**2,
            "memory_usage_mb": memory_usage_mb(),
            "model_path": str(model_path.resolve()),
        })
        del model, base_model, best_model

    records.sort(key=lambda row: (-float(row["f1_macro"]), -np.nan_to_num(float(row["auc_macro_ovr"])), -float(row["accuracy"])))
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    save_ranking(records, output_dir)
    winner = records[0]
    winner_path = Path(str(winner["model_path"]))
    best_path = CONFIG.models_dir / "best_model.keras"
    shutil.copy2(winner_path, best_path)
    tf.keras.models.load_model(best_path, compile=False).export(str(CONFIG.saved_models_dir / "best_model"))
    LOGGER.info("Winner: %s (macro F1 %.4f). Saved to %s", winner["model"], winner["f1_macro"], best_path)


if __name__ == "__main__":
    main()
