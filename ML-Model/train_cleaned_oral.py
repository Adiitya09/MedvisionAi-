"""Phase 8B: Train and Evaluate a fresh EfficientNetB0 classifier on the cleaned 2-class Oral dataset.

Data:
    cleaned_data/oral/Train (753 images)
    cleaned_data/oral/Validation (146 images)
    cleaned_data/oral/Test (163 images)

Outputs:
    models/cleaned_oral_model.keras
    models/cleaned_oral_class_names.json
    checkpoints/cleaned_oral_efficientnetb0_best.keras
    outputs/cleaned_oral/training_history.csv
    outputs/cleaned_oral/training_curves.png
    outputs/cleaned_oral/evaluation/metrics.json
    outputs/cleaned_oral/evaluation/classification_report.txt
    outputs/cleaned_oral/evaluation/classification_report.json
    outputs/cleaned_oral/evaluation/confusion_matrix.png
    outputs/cleaned_oral/evaluation/predictions.csv
    outputs/cleaned_oral/comparison.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf

from config import CONFIG
from utils.helpers import set_random_seed, setup_logging

LOGGER = logging.getLogger("medical_ai.train_cleaned_oral")


def build_cleaned_oral_model(num_classes: int = 2) -> tf.keras.Model:
    """Build a frozen EfficientNetB0 model with data augmentation and preprocessing."""
    input_shape = (CONFIG.image_height, CONFIG.image_width, 3)
    
    # 1. Base model with pretrained ImageNet weights
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    
    backbone = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output,
        name="backbone",
    )
    backbone.trainable = False
    
    # 2. Data augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=CONFIG.random_seed),
        tf.keras.layers.RandomRotation(0.20, seed=CONFIG.random_seed),
        tf.keras.layers.RandomZoom(0.20, seed=CONFIG.random_seed),
        tf.keras.layers.RandomContrast(0.10, seed=CONFIG.random_seed),
    ], name="data_augmentation")

    # 3. Model assembly
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Lambda(
        tf.keras.applications.efficientnet.preprocess_input,
        name="preprocessing",
    )(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cleaned_oral_efficientnetb0")
    
    # 4. Compile with standard baseline settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.initial_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


def main() -> None:
    setup_logging()
    set_random_seed(CONFIG.random_seed)

    LOGGER.info("=================================================================")
    LOGGER.info("STARTING PHASE 8B: RETRAIN ORAL MODEL ON CLEANED 2-CLASS DATASET")
    LOGGER.info("=================================================================")

    cleaned_dir = PROJECT_ROOT / "cleaned_data" / "oral"
    train_dir = cleaned_dir / "Train"
    val_dir = cleaned_dir / "Validation"
    test_dir = cleaned_dir / "Test"

    output_dir = PROJECT_ROOT / "outputs" / "cleaned_oral"
    eval_dir = output_dir / "evaluation"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    models_dir = PROJECT_ROOT / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATASETS
    LOGGER.info("Loading cleaned training dataset from %s...", train_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=CONFIG.batch_size,
        image_size=CONFIG.image_size,
        shuffle=True,
        seed=CONFIG.random_seed,
    )
    class_names = list(train_ds.class_names)
    LOGGER.info("Inferred classes: %s (Count: %d)", class_names, len(class_names))

    LOGGER.info("Loading cleaned validation dataset from %s...", val_dir)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=CONFIG.batch_size,
        image_size=CONFIG.image_size,
        shuffle=False,
    )

    LOGGER.info("Loading cleaned test dataset from %s...", test_dir)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=CONFIG.batch_size,
        image_size=CONFIG.image_size,
        shuffle=False,
    )

    # Save class names JSON
    class_names_path = models_dir / "cleaned_oral_class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    LOGGER.info("Saved class names to %s", class_names_path)

    # Pipeline caching and prefetching
    train_ds_proc = train_ds.cache().shuffle(buffer_size=1024, seed=CONFIG.random_seed).prefetch(tf.data.AUTOTUNE)
    val_ds_proc = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds_proc = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    # 2. BUILD MODEL
    model = build_cleaned_oral_model(num_classes=len(class_names))
    model.summary(print_fn=LOGGER.info)

    # 3. CONFIGURE CALLBACKS
    best_checkpoint_path = checkpoints_dir / "cleaned_oral_efficientnetb0_best.keras"
    history_csv_path = output_dir / "training_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=CONFIG.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv_path)),
    ]

    # 4. TRAIN MODEL
    LOGGER.info("Starting training for %d epochs...", CONFIG.epochs)
    history = model.fit(
        train_ds_proc,
        validation_data=val_ds_proc,
        epochs=CONFIG.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. SAVE FINAL MODEL
    final_model_path = models_dir / "cleaned_oral_model.keras"
    model.save(final_model_path)
    LOGGER.info("Saved final model to %s", final_model_path)

    # 6. PLOT TRAINING CURVES
    plt.figure(figsize=(12, 5))
    
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy", color="#2563eb", linewidth=2)
    plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy", color="#059669", linewidth=2)
    plt.title("Cleaned Oral EfficientNetB0 Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(history.history.get("val_loss", []), label="Val Loss", color="#dc2626", linewidth=2)
    plt.title("Cleaned Oral EfficientNetB0 Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    curves_path = output_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    LOGGER.info("Saved training curves to %s", curves_path)

    # 7. EVALUATE ON CLEAN TEST SPLIT (SINGLE PASS)
    LOGGER.info("Evaluating best model on CLEAN TEST SET (%s)...", test_dir)
    
    # Collect test labels and file paths in stable directory order
    y_true_list: list[int] = []
    for _, batch_labels in test_ds:
        y_true_list.extend(batch_labels.numpy().tolist())
    y_true = np.array(y_true_list)

    # Collect file paths
    test_filepaths: list[Path] = []
    for cls in class_names:
        cls_dir = test_dir / cls
        for f in sorted(cls_dir.iterdir()):
            if f.is_file():
                test_filepaths.append(f)

    if len(test_filepaths) != len(y_true):
        LOGGER.warning("Filepath count (%d) mismatch with label count (%d)", len(test_filepaths), len(y_true))

    # Predict probabilities
    y_prob = model.predict(test_ds_proc, verbose=0)
    y_pred = np.argmax(y_prob, axis=-1)
    confidences = np.max(y_prob, axis=-1)

    # Compute classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # ROC-AUC calculation
    try:
        # Binary ROC-AUC: prob of class 1 (NON CANCER) or class 0 (CANCER)
        # Class 0 = CANCER, Class 1 = NON CANCER
        auc = float(roc_auc_score(y_true, y_prob[:, 1]))
    except Exception as e:
        LOGGER.warning("ROC-AUC computation failed: %s", e)
        auc = 0.0

    cm = sk_confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    report_dict = sk_classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = sk_classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    LOGGER.info("=================================================================")
    LOGGER.info("CLEAN TEST EVALUATION RESULTS:")
    LOGGER.info("Accuracy:     %.4f (%.2f%%)", acc, acc * 100)
    LOGGER.info("Macro F1:     %.4f", macro_f1)
    LOGGER.info("Weighted F1:  %.4f", weighted_f1)
    LOGGER.info("Macro Prec:   %.4f", macro_prec)
    LOGGER.info("Macro Recall: %.4f", macro_rec)
    LOGGER.info("ROC-AUC:      %.4f", auc)
    LOGGER.info("=================================================================")
    LOGGER.info("\n%s", report_text)

    # Save metrics JSON
    metrics_data = {
        "dataset": "oral_cleaned_2class",
        "model": "efficientnetb0",
        "num_classes": len(class_names),
        "class_names": class_names,
        "test_samples": len(y_true),
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": auc,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class_metrics": {
            cls: report_dict[cls] for cls in class_names
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    # Save classification reports
    (eval_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (eval_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    # Save predictions CSV
    pred_rows = []
    for i in range(len(y_true)):
        t_cls = class_names[y_true[i]]
        p_cls = class_names[y_pred[i]]
        f_path = str(test_filepaths[i].resolve()) if i < len(test_filepaths) else ""
        f_name = test_filepaths[i].name if i < len(test_filepaths) else f"test_{i}"
        pred_rows.append({
            "filename": f_name,
            "filepath": f_path,
            "true_class": t_cls,
            "predicted_class": p_cls,
            "is_correct": int(y_true[i] == y_pred[i]),
            "confidence": float(confidences[i]),
            "prob_CANCER": float(y_prob[i, 0]),
            "prob_NON_CANCER": float(y_prob[i, 1]),
        })
    with (eval_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["filename", "filepath", "true_class", "predicted_class", "is_correct", "confidence", "prob_CANCER", "prob_NON_CANCER"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pred_rows)

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(f"Cleaned Oral Test Confusion Matrix\n(Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1:.4f})", fontsize=12, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10, fontweight="bold")
    plt.yticks(tick_marks, class_names, fontsize=10, fontweight="bold")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            plt.text(
                j, i, val_str,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = eval_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    LOGGER.info("Saved confusion matrix heatmap to %s", cm_plot_path)

    # 8. GENERATE COMPARISON REPORT (OLD 4-CLASS VS NEW CLEAN 2-CLASS)
    old_acc = 0.4882
    old_macro_f1 = 0.4262
    old_weighted_f1 = 0.4538

    comparison_rows = [
        {
            "Metric": "Test Accuracy",
            "Old 4-Class Model": f"{old_acc*100:.2f}%",
            "New Clean 2-Class Model": f"{acc*100:.2f}%",
            "Absolute Change": f"{(acc - old_acc)*100:+.2f}%",
            "Relative Improvement": f"{((acc - old_acc) / old_acc)*100:+.2f}%",
        },
        {
            "Metric": "Macro F1-Score",
            "Old 4-Class Model": f"{old_macro_f1:.4f}",
            "New Clean 2-Class Model": f"{macro_f1:.4f}",
            "Absolute Change": f"{macro_f1 - old_macro_f1:+.4f}",
            "Relative Improvement": f"{((macro_f1 - old_macro_f1) / old_macro_f1)*100:+.2f}%",
        },
        {
            "Metric": "Weighted F1-Score",
            "Old 4-Class Model": f"{old_weighted_f1:.4f}",
            "New Clean 2-Class Model": f"{weighted_f1:.4f}",
            "Absolute Change": f"{weighted_f1 - old_weighted_f1:+.4f}",
            "Relative Improvement": f"{((weighted_f1 - old_weighted_f1) / old_weighted_f1)*100:+.2f}%",
        },
        {
            "Metric": "ROC-AUC",
            "Old 4-Class Model": "N/A (Multi-class with duplicates)",
            "New Clean 2-Class Model": f"{auc:.4f}",
            "Absolute Change": "N/A",
            "Relative Improvement": "N/A",
        },
        {
            "Metric": "Number of Classes",
            "Old 4-Class Model": "4 (CANCER, CANCER 1, NON CANCER, NON CANCER 2)",
            "New Clean 2-Class Model": "2 (CANCER, NON CANCER)",
            "Absolute Change": "-2 classes (Consolidated)",
            "Relative Improvement": "Eliminated artificial subclass confusion",
        },
        {
            "Metric": "Total Images in Dataset",
            "Old 4-Class Model": "1,651",
            "New Clean 2-Class Model": "1,062",
            "Absolute Change": "-589 duplicate/conflicting images",
            "Relative Improvement": "100% Unique samples",
        },
        {
            "Metric": "Train-Test Leakage Groups",
            "Old 4-Class Model": "108 groups",
            "New Clean 2-Class Model": "0 groups (0.0%)",
            "Absolute Change": "-108 groups",
            "Relative Improvement": "100% Leakage Elimination",
        },
        {
            "Metric": "Train-Val Leakage Groups",
            "Old 4-Class Model": "93 groups",
            "New Clean 2-Class Model": "0 groups (0.0%)",
            "Absolute Change": "-93 groups",
            "Relative Improvement": "100% Leakage Elimination",
        },
        {
            "Metric": "Cross-Class Conflicts",
            "Old 4-Class Model": "1 group",
            "New Clean 2-Class Model": "0 groups (0.0%)",
            "Absolute Change": "-1 group",
            "Relative Improvement": "100% Label Consistency",
        },
        {
            "Metric": "Class Imbalance Ratio",
            "Old 4-Class Model": "1.94 : 1",
            "New Clean 2-Class Model": "1.25 : 1",
            "Absolute Change": "-0.69 (Better balanced)",
            "Relative Improvement": "Nearly equal class distribution",
        },
    ]

    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["Metric", "Old 4-Class Model", "New Clean 2-Class Model", "Absolute Change", "Relative Improvement"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    LOGGER.info("Saved comparison CSV to %s", output_dir / "comparison.csv")

    LOGGER.info("PHASE 8B RETRAINING & EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
