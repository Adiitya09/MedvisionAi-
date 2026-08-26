"""Phase 7: Deep Error Analysis & Dataset Improvement Suite for MedvisionAI.

Performs comprehensive diagnostic analysis across Skin, Eye, and Oral datasets:
1. Class distribution & imbalance ratio analysis
2. Per-class Precision, Recall, F1, Support decomposition
3. Confusion matrix & top error-pair ranking
4. Misclassification breakdown (high-confidence vs low-confidence vs top confusion pairs)
5. Prediction confidence distribution for correct vs incorrect cases
6. Image quality scan (corrupted, non-RGB, extreme aspect ratios, low variance, small sizes)
7. Duplicate & data leakage detection (cryptographic MD5 + perceptual dHash)
8. Preprocessing pipeline verification
9. Grad-CAM error attention analysis
10. Root-cause synthesis & prioritized next-training recommendations

Outputs are saved in:
- outputs/error_analysis/<dataset>/
- outputs/error_analysis/overall_error_analysis.json
- outputs/error_analysis/overall_error_analysis.csv
- outputs/error_analysis/error_analysis_report.md
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

tf.get_logger().setLevel(logging.ERROR)

from config import CONFIG, ProjectConfig
from utils.helpers import load_model, setup_logging
from explainability.explainability import generate_gradcam

LOGGER = logging.getLogger("medical_ai.error_analysis")
SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"})


# =====================================================================
# 1. CLASS DISTRIBUTION ANALYSIS
# =====================================================================

def analyze_class_distribution(
    dataset_name: str, config: ProjectConfig, output_dir: Path
) -> dict[str, Any]:
    """Calculate class counts, percentages, and imbalance ratios across splits."""
    splits = ["train", "validation", "test"]
    class_names = json.loads(config.class_names_path_for(dataset_name).read_text(encoding="utf-8"))
    
    counts: dict[str, dict[str, int]] = {split: {} for split in splits}
    for split in splits:
        split_dir = config.split_dir(dataset_name, split)
        for cls in class_names:
            cls_dir = split_dir / cls
            if cls_dir.is_dir():
                n_images = sum(
                    1 for p in cls_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                )
                counts[split][cls] = n_images
            else:
                counts[split][cls] = 0

    rows: list[dict[str, Any]] = []
    total_train = sum(counts["train"].values())
    total_val = sum(counts["validation"].values())
    total_test = sum(counts["test"].values())
    total_all = total_train + total_val + total_test

    for cls in class_names:
        c_tr = counts["train"].get(cls, 0)
        c_val = counts["validation"].get(cls, 0)
        c_te = counts["test"].get(cls, 0)
        c_tot = c_tr + c_val + c_te
        rows.append({
            "class_name": cls,
            "train_count": c_tr,
            "train_pct": (c_tr / total_train * 100) if total_train else 0.0,
            "val_count": c_val,
            "val_pct": (c_val / total_val * 100) if total_val else 0.0,
            "test_count": c_te,
            "test_pct": (c_te / total_test * 100) if total_test else 0.0,
            "total_count": c_tot,
            "total_pct": (c_tot / total_all * 100) if total_all else 0.0,
        })

    train_counts = [r["train_count"] for r in rows if r["train_count"] > 0]
    imbalance_ratio = (max(train_counts) / min(train_counts)) if train_counts else 1.0

    # Save CSV
    csv_path = output_dir / "class_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    plt.bar(x - width, [r["train_count"] for r in rows], width, label="Train", color="#2563eb")
    plt.bar(x, [r["val_count"] for r in rows], width, label="Validation", color="#7c3aed")
    plt.bar(x + width, [r["test_count"] for r in rows], width, label="Test", color="#059669")
    plt.xticks(x, class_names, rotation=30, ha="right", fontsize=10)
    plt.ylabel("Image Count", fontsize=11, fontweight="bold")
    plt.title(f"{dataset_name.upper()} Dataset Class Distribution (Imbalance Ratio: {imbalance_ratio:.2f}:1)", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    LOGGER.info("[%s] Saved class distribution to %s and %s", dataset_name.upper(), csv_path, plot_path)
    return {
        "class_names": class_names,
        "counts": counts,
        "rows": rows,
        "total_train": total_train,
        "total_val": total_val,
        "total_test": total_test,
        "total_all": total_all,
        "imbalance_ratio": float(imbalance_ratio),
    }


# =====================================================================
# 2. PER-CLASS PERFORMANCE & CONFUSION ANALYSIS
# =====================================================================

def evaluate_test_predictions(
    dataset_name: str, model_path: Path, config: ProjectConfig, output_dir: Path
) -> dict[str, Any]:
    """Run full test inference to compute metrics, confusion matrices, and misclassifications."""
    class_names = json.loads(config.class_names_path_for(dataset_name).read_text(encoding="utf-8"))
    test_dir = config.split_dir(dataset_name, "test")
    model = load_model(model_path)

    # Collect test files deterministically
    image_paths: list[Path] = []
    y_true: list[int] = []
    for idx, cls in enumerate(class_names):
        cls_dir = test_dir / cls
        files = sorted(
            p for p in cls_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        image_paths.extend(files)
        y_true.extend([idx] * len(files))

    # Load dataset batch-wise for fast vectorized predictions
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=config.image_size,
        batch_size=32,
        shuffle=False,
    )
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)

    # Per-class metrics
    report = classification_report(
        y_true_np, y_pred_np, target_names=class_names, output_dict=True, zero_division=0
    )
    metric_rows: list[dict[str, Any]] = []
    for cls in class_names:
        cls_rep = report.get(cls, {})
        metric_rows.append({
            "class_name": cls,
            "precision": float(cls_rep.get("precision", 0.0)),
            "recall": float(cls_rep.get("recall", 0.0)),
            "f1_score": float(cls_rep.get("f1-score", 0.0)),
            "support": int(cls_rep.get("support", 0)),
        })

    metrics_csv = output_dir / "classification_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(metric_rows)

    # Confusion matrix
    cm_raw = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm_raw.astype(np.float64) / cm_raw.sum(axis=1, keepdims=True))

    np.savetxt(output_dir / "confusion_matrix_raw.csv", cm_raw, delimiter=",", fmt="%d")
    np.savetxt(output_dir / "confusion_matrix_normalized.csv", cm_norm, delimiter=",", fmt="%.4f")

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{dataset_name.upper()} Normalized Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right", fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)
    
    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j, i, f"{cm_raw[i, j]}\n({cm_norm[i, j]:.1%})",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # Identify top confusion pairs
    confusion_pairs: list[dict[str, Any]] = []
    total_errors = int(np.sum(cm_raw) - np.trace(cm_raw))
    for i in range(len(class_names)):
        true_cls = class_names[i]
        true_total = int(cm_raw[i].sum())
        for j in range(len(class_names)):
            if i != j and cm_raw[i, j] > 0:
                count = int(cm_raw[i, j])
                confusion_pairs.append({
                    "true_class": true_cls,
                    "predicted_class": class_names[j],
                    "error_count": count,
                    "pct_of_true_class": (count / true_total * 100) if true_total else 0.0,
                    "pct_of_total_errors": (count / total_errors * 100) if total_errors else 0.0,
                })
    confusion_pairs.sort(key=lambda x: x["error_count"], reverse=True)

    with (output_dir / "confusion_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_class", "predicted_class", "error_count", "pct_of_true_class", "pct_of_total_errors"])
        writer.writeheader()
        writer.writerows(confusion_pairs)

    # Full misclassifications list
    misclassified_items: list[dict[str, Any]] = []
    for idx in range(len(y_true_np)):
        t_idx = int(y_true_np[idx])
        p_idx = int(y_pred_np[idx])
        conf = float(confidences[idx])
        img_p = image_paths[idx]
        if t_idx != p_idx:
            misclassified_items.append({
                "filename": img_p.name,
                "filepath": str(img_p.resolve()),
                "true_class": class_names[t_idx],
                "predicted_class": class_names[p_idx],
                "confidence": conf,
                "dataset": dataset_name,
                "model": model_path.name,
                "is_high_confidence": bool(conf >= 0.75),
                "is_low_confidence": bool(conf < 0.50),
            })

    # Save misclassifications
    (output_dir / "misclassifications.json").write_text(json.dumps(misclassified_items, indent=2), encoding="utf-8")
    with (output_dir / "misclassifications.csv").open("w", newline="", encoding="utf-8") as f:
        if misclassified_items:
            writer = csv.DictWriter(f, fieldnames=list(misclassified_items[0].keys()))
            writer.writeheader()
            writer.writerows(misclassified_items)

    high_conf_errors = [m for m in misclassified_items if m["is_high_confidence"]]
    low_conf_errors = [m for m in misclassified_items if m["is_low_confidence"]]

    # Confidence distribution
    correct_mask = (y_true_np == y_pred_np)
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    conf_stats = {
        "dataset": dataset_name,
        "total_test_samples": len(y_true_np),
        "total_correct": int(np.sum(correct_mask)),
        "total_incorrect": len(misclassified_items),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "correct_confidence": {
            "mean": float(np.mean(correct_conf)) if len(correct_conf) else 0.0,
            "std": float(np.std(correct_conf)) if len(correct_conf) else 0.0,
            "median": float(np.median(correct_conf)) if len(correct_conf) else 0.0,
            "min": float(np.min(correct_conf)) if len(correct_conf) else 0.0,
            "max": float(np.max(correct_conf)) if len(correct_conf) else 0.0,
            "q25": float(np.percentile(correct_conf, 25)) if len(correct_conf) else 0.0,
            "q75": float(np.percentile(correct_conf, 75)) if len(correct_conf) else 0.0,
        },
        "incorrect_confidence": {
            "mean": float(np.mean(incorrect_conf)) if len(incorrect_conf) else 0.0,
            "std": float(np.std(incorrect_conf)) if len(incorrect_conf) else 0.0,
            "median": float(np.median(incorrect_conf)) if len(incorrect_conf) else 0.0,
            "min": float(np.min(incorrect_conf)) if len(incorrect_conf) else 0.0,
            "max": float(np.max(incorrect_conf)) if len(incorrect_conf) else 0.0,
            "q25": float(np.percentile(incorrect_conf, 25)) if len(incorrect_conf) else 0.0,
            "q75": float(np.percentile(incorrect_conf, 75)) if len(incorrect_conf) else 0.0,
        },
        "high_confidence_error_count": len(high_conf_errors),
        "high_confidence_error_pct": (len(high_conf_errors) / len(misclassified_items) * 100) if misclassified_items else 0.0,
        "low_confidence_error_count": len(low_conf_errors),
        "low_confidence_error_pct": (len(low_conf_errors) / len(misclassified_items) * 100) if misclassified_items else 0.0,
    }

    (output_dir / "confidence_analysis.json").write_text(json.dumps(conf_stats, indent=2), encoding="utf-8")

    # Plot Confidence Distribution
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 25)
    plt.hist(correct_conf, bins=bins, alpha=0.6, label=f"Correct (Mean={conf_stats['correct_confidence']['mean']:.2f})", color="#059669", density=True)
    plt.hist(incorrect_conf, bins=bins, alpha=0.6, label=f"Incorrect (Mean={conf_stats['incorrect_confidence']['mean']:.2f})", color="#dc2626", density=True)
    plt.title(f"{dataset_name.upper()} Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Confidence", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    conf_plot_path = output_dir / "confidence_distribution.png"
    plt.savefig(conf_plot_path, dpi=300)
    plt.close()

    LOGGER.info("[%s] Completed evaluation & confidence analysis", dataset_name.upper())

    return {
        "metric_rows": metric_rows,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "confusion_pairs": confusion_pairs,
        "misclassifications": misclassified_items,
        "high_conf_errors": high_conf_errors,
        "low_conf_errors": low_conf_errors,
        "conf_stats": conf_stats,
        "image_paths": [str(p) for p in image_paths],
        "y_true": y_true_np.tolist(),
        "y_pred": y_pred_np.tolist(),
        "probabilities": probabilities.tolist(),
    }


# =====================================================================
# 3 & 4. FAST INTEGRATED QUALITY SCAN & DUPLICATE/LEAKAGE DETECTION
# =====================================================================

def scan_quality_and_duplicates(
    dataset_name: str, config: ProjectConfig, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Single fast pass over all split images to check quality anomalies and hash duplicates."""
    splits = ["train", "validation", "test"]
    issues: list[dict[str, Any]] = []
    
    quality_stats = {
        "total_scanned": 0,
        "corrupted_count": 0,
        "small_dimension_count": 0,
        "large_dimension_count": 0,
        "extreme_aspect_ratio_count": 0,
        "non_rgb_mode_count": 0,
        "blank_or_low_variance_count": 0,
        "color_modes": Counter(),
        "formats": Counter(),
    }

    split_items: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    md5_to_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dhash_to_items: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split in splits:
        split_dir = config.split_dir(dataset_name, split)
        all_files = [
            p for p in split_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]

        for file_path in all_files:
            quality_stats["total_scanned"] += 1
            ext = file_path.suffix.lower()
            quality_stats["formats"][ext] += 1

            try:
                content = file_path.read_bytes()
                md5_hash = hashlib.md5(content).hexdigest()

                with Image.open(io.BytesIO(content)) as img:
                    width, height = img.size
                    mode = img.mode
                    quality_stats["color_modes"][mode] += 1

                    aspect_ratio = width / height if height > 0 else 0.0

                    issue_reasons: list[str] = []
                    if mode != "RGB":
                        quality_stats["non_rgb_mode_count"] += 1
                        issue_reasons.append(f"Non-RGB mode ({mode})")

                    if width < 100 or height < 100:
                        quality_stats["small_dimension_count"] += 1
                        issue_reasons.append(f"Small resolution ({width}x{height})")
                    elif width > 3000 or height > 3000:
                        quality_stats["large_dimension_count"] += 1
                        issue_reasons.append(f"Very large resolution ({width}x{height})")

                    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                        quality_stats["extreme_aspect_ratio_count"] += 1
                        issue_reasons.append(f"Extreme aspect ratio ({aspect_ratio:.2f})")

                    # Fast difference hash (9x8 grayscale thumbnail)
                    gray_thumb = img.convert("L").resize((9, 8), Image.Resampling.BOX)
                    diff = np.asarray(gray_thumb)[:, 1:] > np.asarray(gray_thumb)[:, :-1]
                    dhash_str = "".join(["1" if b else "0" for b in diff.flatten()])

                    # Fast pixel variance on (16x16 thumbnail)
                    var_thumb = img.convert("L").resize((16, 16), Image.Resampling.BOX)
                    std_dev = float(np.std(np.asarray(var_thumb)))
                    if std_dev < 5.0:
                        quality_stats["blank_or_low_variance_count"] += 1
                        issue_reasons.append(f"Low pixel variance (std={std_dev:.2f})")

                    item_record = {
                        "dataset": dataset_name,
                        "split": split,
                        "class": file_path.parent.name,
                        "filename": file_path.name,
                        "filepath": str(file_path.resolve()),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "aspect_ratio": round(aspect_ratio, 2),
                        "pixel_std": round(std_dev, 2),
                        "md5": md5_hash,
                        "dhash": dhash_str,
                    }

                    if issue_reasons:
                        issues.append({
                            **item_record,
                            "issues": "; ".join(issue_reasons),
                        })

                    split_items[split].append(item_record)
                    md5_to_items[md5_hash].append(item_record)
                    dhash_to_items[dhash_str].append(item_record)

            except Exception as error:
                quality_stats["corrupted_count"] += 1
                issues.append({
                    "dataset": dataset_name,
                    "split": split,
                    "class": file_path.parent.name,
                    "filename": file_path.name,
                    "filepath": str(file_path.resolve()),
                    "width": 0,
                    "height": 0,
                    "mode": "CORRUPT",
                    "aspect_ratio": 0.0,
                    "pixel_std": 0.0,
                    "md5": "ERROR",
                    "dhash": "ERROR",
                    "issues": f"Corrupted file: {error}",
                })

    # Save Quality Report
    (output_dir / "image_quality_issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    with (output_dir / "image_quality_issues.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "split", "class", "filename", "filepath", "width", "height", "mode", "aspect_ratio", "pixel_std", "issues"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(issues)

    # Process Duplicate / Leakage Groups
    exact_duplicates = [v for v in md5_to_items.values() if len(v) > 1]
    perceptual_duplicates = [v for v in dhash_to_items.values() if len(v) > 1]

    train_val_leakage: list[list[dict[str, Any]]] = []
    train_test_leakage: list[list[dict[str, Any]]] = []
    val_test_leakage: list[list[dict[str, Any]]] = []
    within_split_duplicates: list[list[dict[str, Any]]] = []

    for group in exact_duplicates:
        group_splits = set(item["split"] for item in group)
        if "train" in group_splits and "test" in group_splits:
            train_test_leakage.append(group)
        elif "train" in group_splits and "validation" in group_splits:
            train_val_leakage.append(group)
        elif "validation" in group_splits and "test" in group_splits:
            val_test_leakage.append(group)
        else:
            within_split_duplicates.append(group)

    leakage_records: list[dict[str, Any]] = []
    for cat_name, groups in [
        ("TRAIN_TEST_LEAKAGE", train_test_leakage),
        ("TRAIN_VAL_LEAKAGE", train_val_leakage),
        ("VAL_TEST_LEAKAGE", val_test_leakage),
        ("WITHIN_SPLIT_DUPLICATE", within_split_duplicates),
    ]:
        for g in groups:
            files_info = " <==> ".join([f"[{item['split']}:{item['class']}] {item['filename']}" for item in g])
            leakage_records.append({
                "dataset": dataset_name,
                "leakage_category": cat_name,
                "duplicate_count": len(g),
                "md5_hash": g[0]["md5"],
                "file_pairs": files_info,
            })

    # Save Duplicate/Leakage Report
    (output_dir / "duplicate_leakage_report.json").write_text(
        json.dumps(
            {
                "exact_duplicate_groups_count": len(exact_duplicates),
                "perceptual_duplicate_groups_count": len(perceptual_duplicates),
                "train_test_leakage_groups": len(train_test_leakage),
                "train_val_leakage_groups": len(train_val_leakage),
                "val_test_leakage_groups": len(val_test_leakage),
                "within_split_duplicate_groups": len(within_split_duplicates),
                "leakage_details": leakage_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "duplicate_leakage_report.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "leakage_category", "duplicate_count", "md5_hash", "file_pairs"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(leakage_records)

    quality_info = {
        "stats": {
            **quality_stats,
            "color_modes": dict(quality_stats["color_modes"]),
            "formats": dict(quality_stats["formats"]),
        },
        "issues_count": len(issues),
        "issues": issues,
    }

    leakage_info = {
        "exact_duplicate_groups_count": len(exact_duplicates),
        "perceptual_duplicate_groups_count": len(perceptual_duplicates),
        "train_test_leakage_count": len(train_test_leakage),
        "train_val_leakage_count": len(train_val_leakage),
        "val_test_leakage_count": len(val_test_leakage),
        "within_split_duplicate_count": len(within_split_duplicates),
        "leakage_records": leakage_records,
    }

    LOGGER.info(
        "[%s] Fast Scan Complete | Scanned: %d | Issues: %d | Corrupt: %d | Leakage Groups: %d",
        dataset_name.upper(),
        quality_stats["total_scanned"],
        len(issues),
        quality_stats["corrupted_count"],
        len(train_test_leakage) + len(train_val_leakage) + len(val_test_leakage),
    )

    return quality_info, leakage_info


# =====================================================================
# 5. GRAD-CAM ERROR CASE INSPECTION
# =====================================================================

def analyze_gradcam_errors(
    dataset_name: str,
    model_path: Path,
    misclassifications: list[dict[str, Any]],
    output_dir: Path,
    max_cases: int = 4,
) -> list[dict[str, Any]]:
    """Run Grad-CAM on selected high-confidence wrong predictions and confusing pairs."""
    gradcam_dir = output_dir / "gradcam_error_cases"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    selected_cases: list[dict[str, Any]] = []
    high_conf = [m for m in misclassifications if m["is_high_confidence"]]
    if high_conf:
        selected_cases.extend(high_conf[:max_cases // 2])
    remaining = [m for m in misclassifications if m not in selected_cases]
    if remaining:
        selected_cases.extend(remaining[: max_cases - len(selected_cases)])

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(selected_cases, 1):
        img_path = Path(case["filepath"])
        prefix = f"error_case_{idx:02d}_{case['true_class']}_pred_{case['predicted_class']}"
        try:
            res = generate_gradcam(
                dataset_name=dataset_name,
                image_path=img_path,
                model_path=model_path,
                output_directory=gradcam_dir,
                filename_prefix=prefix,
            )
            results.append({
                "case_index": idx,
                "image_filename": img_path.name,
                "true_class": case["true_class"],
                "predicted_class": case["predicted_class"],
                "confidence": case["confidence"],
                "target_layer": res.get("target_layer"),
                "overlay_path": res.get("overlay_path"),
            })
        except Exception as e:
            LOGGER.warning("Grad-CAM generation failed for %s: %s", img_path, e)

    (output_dir / "gradcam_error_cases.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# =====================================================================
# 6. DOMAIN ANALYSIS ORCHESTRATOR
# =====================================================================

def run_deep_error_analysis(dataset_name: str, model_path: Path) -> dict[str, Any]:
    """Execute all diagnostic steps for one disease domain."""
    dataset_key = dataset_name.strip().lower()
    output_dir = CONFIG.outputs_dir / "error_analysis" / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STARTING DEEP ERROR ANALYSIS FOR: %s", dataset_key.upper())
    LOGGER.info("Model: %s | Output: %s", model_path, output_dir)
    LOGGER.info("=" * 80)

    # 1. Class distribution
    dist_info = analyze_class_distribution(dataset_key, CONFIG, output_dir)

    # 2. Test performance & confusion analysis
    eval_info = evaluate_test_predictions(dataset_key, model_path, CONFIG, output_dir)

    # 3 & 4. Integrated quality scan and duplicate/leakage detection
    quality_info, leakage_info = scan_quality_and_duplicates(dataset_key, CONFIG, output_dir)

    # 5. Grad-CAM error inspection
    gradcam_info = analyze_gradcam_errors(dataset_key, model_path, eval_info["misclassifications"], output_dir)

    LOGGER.info("[%s] All error analysis modules completed successfully.", dataset_key.upper())

    return {
        "dataset": dataset_key,
        "model_path": str(model_path),
        "distribution": dist_info,
        "evaluation": eval_info,
        "image_quality": quality_info,
        "leakage": leakage_info,
        "gradcam_errors": gradcam_info,
    }


# =====================================================================
# 7. GLOBAL SUMMARY & REPORT BUILDER
# =====================================================================

def generate_global_report(results_by_domain: dict[str, dict[str, Any]]) -> None:
    """Create overall JSON, CSV, and markdown synthesis reports."""
    base_output_dir = CONFIG.outputs_dir / "error_analysis"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    json_summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for domain, res in results_by_domain.items():
        dist = res["distribution"]
        ev = res["evaluation"]
        q = res["image_quality"]
        lk = res["leakage"]
        cf = ev["conf_stats"]

        # Worst 3 classes by F1
        sorted_classes = sorted(ev["metric_rows"], key=lambda x: x["f1_score"])
        worst_classes = [f"{r['class_name']} (F1={r['f1_score']:.4f}, Rec={r['recall']:.4f})" for r in sorted_classes[:3]]

        # Top 3 confusion pairs
        top_confusions = [
            f"{p['true_class']} -> {p['predicted_class']} (N={p['error_count']}, {p['pct_of_true_class']:.1f}%)"
            for p in ev["confusion_pairs"][:3]
        ]

        item = {
            "dataset": domain,
            "test_accuracy": cf["accuracy"],
            "macro_f1": cf["macro_f1"],
            "weighted_f1": cf["weighted_f1"],
            "total_images": dist["total_all"],
            "train_samples": dist["total_train"],
            "test_samples": dist["total_test"],
            "imbalance_ratio": dist["imbalance_ratio"],
            "worst_classes": worst_classes,
            "top_confusion_pairs": top_confusions,
            "total_misclassifications": cf["total_incorrect"],
            "high_confidence_errors": cf["high_confidence_error_count"],
            "high_confidence_error_pct": cf["high_confidence_error_pct"],
            "correct_mean_conf": cf["correct_confidence"]["mean"],
            "incorrect_mean_conf": cf["incorrect_confidence"]["mean"],
            "corrupted_images": q["stats"]["corrupted_count"],
            "non_rgb_images": q["stats"]["non_rgb_mode_count"],
            "small_resolution_images": q["stats"]["small_dimension_count"],
            "train_test_leakage_groups": lk["train_test_leakage_count"],
            "train_val_leakage_groups": lk["train_val_leakage_count"],
            "within_split_duplicate_groups": lk["within_split_duplicate_count"],
        }
        json_summary.append(item)
        csv_rows.append(item)

    (base_output_dir / "overall_error_analysis.json").write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    with (base_output_dir / "overall_error_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    LOGGER.info("Saved overall error analysis JSON and CSV.")


def run_all() -> None:
    """Run Phase 7 deep error analysis across all 3 domains."""
    setup_logging()
    LOGGER.info("STARTING PHASE 7: DEEP ERROR ANALYSIS & DATASET IMPROVEMENT")

    domain_specs = [
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

    results: dict[str, dict[str, Any]] = {}
    for spec in domain_specs:
        res = run_deep_error_analysis(spec["dataset"], spec["model_path"])
        results[spec["dataset"]] = res

    generate_global_report(results)
    LOGGER.info("Phase 7 Deep Error Analysis completed across all domains!")


if __name__ == "__main__":
    run_all()
