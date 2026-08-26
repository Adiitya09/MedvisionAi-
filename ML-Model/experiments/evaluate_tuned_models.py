"""Evaluate the best tuned models strictly on the unseen test split.

This script executes the FINAL TEST EVALUATION phase for:
1. Skin -> ResNet50 best tuned model (models/tuning/skin/resnet50_best_tuned.keras)
2. Eye  -> ResNet50 best tuned model (models/tuning/eye/resnet50_best_tuned.keras)
3. Oral -> EfficientNetB0 best tuned model (models/tuning/oral/efficientnetb0_best_tuned.keras)

For each dataset, it computes:
- Test Accuracy
- Macro Precision / Recall / F1
- Weighted Precision / Recall / F1
- Micro metrics
- ROC-AUC (One-vs-Rest macro average) where supported
- Per-class Precision / Recall / F1

It exports:
- outputs/evaluation/tuned/<dataset>/metrics.json
- outputs/evaluation/tuned/<dataset>/classification_report.json
- outputs/evaluation/tuned/<dataset>/classification_report.txt
- outputs/evaluation/tuned/<dataset>/confusion_matrix.png
- outputs/evaluation/tuned/<dataset>/predictions.csv

It compares results against original baseline EfficientNetB0 test results and saves:
- outputs/evaluation/tuned/test_comparison_summary.json
- outputs/evaluation/tuned/test_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from config import CONFIG
from evaluation.evaluate import evaluate_model
from utils.helpers import setup_logging

LOGGER = logging.getLogger(__name__)

# Configured best tuned models from hyperparameter tuning phase
TUNED_EVAL_SPECS = [
    {
        "dataset": "skin",
        "tuned_model_name": "ResNet50 (Tuned)",
        "baseline_model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "skin" / "resnet50_best_tuned.keras",
        "output_dir": PROJECT_ROOT / "outputs" / "evaluation" / "tuned" / "skin",
        "baseline_metrics_path": PROJECT_ROOT / "outputs" / "evaluation" / "skin" / "metrics.json",
        "tuning_info": {
            "backbone": "resnet50",
            "trial": 3,
            "learning_rate": 3e-4,
            "dropout": 0.30,
            "fine_tuning": "frozen",
            "val_macro_f1": 0.7070,
        },
    },
    {
        "dataset": "eye",
        "tuned_model_name": "ResNet50 (Tuned)",
        "baseline_model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "eye" / "resnet50_best_tuned.keras",
        "output_dir": PROJECT_ROOT / "outputs" / "evaluation" / "tuned" / "eye",
        "baseline_metrics_path": PROJECT_ROOT / "outputs" / "evaluation" / "eye" / "metrics.json",
        "tuning_info": {
            "backbone": "resnet50",
            "trial": 3,
            "learning_rate": 3e-4,
            "dropout": 0.30,
            "fine_tuning": "frozen",
            "val_macro_f1": 0.5999,
        },
    },
    {
        "dataset": "oral",
        "tuned_model_name": "EfficientNetB0 (Tuned)",
        "baseline_model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "oral" / "efficientnetb0_best_tuned.keras",
        "output_dir": PROJECT_ROOT / "outputs" / "evaluation" / "tuned" / "oral",
        "baseline_metrics_path": PROJECT_ROOT / "outputs" / "evaluation" / "oral" / "metrics.json",
        "tuning_info": {
            "backbone": "efficientnetb0",
            "trial": 3,
            "learning_rate": 3e-4,
            "dropout": 0.30,
            "fine_tuning": "frozen",
            "val_macro_f1": 0.4767,
        },
    },
]


def load_json_file(path: Path) -> dict[str, Any]:
    """Safely load JSON data from a file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_final_test_evaluation() -> list[dict[str, Any]]:
    """Run test evaluation on all 3 domain tuned models and compile comparison."""
    logger = setup_logging()
    logger.info("=================================================================")
    logger.info("STARTING FINAL TEST EVALUATION PHASE (UNSEEN TEST SPLITS ONLY)")
    logger.info("=================================================================")

    tuned_base_output_dir = PROJECT_ROOT / "outputs" / "evaluation" / "tuned"
    tuned_base_output_dir.mkdir(parents=True, exist_ok=True)

    comparison_records: list[dict[str, Any]] = []

    for spec in TUNED_EVAL_SPECS:
        dataset_name = spec["dataset"]
        model_path = spec["model_path"]
        output_dir = spec["output_dir"]
        baseline_metrics_path = spec["baseline_metrics_path"]

        logger.info("\n------------------------------------------------------------")
        logger.info("Evaluating %s domain using tuned model: %s", dataset_name.upper(), model_path.name)
        logger.info("Model Path: %s", model_path)
        logger.info("Output Dir: %s", output_dir)
        logger.info("------------------------------------------------------------")

        if not model_path.is_file():
            raise FileNotFoundError(f"Tuned model weights not found at: {model_path}")

        # Run evaluation strictly on the test set
        eval_results = evaluate_model(
            dataset_name=dataset_name,
            model_path=model_path,
            output_directory=output_dir,
        )

        # Load baseline metrics for comparison
        baseline_metrics = load_json_file(baseline_metrics_path)

        base_acc = float(baseline_metrics["test_accuracy"])
        tuned_acc = float(eval_results["test_accuracy"])
        acc_diff = tuned_acc - base_acc
        acc_rel_pct = (acc_diff / base_acc * 100.0) if base_acc > 0 else 0.0

        base_macro_f1 = float(baseline_metrics["macro_f1"])
        tuned_macro_f1 = float(eval_results["macro_f1"])
        macro_f1_diff = tuned_macro_f1 - base_macro_f1
        macro_f1_rel_pct = (macro_f1_diff / base_macro_f1 * 100.0) if base_macro_f1 > 0 else 0.0

        base_weighted_f1 = float(baseline_metrics.get("weighted_f1", 0.0))
        tuned_weighted_f1 = float(eval_results["weighted_f1"])
        weighted_f1_diff = tuned_weighted_f1 - base_weighted_f1
        weighted_f1_rel_pct = (weighted_f1_diff / base_weighted_f1 * 100.0) if base_weighted_f1 > 0 else 0.0

        base_auc = baseline_metrics.get("roc_auc")
        tuned_auc = eval_results.get("roc_auc")
        auc_diff = (tuned_auc - base_auc) if (tuned_auc is not None and base_auc is not None) else None

        base_loss = float(baseline_metrics.get("test_loss", 0.0))
        tuned_loss = float(eval_results["test_loss"])
        loss_diff = tuned_loss - base_loss

        improvement_str = (
            f"Macro F1: {'+' if macro_f1_diff >= 0 else ''}{macro_f1_diff:.4f} ({'+' if macro_f1_rel_pct >= 0 else ''}{macro_f1_rel_pct:.2f}%) | "
            f"Acc: {'+' if acc_diff >= 0 else ''}{acc_diff:.4f} ({'+' if acc_rel_pct >= 0 else ''}{acc_rel_pct:.2f}%)"
        )

        record = {
            "dataset": dataset_name,
            "baseline_model": spec["baseline_model_name"],
            "baseline_test_accuracy": base_acc,
            "baseline_macro_precision": float(baseline_metrics["macro_precision"]),
            "baseline_macro_recall": float(baseline_metrics["macro_recall"]),
            "baseline_macro_f1": base_macro_f1,
            "baseline_weighted_f1": base_weighted_f1,
            "baseline_roc_auc": base_auc,
            "baseline_test_loss": base_loss,
            "tuned_model": spec["tuned_model_name"],
            "tuned_test_accuracy": tuned_acc,
            "tuned_macro_precision": float(eval_results["macro_precision"]),
            "tuned_macro_recall": float(eval_results["macro_recall"]),
            "tuned_macro_f1": tuned_macro_f1,
            "tuned_weighted_f1": tuned_weighted_f1,
            "tuned_roc_auc": tuned_auc,
            "tuned_test_loss": tuned_loss,
            "accuracy_improvement": acc_diff,
            "accuracy_improvement_percent": acc_rel_pct,
            "macro_f1_improvement": macro_f1_diff,
            "macro_f1_improvement_percent": macro_f1_rel_pct,
            "weighted_f1_improvement": weighted_f1_diff,
            "weighted_f1_improvement_percent": weighted_f1_rel_pct,
            "loss_change": loss_diff,
            "auc_change": auc_diff,
            "improvement_summary": improvement_str,
            "tuning_details": spec["tuning_info"],
            "test_samples": eval_results["test_samples"],
        }
        comparison_records.append(record)

    # Save overall summary JSON
    summary_json_path = tuned_base_output_dir / "test_comparison_summary.json"
    summary_json_path.write_text(json.dumps(comparison_records, indent=2), encoding="utf-8")
    logger.info("Saved comparison summary JSON to: %s", summary_json_path)

    # Save overall summary CSV
    summary_csv_path = tuned_base_output_dir / "test_comparison_summary.csv"
    csv_fields = [
        "dataset",
        "baseline_model",
        "baseline_test_accuracy",
        "tuned_model",
        "tuned_test_accuracy",
        "baseline_macro_f1",
        "tuned_macro_f1",
        "accuracy_improvement",
        "accuracy_improvement_percent",
        "macro_f1_improvement",
        "macro_f1_improvement_percent",
        "weighted_f1_improvement",
        "weighted_f1_improvement_percent",
        "baseline_roc_auc",
        "tuned_roc_auc",
        "baseline_test_loss",
        "tuned_test_loss",
        "test_samples",
    ]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in comparison_records:
            writer.writerow(r)
    logger.info("Saved comparison summary CSV to: %s", summary_csv_path)

    # Print comprehensive final report to stdout
    print("\n" + "=" * 120)
    print("FINAL TEST EVALUATION: BASELINE VS. BEST TUNED MODELS")
    print("=" * 120)
    print(
        f"{'Dataset':<8} | {'Baseline Model':<24} | {'Base Acc':<9} | {'Tuned Model':<22} | {'Tuned Acc':<9} | {'Base Macro F1':<13} | {'Tuned Macro F1':<14} | {'F1 Gain (%)':<12} | {'Acc Gain (%)':<12}"
    )
    print("-" * 120)
    for r in comparison_records:
        f1_gain = f"{'+' if r['macro_f1_improvement'] >= 0 else ''}{r['macro_f1_improvement_percent']:.2f}%"
        acc_gain = f"{'+' if r['accuracy_improvement'] >= 0 else ''}{r['accuracy_improvement_percent']:.2f}%"
        print(
            f"{r['dataset'].upper():<8} | {r['baseline_model']:<24} | {r['baseline_test_accuracy']:<9.4f} | {r['tuned_model']:<22} | {r['tuned_test_accuracy']:<9.4f} | {r['baseline_macro_f1']:<13.4f} | {r['tuned_macro_f1']:<14.4f} | {f1_gain:<12} | {acc_gain:<12}"
        )
    print("=" * 120)

    return comparison_records


def main() -> int:
    try:
        run_final_test_evaluation()
        return 0
    except Exception as e:
        LOGGER.exception("Final test evaluation failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
