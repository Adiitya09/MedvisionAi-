"""Run Grad-CAM explainability generation across representative test images.

Evaluates the final selected models:
1. Skin -> EfficientNetB0 baseline (models/skin_model.keras)
2. Eye  -> EfficientNetB0 baseline (models/eye_model.keras)
3. Oral -> EfficientNetB0 tuned (models/tuning/oral/efficientnetb0_best_tuned.keras)

Outputs:
- outputs/explainability/skin/
- outputs/explainability/eye/
- outputs/explainability/oral/
- outputs/explainability/explainability_summary.json
- outputs/explainability/explainability_summary.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Remove the script folder from sys.path to avoid shadowing the package name
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from config import CONFIG
from explainability.explainability import generate_gradcam
from utils.helpers import setup_logging

LOGGER = logging.getLogger(__name__)

# Representative test samples selected from test predictions
REPRESENTATIVE_SAMPLES = [
    # -------------------------------------------------------------
    # SKIN DOMAIN (EfficientNetB0 Baseline)
    # -------------------------------------------------------------
    {
        "dataset": "skin",
        "sample_id": "skin_sample_01_melanoma_correct",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "skin_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "FYP skin disease Dataset" / "Melanoma" / "ISIC_7340247.jpg",
        "true_class": "Melanoma",
        "case_type": "Correct (High Confidence Lesion)",
    },
    {
        "dataset": "skin",
        "sample_id": "skin_sample_02_acne_correct",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "skin_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "FYP skin disease Dataset" / "Acne" / "levle3_126_jpg.rf.f8a83210ecae1ee13f571a90a20e0b40.jpg",
        "true_class": "Acne",
        "case_type": "Correct (High Confidence Inflammatory)",
    },
    {
        "dataset": "skin",
        "sample_id": "skin_sample_03_bcc_correct",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "skin_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "FYP skin disease Dataset" / "Basal Cell Carcinoma" / "BCN_0000014498.jpg",
        "true_class": "Basal Cell Carcinoma",
        "case_type": "Correct (Epithelial Carcinoma)",
    },
    {
        "dataset": "skin",
        "sample_id": "skin_sample_04_melanoma_misclassified",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "skin_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "FYP skin disease Dataset" / "Melanoma" / "ISIC_0000290.jpg",
        "true_class": "Melanoma",
        "case_type": "Misclassified (Melanoma -> BCC)",
    },

    # -------------------------------------------------------------
    # EYE DOMAIN (EfficientNetB0 Baseline)
    # -------------------------------------------------------------
    {
        "dataset": "eye",
        "sample_id": "eye_sample_01_cataract_correct",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "eye_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Eye disease" / "C" / "_253_1359558.jpg",
        "true_class": "C (Cataract)",
        "case_type": "Correct (High Confidence Cataract)",
    },
    {
        "dataset": "eye",
        "sample_id": "eye_sample_02_dr_correct",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "eye_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Eye disease" / "D" / "DR1203.jpg",
        "true_class": "D (Diabetic Retinopathy)",
        "case_type": "Correct (Diabetic Retinopathy)",
    },
    {
        "dataset": "eye",
        "sample_id": "eye_sample_03_normal_correct",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "eye_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Eye disease" / "N" / "658_left.jpg",
        "true_class": "N (Normal)",
        "case_type": "Correct (Normal Fundus)",
    },
    {
        "dataset": "eye",
        "sample_id": "eye_sample_04_myopia_misclassified",
        "model_name": "EfficientNetB0 (Baseline)",
        "model_path": PROJECT_ROOT / "models" / "eye_model.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Eye disease" / "M" / "Myopia164.jpg",
        "true_class": "M (Myopia)",
        "case_type": "Misclassified (Myopia -> Glaucoma)",
    },

    # -------------------------------------------------------------
    # ORAL DOMAIN (EfficientNetB0 Tuned)
    # -------------------------------------------------------------
    {
        "dataset": "oral",
        "sample_id": "oral_sample_01_noncancer_correct",
        "model_name": "EfficientNetB0 (Tuned)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "oral" / "efficientnetb0_best_tuned.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Oral Cancer" / "NON CANCER" / "382.jpeg",
        "true_class": "NON CANCER",
        "case_type": "Correct (High Confidence Non-Cancer)",
    },
    {
        "dataset": "oral",
        "sample_id": "oral_sample_02_cancer1_correct",
        "model_name": "EfficientNetB0 (Tuned)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "oral" / "efficientnetb0_best_tuned.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Oral Cancer" / "CANCER 1" / "215.jpeg",
        "true_class": "CANCER 1",
        "case_type": "Correct (Cancer Subtype 1)",
    },
    {
        "dataset": "oral",
        "sample_id": "oral_sample_03_cancer_correct",
        "model_name": "EfficientNetB0 (Tuned)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "oral" / "efficientnetb0_best_tuned.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Oral Cancer" / "CANCER" / "486.jpeg",
        "true_class": "CANCER",
        "case_type": "Correct (Oral Carcinoma)",
    },
    {
        "dataset": "oral",
        "sample_id": "oral_sample_04_cancer_misclassified",
        "model_name": "EfficientNetB0 (Tuned)",
        "model_path": PROJECT_ROOT / "models" / "tuning" / "oral" / "efficientnetb0_best_tuned.keras",
        "image_path": PROJECT_ROOT / "data" / "Test" / "Oral Cancer" / "CANCER" / "157.jpeg",
        "true_class": "CANCER",
        "case_type": "Misclassified (CANCER -> CANCER 1)",
    },
]


def verify_image_file(path: Path) -> bool:
    """Verify that an image file exists, is non-empty, and can be opened."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def run_explainability_pipeline() -> list[dict[str, Any]]:
    """Execute Grad-CAM across all representative samples and compile reports."""
    logger = setup_logging()
    logger.info("=================================================================")
    logger.info("STARTING GRAD-CAM EXPLAINABILITY EVALUATION ON TEST SAMPLES")
    logger.info("=================================================================")

    base_output_dir = PROJECT_ROOT / "outputs" / "explainability"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for item in REPRESENTATIVE_SAMPLES:
        dataset = item["dataset"]
        sample_id = item["sample_id"]
        model_path = item["model_path"]
        image_path = item["image_path"]
        domain_output_dir = base_output_dir / dataset
        domain_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n------------------------------------------------------------")
        logger.info("Generating Grad-CAM for %s [%s]", dataset.upper(), sample_id)
        logger.info("Image: %s", image_path.name)
        logger.info("Model: %s", model_path.name)
        logger.info("------------------------------------------------------------")

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Run Grad-CAM
        gradcam_res = generate_gradcam(
            image_path=str(image_path),
            dataset_name=dataset,
            model_path=model_path,
            output_directory=domain_output_dir,
            filename_prefix=sample_id,
        )

        orig_path = Path(gradcam_res["original_path"])
        heat_path = Path(gradcam_res["heatmap_path"])
        over_path = Path(gradcam_res["overlay_path"])

        # Verification step
        orig_valid = verify_image_file(orig_path)
        heat_valid = verify_image_file(heat_path)
        over_valid = verify_image_file(over_path)

        if not (orig_valid and heat_valid and over_valid):
            raise RuntimeError(f"Generated image verification failed for {sample_id}")

        record = {
            "dataset": dataset,
            "model": item["model_name"],
            "model_file": model_path.name,
            "sample_id": sample_id,
            "image_filename": image_path.name,
            "image_path": str(image_path),
            "true_class": item["true_class"],
            "predicted_class": gradcam_res["predicted_class"],
            "confidence": gradcam_res["confidence"],
            "confidence_percent": f"{gradcam_res['confidence'] * 100:.2f}%",
            "target_class": gradcam_res["target_class"],
            "gradcam_layer": gradcam_res["target_layer"],
            "case_type": item["case_type"],
            "is_correct": (
                gradcam_res["predicted_class"].strip().lower()
                == item["true_class"].split(" ")[0].strip().lower()
            ),
            "original_path": str(orig_path),
            "heatmap_path": str(heat_path),
            "overlay_path": str(over_path),
            "images_verified": True,
        }
        records.append(record)

    # Save summary JSON
    summary_json = base_output_dir / "explainability_summary.json"
    summary_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Saved explainability summary JSON to: %s", summary_json)

    # Save summary CSV
    summary_csv = base_output_dir / "explainability_summary.csv"
    csv_fields = [
        "dataset",
        "model",
        "sample_id",
        "image_filename",
        "true_class",
        "predicted_class",
        "confidence",
        "confidence_percent",
        "gradcam_layer",
        "case_type",
        "original_path",
        "heatmap_path",
        "overlay_path",
        "images_verified",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    logger.info("Saved explainability summary CSV to: %s", summary_csv)

    # Print summary table to stdout
    print("\n" + "=" * 120)
    print("GRAD-CAM EXPLAINABILITY SUMMARY TABLE")
    print("=" * 120)
    print(
        f"{'Dataset':<8} | {'Model':<22} | {'Image':<32} | {'Predicted Class':<22} | {'Confidence':<12} | {'Grad-CAM Layer':<18}"
    )
    print("-" * 120)
    for r in records:
        print(
            f"{r['dataset'].upper():<8} | {r['model']:<22} | {r['image_filename']:<32} | {r['predicted_class']:<22} | {r['confidence_percent']:<12} | {r['gradcam_layer']:<18}"
        )
    print("=" * 120)

    return records


def main() -> int:
    try:
        run_explainability_pipeline()
        return 0
    except Exception as e:
        LOGGER.exception("Explainability evaluation failed: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
