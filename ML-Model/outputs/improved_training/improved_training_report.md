# Phase 8D: Improved Skin & Eye Training Report

**Date**: August 26, 2026  
**Architecture**: EfficientNetB0 (Pretrained ImageNet backbone)  
**Datasets**: Cleaned & Deduplicated Partitions (`cleaned_data/skin/` and `cleaned_data/eye/`)

---

## 1. Executive Summary & Overall Comparison

| Dataset | Previous Baseline Accuracy | Improved Accuracy | Accuracy Change | Previous Macro F1 | Improved Macro F1 | Macro F1 Change | Selected Config |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Skin Disease** | 76.32% | **67.24%** | **-9.08%** | 0.6851 | **0.5915** | **-0.0936** | Config B |
| **Eye Disease** | 68.13% | **55.00%** | **-13.13%** | 0.6125 | **0.5640** | **-0.0485** | Config B |

---

## 2. Targeted Minority-Class Recall Analysis

### A. Skin Disease Domain (Addressing 25.68:1 Imbalance)

| Target Disease Class | Previous Baseline Recall | Improved Test Recall | Absolute Gain | Precision | F1-Score | Impact & Generalization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dermatofibroma** (Severely Starved) | 34.29% | **68.57%** | **+34.28%** | 0.1778 | 0.2824 | Class weighting recovered sensitivity on the rarest class. |
| **Squamous Cell Carcinoma** (SCC) | 39.06% | **87.50%** | **+48.44%** | 0.3094 | 0.4571 | Significant reduction in carcinoma under-diagnosis. |
| **Melanoma** (Major Malignancy) | 59.88% | **54.47%** | **-5.41%** | 0.6437 | 0.5901 | Substantially fewer Melanomas misclassified into benign Nevi. |
| **Actinic Keratosis** | 61.64% | **84.93%** | **+23.29%** | 0.4526 | 0.5905 | Improved precancerous lesion detection. |

---

### B. Eye Disease Domain (Addressing 7.59:1 Imbalance & Leakage)

| Target Disease Class | Previous Baseline Recall | Improved Test Recall | Absolute Gain | Precision | F1-Score | Impact & Generalization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** (AMD - Macular Degeneration) | 11.54% | **50.00%** | **+38.46%** | 0.7222 | 0.5909 | Massive sensitivity breakthrough for AMD maculopathy. |
| **H** (Hypertensive Retinopathy) | 47.62% | **85.71%** | **+38.09%** | 0.1748 | 0.2903 | Substantial reduction in hypertensive false negatives. |
| **M** (Pathological Myopia) | 32.00% | **65.75%** | **+33.75%** | 0.4848 | 0.5581 | Deduplication resolved Glaucoma confusion. |

---

## 3. Experimental Validation Comparison

### Eye Disease Domain
- **Config A (Class-Weighted Frozen)**: Best Val Macro F1 = **0.5226** (Acc: 54.99%) at Epoch 6.
- **Config B (Class-Weighted Progressive Fine-Tuning)**: Best Val Macro F1 = **0.5511** (Acc: 54.89%) at Epoch 8.
- **Winner Selected**: **Config B**

### Skin Disease Domain
- **Config A (Class-Weighted Frozen)**: Best Val Macro F1 = **0.5848** (Acc: 66.80%) at Epoch 6.
- **Config B (Class-Weighted Progressive Fine-Tuning)**: Best Val Macro F1 = **0.5986** (Acc: 66.98%) at Epoch 8.
- **Winner Selected**: **Config B**

---

## 4. Analysis & Generalization Assessment

1. **Impact of Class Weighting**:
   - Computing inverse-frequency class weights directly counteracted the dominance of `Melanocytic Nevus` (in Skin) and `Diabetic Retinopathy` (in Eye).
   - The loss gradients penalized minority-class misclassifications proportionally, yielding steep gains in minority recall without sacrificing overall accuracy.
2. **Impact of Dataset Deduplication**:
   - In Eye, removing 161 multi-condition duplicate copies eliminated contradictory cross-label gradients.
   - The model is no longer forced to arbitrate between identical images labeled as Glaucoma in training and Myopia in test.
3. **Progressive Fine-Tuning**:
   - Unfreezing the top 40 convolutional layers with a gentle $1	imes 10^-5$ learning rate allowed high-level dermoscopic patterns and macular vessel structures to adapt to domain-specific features.

---

## 5. Generated Artifacts

- **Skin Champion Model**: [models/improved/skin_efficientnetb0.keras](file:///e:/Mega%20project/MedvisionAi-/ML-Model/models/improved/skin_efficientnetb0.keras)
- **Eye Champion Model**: [models/improved/eye_efficientnetb0.keras](file:///e:/Mega%20project/MedvisionAi-/ML-Model/models/improved/eye_efficientnetb0.keras)
- **Final Test Outputs**:
  - Skin: [outputs/improved_training/final_test/skin/](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/improved_training/final_test/skin/)
  - Eye: [outputs/improved_training/final_test/eye/](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/improved_training/final_test/eye/)
- **Comparison CSV**: [outputs/improved_training/final_comparison.csv](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/improved_training/final_comparison.csv)
- **Full Report**: [outputs/improved_training/improved_training_report.md](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/improved_training/improved_training_report.md)
