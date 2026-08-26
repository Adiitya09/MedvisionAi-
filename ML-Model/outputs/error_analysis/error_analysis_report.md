# Phase 7: Deep Error Analysis & Dataset Improvement Report

**Project**: MedvisionAI Medical Diagnostic Suite  
**Date**: August 26, 2026  
**Status**: Comprehensive Diagnostic Analysis Completed (No modifications made to datasets or model weights)

---

## Executive Summary

Phase 7 conducted an exhaustive, empirical root-cause diagnostic investigation across all three medical domains (**Skin**, **Eye**, and **Oral Cancer**). Rather than assuming performance bottlenecks, every potential limiting factor was systematically measured: **class distribution imbalance**, **per-class precision/recall/F1 decomposition**, **confusion matrix topology**, **prediction confidence calibration**, **image quality and resolution anomalies**, **cryptographic and perceptual cross-split duplicate leakage**, **preprocessing pipeline integrity**, and **Grad-CAM attention maps**.

### High-Level Diagnostic Summary

| Dataset | Current Test Acc | Macro F1 | Imbalance Ratio | Primary Root-Cause Bottleneck | Leakage Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SKIN** | 76.32% | 0.6851 | **25.68 : 1** | Extreme class imbalance & high clinical confusion between Melanoma and benign Melanocytic Nevus (35.8% of Melanomas misclassified as Nevus). | **0% Leakage** (Clean split) |
| **EYE** | 68.13% | 0.6125 | **7.60 : 1** | Heavy dominance of `D` (Diabetic Retinopathy) & `N` (Normal); high cross-condition image overlap between `Glaucoma` and `Myopia` in raw dataset; low AMD (`A`) recall (11.5%). | **26 Train-Test Leakage Groups** (Due to multi-label patient images assigned to multiple single-label folders) |
| **ORAL** | 48.82% | 0.4262 | **1.94 : 1** | **Severe Artificial Sub-Class Redundancy**: `CANCER` vs `CANCER 1` and `NON CANCER` vs `NON CANCER 2` contain identical duplicate images across classes. 67.8% of all errors are intra-condition confusions. | **108 Train-Test Leakage Groups** & **588 Exact Duplicate Groups** |

---

## 1. Class Distribution Analysis

### A. Skin Disease Domain (ISIC / HAM10000 / PAD-UFES-20)
- **Total Images**: 22,719 (Train: 18,171 | Validation: 2,268 | Test: 2,280)
- **Class Imbalance Ratio**: **25.68 : 1** (Maximum class: `Melanocytic Nevus` with 7,010 train images vs Minimum class: `Dermatofibroma` with 273 train images).

| Class Name | Train Count | Train % | Val Count | Test Count | Total Count | Total % | Imbalance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Melanocytic Nevus (Nevus)** | 7,010 | 38.58% | 876 | 877 | 8,763 | 38.57% | Heavily Dominant Class |
| **Melanoma** | 3,837 | 21.12% | 479 | 481 | 4,797 | 21.11% | Secondary Dominant Class |
| **Acne** | 2,030 | 11.17% | 253 | 255 | 2,538 | 11.17% | Balanced |
| **Seborrheic Keratosis** | 1,977 | 10.88% | 247 | 248 | 2,472 | 10.88% | Balanced |
| **Basal Cell Carcinoma** | 1,965 | 10.81% | 245 | 247 | 2,457 | 10.81% | Balanced |
| **Actinic Keratosis** | 577 | 3.18% | 72 | 73 | 722 | 3.18% | Moderately Underrepresented |
| **Squamous Cell Carcinoma** | 502 | 2.76% | 62 | 64 | 628 | 2.76% | Severely Underrepresented |
| **Dermatofibroma** | 273 | 1.50% | 34 | 35 | 342 | 1.51% | Severely Underrepresented |

---

### B. Eye Disease Domain (ODIR-5K / Kaggle Eye Diseases)
- **Total Images**: 10,438 (Train: 8,347 | Validation: 1,040 | Test: 1,051)
- **Class Imbalance Ratio**: **7.60 : 1** (`D` Diabetic Retinopathy has 2,500 train images vs `H` Hypertensive Retinopathy with 329).

| Class Name | Condition | Train Count | Train % | Val Count | Test Count | Total Count | Imbalance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D** | Diabetic Retinopathy | 2,500 | 29.95% | 312 | 314 | 3,126 | Heavily Dominant |
| **N** | Normal Fundus | 2,397 | 28.72% | 299 | 301 | 2,997 | Heavily Dominant |
| **G** | Glaucoma | 1,316 | 15.77% | 164 | 165 | 1,645 | Moderate |
| **C** | Cataract | 806 | 9.66% | 100 | 102 | 1,008 | Moderate |
| **M** | Pathological Myopia | 591 | 7.08% | 73 | 75 | 739 | Underrepresented |
| **A** | AMD (Macular Degeneration) | 408 | 4.89% | 51 | 52 | 511 | Severely Underrepresented |
| **H** | Hypertensive Retinopathy | 329 | 3.94% | 41 | 42 | 412 | Severely Underrepresented |

---

### C. Oral Cancer Domain
- **Total Images**: 1,651 (Train: 1,319 | Validation: 162 | Test: 170)
- **Class Imbalance Ratio**: **1.94 : 1** (Relatively balanced nominally, but with redundant splits).

| Class Name | Train Count | Train % | Val Count | Test Count | Total Count | Semantic Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CANCER** | 383 | 29.04% | 47 | 49 | 479 | Malignant Sub-group A |
| **CANCER 1** | 383 | 29.04% | 47 | 49 | 479 | Malignant Sub-group B (Identical duplicate images) |
| **NON CANCER** | 356 | 26.99% | 44 | 46 | 446 | Benign Sub-group A |
| **NON CANCER 2** | 197 | 14.94% | 24 | 26 | 247 | Benign Sub-group B (Duplicate overlap) |

---

## 2. Per-Class Test Performance Decomposition

### A. Skin Disease Domain (`models/skin_model.keras`)
- **Overall Metrics**: Accuracy = **76.32%**, Macro F1 = **0.6851**, Weighted F1 = **0.7567**

| Class Name | Precision | Recall | F1-Score | Support | Clinical Impact & Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Acne** | **0.9798** | **0.9529** | **0.9662** | 255 | Near-perfect isolation; visually distinct inflammatory papules. |
| **Melanocytic Nevus** | 0.7544 | **0.8791** | 0.8120 | 877 | High recall driven by prior probability dominance. |
| **Seborrheic Keratosis** | 0.7299 | 0.8065 | 0.7663 | 248 | Well-separated benign keratosis features. |
| **Actinic Keratosis** | 0.7759 | 0.6164 | 0.6870 | 73 | Moderate recall; confused with BCC and Seborrheic Keratosis. |
| **Basal Cell Carcinoma** | 0.7429 | 0.6316 | 0.6827 | 247 | Moderate recall; 19% confused with Nevus. |
| **Melanoma** | 0.7007 | **0.5988** | 0.6457 | 481 | **Critical Deficit**: Low recall for life-threatening malignancy. |
| **Squamous Cell Carcinoma** | 0.7812 | **0.3906** | 0.5208 | 64 | **Severe Deficit**: Underrepresented class with $< 40\%$ recall. |
| **Dermatofibroma** | 0.4800 | **0.3429** | **0.4000** | 35 | **Worst Class**: Severe class starvation (only 273 train images). |

---

### B. Eye Disease Domain (`models/eye_model.keras`)
- **Overall Metrics**: Accuracy = **68.13%**, Macro F1 = **0.6125**, Weighted F1 = **0.6680**

| Class Name | Condition | Precision | Recall | F1-Score | Support | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C** | Cataract | **0.9880** | **0.8039** | **0.8865** | 102 | High precision; distinct lens opacification. |
| **G** | Glaucoma | 0.7902 | 0.6848 | 0.7338 | 165 | Strong disc cupping detection. |
| **N** | Normal | 0.6260 | **0.8007** | 0.7026 | 301 | High false positive rate from subtle disease confusion. |
| **D** | Diabetic Retinopathy | 0.6005 | 0.7325 | 0.6600 | 314 | Moderate precision; absorbs minority disease classes. |
| **H** | Hypertensive Retinopathy | **1.0000** | **0.4762** | 0.6452 | 42 | Zero false positives, but $< 50\%$ recall (confused with Normal & DR). |
| **M** | Pathological Myopia | 0.7742 | **0.3200** | 0.4528 | 75 | Low recall; 36% misclassified as DR, 26.7% as Glaucoma. |
| **A** | AMD (Macular Degeneration) | **1.0000** | **0.1154** | **0.2069** | 52 | **Severe Failure**: 75% of AMD cases misclassified as Diabetic Retinopathy. |

---

### C. Oral Cancer Domain (`models/tuning/oral/efficientnetb0_best_tuned.keras`)
- **Overall Metrics**: Accuracy = **48.82%**, Macro F1 = **0.4262**, Weighted F1 = **0.4538**

| Class Name | Precision | Recall | F1-Score | Support | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NON CANCER** | 0.5270 | **0.8478** | **0.6500** | 46 | Absorbs benign subclass `NON CANCER 2`. |
| **CANCER 1** | 0.4444 | 0.4898 | 0.4660 | 49 | Heavy mutual confusion with `CANCER`. |
| **CANCER** | 0.4595 | 0.3469 | 0.3953 | 49 | 49% misclassified into `CANCER 1`. |
| **NON CANCER 2** | **0.6000** | **0.1154** | **0.1935** | 26 | **Severe Failure**: 73.1% misclassified into `NON CANCER`. |

---

## 3. Confusion Matrix & Top Error-Pair Topology

### Top Class Confusion Pairs Ranked by Frequency

| Domain | True Class | Predicted Class | Error Count | % of True Class | % of Total Domain Errors | Clinical & Technical Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SKIN** | **Melanoma** | **Melanocytic Nevus** | **172** | **35.76%** | **31.85%** | Overlap in pigment network architecture; prior probability bias favoring dominant Nevus class. |
| **SKIN** | **Melanocytic Nevus** | **Melanoma** | **75** | 8.55% | 13.89% | Atypical dysplastic nevi exhibiting borderline asymmetric dermoscopy. |
| **SKIN** | **Basal Cell Carcinoma** | **Melanocytic Nevus** | **47** | 19.03% | 8.70% | Pigmented nodular BCC mimicking benign melanocytic lesions. |
| **SKIN** | **Squamous Cell Carcinoma**| **Basal Cell Carcinoma** | **18** | 28.13% | 3.33% | Keratinizing epidermal carcinoma morphological overlap. |
| **SKIN** | **Dermatofibroma** | **Seborrheic Keratosis** | **10** | 28.57% | 1.85% | Severe sample sparsity for Dermatofibroma. |
| **EYE** | **Diabetic Retinopathy (`D`)**| **Normal (`N`)** | **81** | 25.80% | 24.18% | Mild non-proliferative DR with subtle isolated microaneurysms. |
| **EYE** | **Normal (`N`)** | **Diabetic Retinopathy (`D`)**| **52** | 17.28% | 15.52% | Physiological retinal vessel reflections and choroidal tessellation. |
| **EYE** | **AMD (`A`)** | **Diabetic Retinopathy (`D`)**| **39** | **75.00%** | 11.64% | Shared pathology: hard exudates and hemorrhages in the macula. |
| **EYE** | **Pathological Myopia (`M`)**| **Diabetic Retinopathy (`D`)**| **27** | 36.00% | 8.06% | Myopic macular degenerations mistaken for diabetic retinopathy. |
| **EYE** | **Pathological Myopia (`M`)**| **Glaucoma (`G`)** | **20** | 26.67% | 5.97% | Tilted myopic optic discs mimicking glaucomatous optic cupping. |
| **ORAL** | **CANCER** | **CANCER 1** | **24** | **48.98%** | **27.59%** | **Artificial Subclasses**: Identical/duplicate images assigned to both classes. |
| **ORAL** | **NON CANCER 2** | **NON CANCER** | **19** | **73.08%** | **21.84%** | **Artificial Subclasses**: Identical mucosal conditions arbitrarily split. |
| **ORAL** | **CANCER 1** | **CANCER** | **16** | **32.65%** | **18.39%** | **Artificial Subclasses**: Reverse duplicate confusion. |

---

## 4. Prediction Confidence Distribution Analysis

| Dataset | Correct Predictions Mean Conf | Incorrect Predictions Mean Conf | High-Confidence Errors ($\text{Conf} \ge 0.75$) | High-Conf Error % | Diagnostic Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SKIN** | **79.49%** ($\pm 18.2\%$) | **57.93%** ($\pm 18.8\%$) | **95 cases** | **17.59%** | High-confidence errors occur almost exclusively on Melanoma $\leftrightarrow$ Nevus due to strong feature overlaps and high class prior belief. |
| **EYE** | **67.69%** ($\pm 18.0\%$) | **54.65%** ($\pm 14.5\%$) | **27 cases** | **8.06%** | Majority of errors are uncertain ($\text{conf} < 0.60$), reflecting subtle multi-condition retinopathies. |
| **ORAL** | **52.58%** ($\pm 16.4\%$) | **49.06%** ($\pm 15.0\%$) | **1 case** | **1.15%** | Low average confidence across all predictions ($50.8\%$), proving the network is perpetually uncertain due to contradictory duplicate labels in the training set. |

---

## 5. Image Quality & Resolution Findings

All 34,808 image files across Train, Validation, and Test splits were programmatically verified for file corruption, resolution anomalies, color modes, and pixel variance:

1. **File Corruptions**: **0 corrupted files** across all datasets. Every image is valid and decodable by libjpeg/libpng.
2. **Color Modes**:
   - **Skin**: 21,529 RGB images and **1,190 RGBA images** (from the smartphone clinical photo subset `PAD-UFES-20` in Actinic Keratosis and BCC). `tf.keras.utils.image_dataset_from_directory(color_mode="rgb")` properly drops the alpha channel without error.
   - **Eye**: 100% RGB images.
   - **Oral**: 1,455 RGB images and **196 Palette/RGBA images** (from PNG screen captures).
3. **Dimensions & Extreme Aspect Ratios**:
   - **Skin**: High resolution ($300 \times 300$ to $1024 \times 1024$), standard aspect ratios ($1.00$ to $1.33$).
   - **Eye**: Standard fundus photographs ($512 \times 512$ to $2048 \times 1536$), circular mask.
   - **Oral**: **14 images** with small native dimensions ($< 100 \text{ px}$), with extreme aspect ratios up to $3.2:1$ from tight cropping.

---

## 6. Duplicate & Data Leakage Findings

Cryptographic MD5 hashing and perceptual difference hashing (dHash) uncovered critical structural findings:

### A. Skin Domain
- **Exact Duplicate Groups**: **0**
- **Train-Test Leakage**: **0% (Clean split)**
- **Train-Val Leakage**: **0%**
- **Assessment**: The ISIC/HAM10000 dataset partition was properly executed by lesion ID, preventing patient overlap between train and test.

### B. Eye Domain
- **Exact Duplicate Groups**: **161 groups**
- **Train-Test Leakage**: **26 image groups**
- **Train-Val Leakage**: **28 image groups**
- **Root Cause**: The raw ODIR dataset contains multi-label patient records. An individual patient eye with both Glaucoma and Pathological Myopia was saved simultaneously into the `Glaucoma` folder (e.g. `Glaucoma379.jpg`) and the `Myopia` folder (e.g. `Myopia164.jpg`). When single-label directory partitioning was applied, identical photos were placed into different splits under different class names.
- **Impact**: Explains high-confidence cross-condition errors where the model learned the training set's duplicate label.

### C. Oral Cancer Domain
- **Exact Duplicate Groups**: **588 groups**
- **Train-Test Leakage**: **108 image groups**
- **Train-Val Leakage**: **93 image groups**
- **Within-Split Duplicates**: **377 groups**
- **Root Cause**: The raw Oral dataset contains extensive exact copy-pasted duplicates across folders (`CANCER` vs `CANCER 1` and `NON CANCER` vs `NON CANCER 2`).
- **Impact**: Directly responsible for capping test accuracy at $48.82\%$. When identical images are assigned contradictory labels, the theoretical Bayes optimal accuracy for those samples is at most $50\%$.

---

## 7. Preprocessing Verification

Inspection of [utils/dataset_loader.py](file:///e:/Mega%20project/MedvisionAi-/ML-Model/utils/dataset_loader.py), [utils/model_builder.py](file:///e:/Mega%20project/MedvisionAi-/ML-Model/utils/model_builder.py), and [inference/predict.py](file:///e:/Mega%20project/MedvisionAi-/ML-Model/inference/predict.py):

- **Image Resizing**: Consistent bilinear interpolation to `(224, 224)` across all loaders and inference entry points.
- **Normalization & Preprocessing Layer**: EfficientNet models utilize an internal `Lambda(preprocess_input)` layer. Raw pixel values in $[0, 255]$ are supplied consistently without accidental double normalization (`/ 255.0`).
- **Channel Handling**: `color_mode="rgb"` ensures consistent 3-channel tensors across PNG, JPEG, and BMP formats.
- **Conclusion**: The preprocessing pipeline is fully verified, mathematically consistent, and free of defects.

---

## 8. Grad-CAM Error Attention Observations

Grad-CAM heatmaps generated for high-confidence and top-confusion errors reveal distinct operational modes:

1. **Skin Domain Errors**:
   - **Observation**: For Melanomas misclassified as Nevus, the model focuses precisely on the central pigment network and lesion core rather than background skin or hair artifacts.
   - **Diagnosis**: The error is not caused by shortcut learning or background noise, but rather subtle dermoscopic feature overlap and strong prior probability bias toward Nevi.
2. **Eye Domain Errors**:
   - **Observation**: In AMD misclassified as Diabetic Retinopathy, the heatmap highlights macular hard exudates and drusen. Because both conditions present with yellowish retinal exudates, the model defaults to the dominant DR class.
   - **Diagnosis**: Visual confusion of similar pathological lesions exacerbated by severe class imbalance ($2,500$ DR vs $408$ AMD).
3. **Oral Domain Errors**:
   - **Observation**: The model focuses accurately on mucosal ulcerations and indurations, ignoring teeth and medical retractors.
   - **Diagnosis**: Feature extraction is functioning well, but arbitrary subclass partitioning (`CANCER` vs `CANCER 1`) forces the network into random guessing.

---

## 9. Root-Cause Diagnostic Synthesis

### SKIN:
- **Major Error Sources**: Class imbalance (25.68:1); 35.8% of Melanomas misclassified as benign Nevi; severe recall starvation on minority classes (`Dermatofibroma` at 34.3% recall, `Squamous Cell Carcinoma` at 39.1% recall).
- **Dataset Integrity**: High (0% leakage, no corruptions).
- **Opportunities**: Class-weighted focal loss, targeted minority class augmentation, and cost-sensitive thresholding for Melanoma sensitivity.

### EYE:
- **Major Error Sources**: Dominance of `D` and `N` (58.7% of dataset); severe underperformance on `A` (AMD recall 11.5%) and `M` (Myopia recall 32.0%); multi-label patient image leakage between Glaucoma and Myopia.
- **Dataset Integrity**: Moderate (multi-condition duplicates across folders in raw ODIR data).
- **Opportunities**: Deduplication of multi-condition fundus images, class-balanced sampling/weighting, and focal loss for AMD/Myopia/Hypertension.

### ORAL:
- **Major Error Sources**: **Artificial Sub-Class Splitting** (`CANCER` vs `CANCER 1`, `NON CANCER` vs `NON CANCER 2`) with 588 exact duplicate groups and 108 train-test leakage groups; inter-subclass confusion constitutes 67.8% of all errors.
- **Dataset Integrity**: Severely Compromised by redundant duplicate folders in raw data.
- **Opportunities**: **Consolidate 4 classes into 2 clean clinical classes (`CANCER` vs `NON CANCER`)** and deduplicate dataset before retraining. This single change will immediately resolve over 65% of all errors.

---

## 10. Prioritized Problem Summary

| Dataset | Main Problem | Evidence | Priority | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| **ORAL** | **Contradictory Duplicate Subclasses** | 588 duplicate groups; 108 train-test leakage groups; 67.8% of errors are between duplicate sub-classes (`CANCER` $\leftrightarrow$ `CANCER 1`, `NON CANCER` $\leftrightarrow$ `NON CANCER 2`). | **CRITICAL (P0)** | **Merge classes into clean 2-class setup (`CANCER`, `NON CANCER`)**, deduplicate hashes, and regenerate clean train/val/test splits. |
| **SKIN** | **Melanoma Under-diagnosis & Class Imbalance** | 25.68:1 imbalance ratio; 35.8% of Melanomas misclassified as Nevus (31.8% of all errors); Dermatofibroma recall 34.3%. | **HIGH (P1)** | Implement **Focal Loss** / **Class-Weighted Loss**, targeted minority augmentation, and cost-sensitive decision thresholding. |
| **EYE** | **AMD Suppression & Cross-Label Duplication** | 7.60:1 imbalance; AMD recall is only 11.5% (75% classified as DR); 26 train-test cross-condition leakage groups (Glaucoma $\leftrightarrow$ Myopia). | **HIGH (P1)** | **Remove cross-condition duplicate leaks**, apply class-balanced focal loss, and add specialized macular region augmentation. |
| **ALL** | **Backbone Fine-Tuning Depth** | Baseline models used frozen backbones or shallow top-layer tuning during initial training. | **MEDIUM (P2)** | Apply **progressive 2-stage fine-tuning** (unfreezing top 30-50 layers of EfficientNetB0 with low learning rate $1\times 10^{-5}$) on cleaned datasets. |

---

## 11. Next Training Recommendations

*Ranked strictly by evidence-backed expected value:*

1. **[ORAL] Dataset Consolidation & Deduplication (Expected Gain: $+35\%\text{ to }+45\%$ Accuracy)**:
   - Merge `CANCER 1` into `CANCER`, and `NON CANCER 2` into `NON CANCER`.
   - Remove 588 duplicate image pairs to eliminate contradictory labels.
   - Retrain on clean binary oral cancer dataset.
2. **[SKIN & EYE] Focal Loss & Cost-Sensitive Class Weighting (Expected Gain: $+6\%\text{ to }+10\%$ Macro F1)**:
   - Replace standard Categorical Crossentropy with **$\gamma=2.0$ Focal Loss** or compute inverse-frequency class weights $w_c = \frac{N}{K \cdot N_c}$.
   - Heavily penalize false negatives on Melanoma and AMD.
3. **[EYE] Multi-Condition Duplicate Cleaning (Expected Gain: $+5\%\text{ to }+8\%$ Macro F1)**:
   - Filter out duplicate image hashes that appear under multiple distinct disease labels in raw ODIR data.
4. **[ALL] Progressive Unfrozen Backbone Fine-Tuning (Expected Gain: $+3\%\text{ to }+6\%$ Accuracy)**:
   - Train top classification head for 10 epochs ($lr = 3\times 10^{-4}$), then unfreeze the top 40 convolutional layers of EfficientNetB0 and fine-tune with cosine decay ($lr = 1\times 10^{-5}$).
5. **[SKIN & EYE] Targeted Minor Class Data Augmentation (Expected Gain: $+3\%\text{ to }+5\%$ Minority Recall)**:
   - Apply heavy affine transformations, color jittering, and CutMix/Mixup specifically to underrepresented classes (`Dermatofibroma`, `SCC`, `AMD`, `Hypertension`).

---

*(Phase 7 complete. Analysis only; no data modified; ready for user direction on next phase).*
