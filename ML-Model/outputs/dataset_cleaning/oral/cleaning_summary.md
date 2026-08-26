# Phase 8A: Oral Dataset Cleaning & Reconstruction Summary

## 1. Overview & Key Metrics

- **Original Dataset Path**: `data/Train/Oral Cancer/`, `data/Validation/Oral Cancer/`, `data/Test/Oral Cancer/` (Preserved intact)
- **Cleaned Dataset Path**: `cleaned_data/oral/`
- **Original Total Images**: **1,651**
- **Exact Duplicate Copies Removed**: **588**
- **Conflicting Cross-Class Images Removed**: **1 image pair** (`00b5c569...` labeled as both `CANCER 1` and `NON CANCER 2`)
- **Corrupted Images**: **0**
- **Final Cleaned Dataset Images**: **1,062 unique, valid images**

---

## 2. Class Consolidation Mapping

The artificial 4-class taxonomy was consolidated into a clean, clinically grounded 2-class binary diagnostic taxonomy:

| Original Subclass Folder | Consolidated Clean Class | Diagnostic Role |
| :--- | :--- | :--- |
| `CANCER` | **CANCER** | Malignant Oral Squamous Cell Carcinoma / Dysplasia |
| `CANCER 1` | **CANCER** | Malignant Oral Lesions (Duplicates consolidated) |
| `NON CANCER` | **NON CANCER** | Benign Mucosal Tissue / Normal Oral Cavity |
| `NON CANCER 2` | **NON CANCER** | Benign Mucosal Conditions (Duplicates consolidated) |

---

## 3. Split Distribution & Balance Summary

| Split | CANCER Count | NON CANCER Count | Split Total | % of Cleaned Dataset | Balance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **412** (54.71%) | **341** (45.29%) | **753** | 70.90% | **1.21 : 1** (Well Balanced) |
| **Validation** | **85** (58.22%) | **61** (41.78%) | **146** | 13.75% | **1.39 : 1** (Well Balanced) |
| **Test** | **94** (57.67%) | **69** (42.33%) | **163** | 15.35% | **1.36 : 1** (Well Balanced) |
| **TOTAL** | **591** (55.65%) | **471** (44.35%) | **1,062** | 100.00% | **1.25 : 1** (Optimal Balance) |

---

## 4. Verification & Leakage Elimination Audit

An exhaustive post-reconstruction hash scan on `cleaned_data/oral/` confirmed:

- **Cross-Split Exact Duplicates**: **0** (Train-Test: 0, Train-Val: 0, Val-Test: 0)
- **Within-Split Duplicates**: **0**
- **Conflicting Class Labels**: **0**
- **Corrupted / Unreadable Files**: **0** (All 1,062 images verified via PIL)
- **Classes**: Exactly **2 (`CANCER`, `NON CANCER`)**
- **Test Split Preservation**: All **163 unique images** from the original test split are preserved in `cleaned_data/oral/Test/`.

---

## 5. Generated Artifacts

- [outputs/dataset_cleaning/oral/original_statistics.csv](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/original_statistics.csv)
- [outputs/dataset_cleaning/oral/cleaned_statistics.csv](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/cleaned_statistics.csv)
- [outputs/dataset_cleaning/oral/removed_duplicates.csv](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/removed_duplicates.csv)
- [outputs/dataset_cleaning/oral/leakage_report.json](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/leakage_report.json)
- [outputs/dataset_cleaning/oral/image_quality_report.json](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/image_quality_report.json)
- [outputs/dataset_cleaning/oral/class_distribution.png](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/class_distribution.png)
- [outputs/dataset_cleaning/oral/cleaning_summary.md](file:///e:/Mega%20project/MedvisionAi-/ML-Model/outputs/dataset_cleaning/oral/cleaning_summary.md)
