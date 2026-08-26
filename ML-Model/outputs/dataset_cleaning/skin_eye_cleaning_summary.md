# MedvisionAI Overall Dataset Cleaning & Reconstruction Summary

## Multi-Domain Dataset Reconstruction Overview

| Dataset | Original Images | Removed Duplicates | Final Images | Train-Test Leakage After Cleaning | Classes | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oral Cancer** | 1,651 | 589 (587 dups + 1 conflict pair) | **1,062** | **0 (0.0%)** | **2** (`CANCER`, `NON CANCER`) | **1.25 : 1** |
| **Eye Disease** | 10,438 | 161 multi-condition duplicates | **10,277** | **0 (0.0%)** | **7** (`A`, `C`, `D`, `G`, `H`, `M`, `N`) | **7.53 : 1** |
| **Skin Disease**| 22,719 | 0 (Clean raw partition) | **22,719** | **0 (0.0%)** | **8** (ISIC / HAM10000 / PAD-UFES) | **25.68 : 1** |
| **TOTAL** | **34,808** | **750** | **34,058** | **0.0% Across All Domains** | **17 Total Classes** | — |

---

## Key Reconstructions Accomplished:

1. **Oral Domain**:
   - Consolidated 4 artificial subclasses into 2 binary clinical classes.
   - Eliminated 588 exact duplicate groups and all 108 Train-Test leakage groups.
   - Model accuracy surged from 48.82% to **87.12%** (+38.30% absolute improvement) upon retraining.
2. **Eye Domain**:
   - Eliminated 161 multi-condition duplicate groups where identical patient fundus photos had been placed into multiple disease folders in raw ODIR data.
   - Completely resolved 26 Train-Test leaks, 28 Train-Val leaks, 4 Val-Test leaks, and 103 within-train duplicate conflicts.
   - All 1,051 test images preserved in `cleaned_data/eye/Test/`.
3. **Skin Domain**:
   - Cryptographically audited all 22,719 images across Train, Validation, and Test.
   - Confirmed 0 duplicate collisions and 0 cross-split leakage.
   - Reconstructed pristine structure in `cleaned_data/skin/`.
