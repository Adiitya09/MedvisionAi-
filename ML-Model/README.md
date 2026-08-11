<<<<<<< HEAD
# MedvisionAi
**MedVision AI** is a deep learning-based medical image classification system that detects skin, oral, and eye diseases from uploaded images. Using transfer learning with MobileNetV2 and ResNet50, it provides accurate predictions, confidence scores, and AI-assisted disease screening through an intuitive web interface.
=======
# AI-Powered Multi-Disease Detection

Transfer-learning pipeline for classifying medical images across three domains:

- **Skin** — FYP skin disease dataset
- **Eye** — Eye disease dataset
- **Oral** — Oral cancer dataset

## Project Structure

```
MP/
├── config.py              # Central Python configuration
├── config/
│   └── config.yaml        # Extended YAML settings (research pipeline)
├── scripts/               # CLI entry points
│   ├── train.py
│   ├── compare_transfer_models.py
│   ├── hyperparameter_tuning.py
│   ├── optimize_model.py
│   ├── explainability.py
│   ├── mlflow_tracking.py
│   └── split_medical_dataset.py
├── utils/                 # Shared training utilities
├── data/                  # Dataset root (not committed — see data/README.md)
├── models/                # Saved Keras models
├── checkpoints/           # Best-validation checkpoints
├── logs/                  # Training logs & TensorBoard
├── outputs/               # Plots, CSV exports, MLflow runs
└── saved_models/          # Exported SavedModel / TFLite artifacts
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place your dataset under `data/` following the layout described in [data/README.md](data/README.md).

## Usage

Train a single model:

```bash
python scripts/train.py --dataset skin --model efficientnetb0
```

Compare transfer-learning backbones:

```bash
python scripts/compare_transfer_models.py
```

Split a raw dataset into train / validation / test:

```bash
python scripts/split_medical_dataset.py
```

## Configuration

Runtime settings live in `config.py`. Override defaults with environment variables:

| Variable | Description | Default |
|---|---|---|
| `MEDICAL_AI_BACKBONE` | Model backbone | `efficientnetb0` |
| `MEDICAL_AI_DATASET_ROOT` | Dataset folder name | `data` |

## Supported Backbones

- MobileNetV2
- EfficientNetB0
- ResNet50
>>>>>>> 09426dc (Organize project structure for GitHub with scripts, config, and gitignore.)
