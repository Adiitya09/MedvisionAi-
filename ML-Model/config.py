"""Central configuration for the multi-disease image-classification project.

The module deliberately contains no TensorFlow imports, making it safe to use
from training, evaluation, inference, and deployment code.  Set environment
variables such as ``MEDICAL_AI_BACKBONE`` or ``MEDICAL_AI_DATASET_ROOT`` to
override defaults without modifying source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BackboneName = Literal["mobilenetv2", "efficientnetb0", "resnet50"]
DatasetName = Literal["skin", "eye", "oral"]


def _environment_value(name: str, default: str) -> str:
    """Return a non-empty environment override or a documented default."""
    return os.getenv(name, default).strip() or default


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    """Directory mapping for one disease domain across all data splits."""

    name: DatasetName
    directory_name: str
    model_file_name: str


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Immutable settings shared by all project entry points."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    dataset_root_name: str = field(
        default_factory=lambda: _environment_value("MEDICAL_AI_DATASET_ROOT", "data")
    )
    backbone: BackboneName = field(
        default_factory=lambda: _environment_value("MEDICAL_AI_BACKBONE", "efficientnetb0")  # type: ignore[arg-type]
    )
    image_height: int = 224
    image_width: int = 224
    image_channels: int = 3
    batch_size: int = 32
    epochs: int = 50
    initial_learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5
    fine_tune_epochs: int = 15
    dropout_rate: float = 0.30
    random_seed: int = 42
    use_mixed_precision: bool = True
    cache_datasets: bool = True
    shuffle_buffer_size: int = 1_024
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    verbose: int = 1

    datasets: tuple[DatasetPaths, ...] = (
        DatasetPaths("skin", "FYP skin disease Dataset", "skin_model.keras"),
        DatasetPaths("eye", "Eye disease", "eye_model.keras"),
        DatasetPaths("oral", "Oral Cancer", "oral_model.keras"),
    )

    def __post_init__(self) -> None:
        """Reject an invalid backbone override before a training run starts."""
        valid_backbones = {"mobilenetv2", "efficientnetb0", "resnet50"}
        if self.backbone not in valid_backbones:
            valid = ", ".join(sorted(valid_backbones))
            raise ValueError(f"Unsupported MEDICAL_AI_BACKBONE '{self.backbone}'. Use: {valid}.")

    @property
    def image_size(self) -> tuple[int, int]:
        """Return Keras-compatible ``(height, width)`` dimensions."""
        return (self.image_height, self.image_width)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return the model input tensor shape."""
        return (self.image_height, self.image_width, self.image_channels)

    @property
    def dataset_root(self) -> Path:
        """Return the root containing Train, Validation, and test folders."""
        return self.project_root / self.dataset_root_name

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def utils_dir(self) -> Path:
        return self.project_root / "utils"

    @property
    def saved_models_dir(self) -> Path:
        return self.project_root / "saved_models"

    def get_dataset(self, dataset_name: str) -> DatasetPaths:
        """Return one configured domain, rejecting invalid names early."""
        normalized_name = dataset_name.strip().lower()
        for dataset in self.datasets:
            if dataset.name == normalized_name:
                return dataset
        supported = ", ".join(dataset.name for dataset in self.datasets)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose one of: {supported}.")

    def split_dir(self, dataset_name: str, split_name: str) -> Path:
        """Return a dataset split directory using the supplied project layout."""
        split_aliases = {
            "train": "Train",
            "validation": "Validation",
            "test": "Test",
        }
        try:
            directory = split_aliases[split_name.strip().lower()]
        except KeyError as error:
            raise ValueError("Split must be train, validation, or test.") from error
        return self.dataset_root / directory / self.get_dataset(dataset_name).directory_name

    def model_path_for(self, dataset_name: str) -> Path:
        """Return the required final Keras model location for a domain."""
        return self.models_dir / self.get_dataset(dataset_name).model_file_name

    def checkpoint_path_for(self, dataset_name: str) -> Path:
        """Return the best-validation checkpoint path for a domain."""
        return self.checkpoints_dir / f"{dataset_name}_best.keras"

    def class_names_path_for(self, dataset_name: str) -> Path:
        """Return the JSON label mapping location for a domain."""
        return self.models_dir / f"{dataset_name}_class_names.json"

    def create_project_directories(self) -> None:
        """Create all generated-artifact directories required by the project."""
        for directory in (
            self.models_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.outputs_dir,
            self.utils_dir,
            self.saved_models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def validate_dataset(self, dataset_name: str) -> None:
        """Validate that all expected split directories exist for one domain."""
        missing = [
            str(self.split_dir(dataset_name, split))
            for split in ("train", "validation", "test")
            if not self.split_dir(dataset_name, split).is_dir()
        ]
        if missing:
            raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))


CONFIG = ProjectConfig()
CONFIG.create_project_directories()
