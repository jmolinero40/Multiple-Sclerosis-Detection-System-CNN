"""Experiment configuration loaded from YAML.

Every hyperparameter lives in a config file under ``configs/``, never in the
source. Two reasons: a run is reproducible from the config alone, and the exact
config of a finished run can be saved next to its checkpoint, so months later
there is no guesswork about which settings produced which number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DataConfig", "ModelConfig", "TrainConfig", "EvalConfig", "ExperimentConfig"]


def _filter_known(cls, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys that are actual fields of ``cls``, and report the rest."""
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return {k: v for k, v in payload.items() if k in known}


@dataclass
class DataConfig:
    processed_root: str = "data/processed"
    manifest: str = "data/manifests/splits.csv"
    mode: str = "multimodal"  # "multimodal" | "axial_context"
    num_workers: int = 0


@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 32
    use_se: bool = True
    se_reduction: int = 16


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        selection_metric: What decides the best epoch.

            ``"f1_swept"``
                Voxel-level F-beta computed at the best threshold found on the
                validation split that epoch. Default: it tracks the quality of
                the final, thresholded segmentation, which is what is actually
                reported.
            ``"soft_dice"``
                Mean soft Dice on validation, computed on probabilities without
                thresholding. This is what the original thesis used. Kept so
                that the thesis run can be reproduced.

        fixed_threshold: When set, skips the validation threshold sweep and
            stores this value in the checkpoint instead. Use it to reproduce a
            run that used a hardcoded decision threshold.
    """

    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    bce_weight: float = 0.7
    pos_weight: float = 3.0
    grad_clip_norm: float = 1.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 8
    seed: int = 42
    amp: bool = True
    output_dir: str = "runs/multimodal"
    threshold_grid_beta: float = 1.0
    selection_metric: str = "f1_swept"
    fixed_threshold: float | None = None


@dataclass
class EvalConfig:
    threshold: float | None = None  # None -> read from the checkpoint metadata
    min_component_size: int = 10
    batch_size: int = 8
    output_dir: str = "results"


@dataclass
class ExperimentConfig:
    name: str = "multimodal"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config, validating that no key is silently ignored."""
        with Path(path).open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}

        return cls(
            name=payload.get("name", "experiment"),
            data=DataConfig(**_filter_known(DataConfig, payload.get("data", {}))),
            model=ModelConfig(**_filter_known(ModelConfig, payload.get("model", {}))),
            train=TrainConfig(**_filter_known(TrainConfig, payload.get("train", {}))),
            eval=EvalConfig(**_filter_known(EvalConfig, payload.get("eval", {}))),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)
