"""Evaluation entry point.

Run with::

    python -m msseg.evaluate --checkpoint runs/multimodal/best.pt --split test

Produces, in ``--output-dir``:

* ``metrics.json``    -- all three aggregation levels plus the confusion matrix
* ``per_patient.csv`` -- one row per patient, with a TOTAL row (mean +/- std)
* ``worst_slices.csv``-- the lowest-Dice slices, for error analysis

The decision threshold is read from the checkpoint, where training stored the
value it selected on the validation split. Passing ``--threshold`` overrides it,
which is useful for a sensitivity analysis but must not be used to pick a value
that flatters the test numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from msseg.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from msseg.data.dataset import AxialContextDataset, MultimodalMSDataset
from msseg.metrics import ConfusionAccumulator
from msseg.models.unet import UNet2D
from msseg.postprocess import postprocess_batch

logger = logging.getLogger(__name__)

__all__ = ["load_checkpoint", "evaluate", "main"]


def load_checkpoint(path: str | Path, device: torch.device):
    """Load a checkpoint and rebuild the model it describes.

    Returns:
        ``(model, config, threshold)``.
    """
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise KeyError(
            f"{path} has no embedded config. It was probably saved with "
            "torch.save(model.state_dict()) alone; rebuild the model manually."
        )

    payload = checkpoint["config"]
    cfg = ExperimentConfig(
        name=payload.get("name", "loaded"),
        data=DataConfig(**payload["data"]),
        model=ModelConfig(**payload["model"]),
        train=TrainConfig(**payload["train"]),
        eval=EvalConfig(**payload["eval"]),
    )

    model = UNet2D(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
        use_se=cfg.model.use_se,
        se_reduction=cfg.model.se_reduction,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5))
    logger.info(
        "Loaded %s (epoch %s, val score %.4f, threshold %.2f)",
        path, checkpoint.get("epoch", "?"), checkpoint.get("val_score", float("nan")),
        threshold,
    )
    return model, cfg, threshold


def evaluate(
    model,
    cfg: ExperimentConfig,
    split: str,
    threshold: float,
    min_component_size: int,
    device: torch.device,
) -> dict:
    """Run the model over one split and return the full metric report."""
    dataset_cls = {
        "multimodal": MultimodalMSDataset,
        "axial_context": AxialContextDataset,
    }[cfg.data.mode]

    dataset = dataset_cls(
        processed_root=cfg.data.processed_root,
        manifest=cfg.data.manifest,
        split=split,
        transform=None,
        return_name=True,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.eval.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    accumulator = ConfusionAccumulator()
    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device).float()
            masks = masks.to(device).float()
            probabilities = torch.sigmoid(model(images))
            predictions = postprocess_batch(probabilities, threshold, min_component_size)
            accumulator.update(predictions, masks, list(names))

    report = accumulator.report()
    report["settings"] = {
        "split": split,
        "threshold": threshold,
        "min_component_size": min_component_size,
        "mode": cfg.data.mode,
    }
    report["worst_slices"] = accumulator.worst_slices(20)
    return report


def _write_per_patient_csv(rows: list[dict], path: Path) -> None:
    """Write one row per patient plus a TOTAL row with mean and std across patients."""
    if not rows:
        return

    numeric = ["precision", "recall", "f1", "dice", "iou", "dice_slice_mean"]
    fieldnames = list(rows[0].keys())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        total = dict.fromkeys(fieldnames, "")
        total["patient"] = "TOTAL (mean +/- std over patients)"
        for key in numeric:
            values = np.array([r[key] for r in rows], dtype=float)
            std = values.std(ddof=1) if values.size > 1 else 0.0
            total[key] = f"{values.mean():.4f} +/- {std:.4f}"
        total["n_slices"] = sum(r["n_slices"] for r in rows)
        total["gt_voxels"] = sum(r["gt_voxels"] for r in rows)
        total["pred_voxels"] = sum(r["pred_voxels"] for r in rows)
        writer.writerow(total)


def _print_summary(report: dict) -> None:
    voxel, slice_, patient = report["voxel"], report["slice"], report["patient"]
    settings = report["settings"]

    print(f"\n=== {settings['split'].upper()} "
          f"(threshold {settings['threshold']:.2f}, min_size {settings['min_component_size']}) ===")
    print(f"slices: {slice_['n_slices']}  with lesion: {slice_['n_slices_with_lesion']}  "
          f"patients: {patient['n_patients']}")
    print("\n-- voxel level (pooled over all voxels) --")
    print(f"  precision {voxel['precision']:.4f}   recall {voxel['recall']:.4f}   "
          f"specificity {voxel['specificity']:.4f}")
    print(f"  Dice = F1 {voxel['dice']:.4f}   IoU {voxel['iou']:.4f}")
    print("\n-- patient level (each patient weighted equally) --")
    print(f"  Dice {patient['dice_mean']:.4f} +/- {patient['dice_std']:.4f}")
    print("\n-- slice level --")
    print(f"  mean Dice, all slices      {slice_['dice_mean']:.4f}")
    print(f"  mean Dice, lesion slices   {slice_['dice_mean_positive_only']:.4f}")
    print("\nconfusion matrix [[TN, FP], [FN, TP]]:")
    print(report["confusion_matrix"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the threshold stored in the checkpoint.")
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, stored_threshold = load_checkpoint(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else stored_threshold

    if args.threshold is not None:
        logger.warning(
            "Threshold overridden on the command line (%.2f instead of %.2f). "
            "Do not tune this on the test split.", args.threshold, stored_threshold,
        )

    report = evaluate(
        model, cfg, args.split, threshold, args.min_component_size, device
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serialisable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in report.items()
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )
    _write_per_patient_csv(report["per_patient"], args.output_dir / "per_patient.csv")

    with (args.output_dir / "worst_slices.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dice", "slice", "false_positives", "false_negatives"])
        writer.writerows(report["worst_slices"])

    _print_summary(report)
    print(f"\nWritten to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
