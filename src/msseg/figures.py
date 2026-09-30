"""Figures for the report: qualitative panels and per-patient distributions.

Run with::

    python -m msseg.figures --checkpoint runs/multimodal/best.pt \
        --results results/metrics.json --output-dir docs/images

The qualitative panel is the figure that matters most. Metrics say how well the
model does; the panel says *how it fails*, which is what a reader needs in order
to judge whether the failures are clinically acceptable. Each row shows one
slice as FLAIR, ground truth, prediction, and an error overlay with false
positives in red and false negatives in blue.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or on a headless machine
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from msseg.data.dataset import AxialContextDataset, MultimodalMSDataset
from msseg.data.naming import patient_key, z_index
from msseg.evaluate import load_checkpoint
from msseg.postprocess import postprocess_batch

logger = logging.getLogger(__name__)

__all__ = ["plot_confusion_matrix", "plot_patient_distribution",
           "plot_dice_vs_lesion_load", "qualitative_panel"]

FP_COLOUR = (1.0, 0.0, 0.0)
FN_COLOUR = (0.0, 0.4, 1.0)


def plot_confusion_matrix(matrix: np.ndarray, out_path: Path) -> None:
    """Confusion matrix as proportions of all voxels."""
    proportions = matrix.astype(float) / matrix.sum()

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    image = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion matrix (proportion of voxels)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks([0, 1], ["No lesion", "Lesion"])
    ax.set_yticks([0, 1], ["No lesion", "Lesion"])

    for i in range(2):
        for j in range(2):
            value = proportions[i, j]
            ax.text(j, i, f"{value * 100:.3f}%", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=10)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_patient_distribution(per_patient: list[dict], out_path: Path) -> None:
    """Box plot of per-patient Dice and F1, with individual patients overlaid."""
    dice = np.array([p["dice"] for p in per_patient], dtype=float)
    recall = np.array([p["recall"] for p in per_patient], dtype=float)
    precision = np.array([p["precision"] for p in per_patient], dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.boxplot([dice, precision, recall], widths=0.55,
               tick_labels=["Dice", "Precision", "Recall"])

    rng = np.random.default_rng(0)
    for position, values in enumerate([dice, precision, recall], start=1):
        jitter = rng.normal(0, 0.035, size=values.size)
        ax.scatter(position + jitter, values, s=22, alpha=0.6, zorder=3)

    ax.set_ylabel("Voxel-level score within each patient")
    ax.set_title(f"Per-patient distribution on test (n={len(per_patient)})")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dice_vs_lesion_load(per_patient: list[dict], out_path: Path) -> None:
    """Dice against ground-truth lesion load, one labelled point per patient.

    The expected shape is a rising curve that saturates: patients with very few
    lesion voxels are penalised heavily by Dice, because a handful of false
    positives is large relative to a tiny ground truth.
    """
    load = np.array([p["gt_voxels"] for p in per_patient], dtype=float)
    dice = np.array([p["dice"] for p in per_patient], dtype=float)
    labels = [p["patient"].split("_")[-1] for p in per_patient]

    positive = load[load > 0]
    use_log = positive.size > 0 and positive.max() / positive.min() > 100

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(load, dice, s=55, alpha=0.8)
    for x, y, label in zip(load, dice, labels, strict=True):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Ground-truth lesion load (voxels)" + (" — log scale" if use_log else ""))
    ax.set_ylabel("Dice (voxel level, within patient)")
    ax.set_title("Dice against lesion load")
    if use_log:
        ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def qualitative_panel(
    model,
    cfg,
    specs: list[tuple[str, int | None]],
    threshold: float,
    min_component_size: int,
    device: torch.device,
    out_path: Path,
) -> None:
    """Render a FLAIR / GT / prediction / error panel for selected slices.

    Args:
        specs: ``(patient_key, z)`` pairs, e.g. ``("mslesseg_p68", 80)``.
            Pass ``z=None`` to pick that patient's slice with the largest
            ground-truth lesion automatically.
    """
    dataset_cls = {
        "multimodal": MultimodalMSDataset,
        "axial_context": AxialContextDataset,
    }[cfg.data.mode]
    dataset = dataset_cls(
        processed_root=cfg.data.processed_root, manifest=cfg.data.manifest,
        split="test", transform=None, return_name=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    selected: dict[tuple[str, int | None], dict] = {
        spec: {"gt_sum": -1.0, "image": None, "gt": None, "pred": None, "z": None}
        for spec in specs
    }

    model.eval()
    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device).float()
            masks = masks.to(device).float()
            predictions = postprocess_batch(
                torch.sigmoid(model(images)), threshold, min_component_size
            )

            for i, name in enumerate(names):
                key, z = patient_key(name), z_index(name)
                gt = masks[i, 0].cpu().numpy()

                for spec in specs:
                    spec_key, spec_z = spec
                    if spec_key != key:
                        continue
                    matches = (spec_z == z) if spec_z is not None else (
                        gt.sum() > selected[spec]["gt_sum"]
                    )
                    if matches:
                        selected[spec] = {
                            "gt_sum": float(gt.sum()),
                            "image": images[i, 0].cpu().numpy(),  # channel 0 = FLAIR
                            "gt": gt,
                            "pred": predictions[i, 0].cpu().numpy(),
                            "z": z,
                        }

    n_rows = len(specs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3.1 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, spec in enumerate(specs):
        item = selected[spec]
        if item["image"] is None:
            logger.warning("No slice found for %s", spec)
            for col in range(4):
                axes[row, col].axis("off")
            axes[row, 0].set_title(f"{spec[0]} z={spec[1]} — not found", fontsize=10)
            continue

        flair, gt, pred = item["image"], item["gt"], item["pred"]
        false_positive = (pred == 1) & (gt == 0)
        false_negative = (pred == 0) & (gt == 1)

        for col, (data, title) in enumerate(
            [(flair, f"{spec[0]} z={item['z']} — FLAIR"),
             (gt, "Ground truth"),
             (pred, "Prediction")]
        ):
            axes[row, col].imshow(data, cmap="gray")
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")

        axes[row, 3].imshow(flair, cmap="gray")
        overlay_fp = np.zeros((*false_positive.shape, 4))
        overlay_fp[..., :3] = FP_COLOUR
        overlay_fp[..., 3] = false_positive * 0.65
        overlay_fn = np.zeros((*false_negative.shape, 4))
        overlay_fn[..., :3] = FN_COLOUR
        overlay_fn[..., 3] = false_negative * 0.65
        axes[row, 3].imshow(overlay_fp)
        axes[row, 3].imshow(overlay_fn)
        axes[row, 3].set_title("Errors", fontsize=10)
        axes[row, 3].axis("off")

    fig.legend(
        handles=[
            mpatches.Patch(color=FP_COLOUR, label="False positive"),
            mpatches.Patch(color=FN_COLOUR, label="False negative"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument(
        "--panel", nargs="*", default=[],
        metavar="PATIENT:Z",
        help="Slices for the qualitative panel, e.g. mslesseg_p68:80 mslesseg_p75:auto",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(args.results.read_text(encoding="utf-8"))
    plot_confusion_matrix(
        np.array(report["confusion_matrix"]), args.output_dir / "confusion_matrix.png"
    )
    plot_patient_distribution(
        report["per_patient"], args.output_dir / "per_patient_distribution.png"
    )
    plot_dice_vs_lesion_load(
        report["per_patient"], args.output_dir / "dice_vs_lesion_load.png"
    )

    if args.panel:
        specs = []
        for item in args.panel:
            key, _, z = item.partition(":")
            specs.append((key, None if z in ("", "auto") else int(z)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, threshold = load_checkpoint(args.checkpoint, device)
        qualitative_panel(
            model, cfg, specs, threshold, args.min_component_size, device,
            args.output_dir / "qualitative_panel.png",
        )

    logger.info("Figures written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
