"""Training entry point.

Run with::

    python -m msseg.train --config configs/multimodal.yaml

What the loop does beyond the obvious:

* **Selects on validation F-beta at the best threshold**, not on the loss and
  not on soft Dice. The loss is dominated by the background and moves very
  little once the model stops predicting "all zeros"; a threshold-aware metric
  is what actually tracks the quality of the final segmentation.
* **Sweeps the decision threshold on validation every epoch** and stores the
  best one in the checkpoint, so evaluation never has to guess it and never
  tunes it on test.
* **Saves a single best checkpoint** with the config, the metrics and the
  threshold embedded, so the artefact is self-describing.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from msseg.config import ExperimentConfig
from msseg.data.dataset import AxialContextDataset, MultimodalMSDataset, train_augment
from msseg.losses import BCEDiceLoss, soft_dice
from msseg.models.unet import UNet2D
from msseg.postprocess import sweep_threshold

logger = logging.getLogger(__name__)

__all__ = ["set_seed", "build_datasets", "train", "main"]


def set_seed(seed: int) -> None:
    """Seed every RNG that affects the run.

    Note:
        This does not make CUDA fully deterministic. ``torch.backends.cudnn``
        picks non-deterministic algorithms for some convolutions. Setting
        ``cudnn.deterministic = True`` fixes that at a noticeable speed cost;
        it is left off here, so expect small run-to-run variation on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(cfg: ExperimentConfig):
    """Instantiate the train and validation datasets for the configured mode."""
    dataset_cls = {
        "multimodal": MultimodalMSDataset,
        "axial_context": AxialContextDataset,
    }.get(cfg.data.mode)
    if dataset_cls is None:
        raise ValueError(f"Unknown data.mode: {cfg.data.mode!r}")

    common = {"processed_root": cfg.data.processed_root, "manifest": cfg.data.manifest}
    train_ds = dataset_cls(**common, split="train", transform=train_augment)
    val_ds = dataset_cls(**common, split="val", transform=None)
    return train_ds, val_ds


def _run_validation(model, loader, criterion, device) -> tuple[float, float, list, list]:
    """Return ``(mean_loss, mean_soft_dice, probabilities, targets)``."""
    model.eval()
    loss_sum, dice_sum, n = 0.0, 0.0, 0
    probabilities, targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()

            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)

            batch = images.size(0)
            loss_sum += loss.item() * batch
            dice_sum += soft_dice(probs, masks).item() * batch
            n += batch

            probabilities.append(probs.cpu())
            targets.append(masks.cpu())

    return loss_sum / max(n, 1), dice_sum / max(n, 1), probabilities, targets


def train(cfg: ExperimentConfig) -> Path:
    """Run training and return the path of the best checkpoint."""
    if cfg.train.selection_metric not in ("f1_swept", "soft_dice"):
        raise ValueError(
            f"Unknown selection_metric: {cfg.train.selection_metric!r}. "
            "Expected 'f1_swept' or 'soft_dice'."
        )
    set_seed(cfg.train.seed)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(output_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = UNet2D(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
        use_se=cfg.model.use_se,
        se_reduction=cfg.model.se_reduction,
    ).to(device)
    logger.info("Model: UNet2D, %.2fM trainable parameters", model.num_parameters / 1e6)
    logger.info(
        "Selecting the best epoch on: %s%s",
        cfg.train.selection_metric,
        ""
        if cfg.train.fixed_threshold is None
        else f"; threshold fixed at {cfg.train.fixed_threshold}",
    )

    criterion = BCEDiceLoss(
        bce_weight=cfg.train.bce_weight,
        pos_weight=torch.tensor([cfg.train.pos_weight], device=device),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg.train.scheduler_factor, patience=cfg.train.scheduler_patience,
    )
    # Mixed precision roughly halves memory and speeds up training on any
    # recent NVIDIA card. It is a no-op on CPU, where autocast to fp16 would
    # be slower rather than faster.
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_epoch, best_threshold = -1.0, -1, 0.5
    epochs_without_improvement = 0
    history: list[dict] = []
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running_loss, n = 0.0, 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), masks)

            scaler.scale(loss).backward()
            if cfg.train.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            n += images.size(0)

        train_loss = running_loss / max(n, 1)
        val_loss, val_soft_dice, probabilities, targets = _run_validation(
            model, val_loader, criterion, device
        )
        if cfg.train.fixed_threshold is not None:
            # Reproduction mode: the threshold was hardcoded, not tuned.
            threshold = float(cfg.train.fixed_threshold)
            _, swept_score = sweep_threshold(
                probabilities, targets, beta=cfg.train.threshold_grid_beta
            )
        else:
            threshold, swept_score = sweep_threshold(
                probabilities, targets, beta=cfg.train.threshold_grid_beta
            )

        score = val_soft_dice if cfg.train.selection_metric == "soft_dice" else swept_score
        scheduler.step(score)

        logger.info(
            "epoch %02d  train_loss=%.4f  val_loss=%.4f  val_soft_dice=%.4f  "
            "val_f%.1f=%.4f @ thr=%.2f  lr=%.2e",
            epoch, train_loss, val_loss, val_soft_dice,
            cfg.train.threshold_grid_beta, swept_score, threshold,
            optimizer.param_groups[0]["lr"],
        )
        history.append(
            {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_soft_dice": val_soft_dice, "val_swept_f": swept_score,
                "selection_score": score, "threshold": threshold,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if score > best_score:
            best_score, best_epoch, best_threshold = score, epoch, threshold
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "val_score": score,
                    "threshold": threshold,
                    "val_soft_dice": val_soft_dice,
                },
                checkpoint_path,
            )
            logger.info("  new best -> %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
                logger.info(
                    "Early stopping: no improvement for %d epochs",
                    cfg.train.early_stopping_patience,
                )
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    logger.info(
        "Finished in %.1f min. Best epoch %d, val score %.4f at threshold %.2f.",
        elapsed / 60, best_epoch, best_score, best_threshold,
    )
    return checkpoint_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the MS lesion segmentation U-Net.")
    parser.add_argument("--config", type=Path, default=Path("configs/multimodal.yaml"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    train(ExperimentConfig.from_yaml(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
