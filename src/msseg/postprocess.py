"""Post-processing applied to the binarised network output.

Two operations, in this order:

1. **Thresholding.** The decision threshold is a free parameter and must be
   chosen on the *validation* split, never on test. Choosing it on test is a
   subtle but real form of leakage that can move Dice by several points.
2. **Small-component removal.** The network produces isolated single-pixel
   responses, mostly at the grey/white matter boundary and near the skull. A
   lesion smaller than about 10 pixels at this resolution is below what a
   radiologist would annotate, so components under ``min_size`` are dropped.
   This trades a small amount of recall for a larger gain in precision.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label

__all__ = ["remove_small_components", "postprocess_batch", "sweep_threshold"]

# 8-connectivity: diagonally touching pixels belong to the same lesion.
_STRUCTURE = np.ones((3, 3), dtype=np.int32)


def remove_small_components(mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` pixels.

    Args:
        mask: 2D binary array ``(H, W)``.
        min_size: Minimum component size in pixels. ``0`` disables the filter.

    Returns:
        ``uint8`` array of the same shape.
    """
    if min_size <= 0:
        return mask.astype(np.uint8)

    binary = mask.astype(bool)
    labelled, n_components = label(binary, structure=_STRUCTURE)
    if n_components == 0:
        return binary.astype(np.uint8)

    sizes = np.bincount(labelled.ravel())
    too_small = sizes < min_size
    too_small[0] = False  # index 0 is the background label

    cleaned = binary.copy()
    cleaned[too_small[labelled]] = False
    return cleaned.astype(np.uint8)


def postprocess_batch(
    probabilities: torch.Tensor, threshold: float, min_size: int = 10
) -> torch.Tensor:
    """Threshold a batch of probability maps and clean each one.

    Args:
        probabilities: ``(B, 1, H, W)`` tensor in ``[0, 1]``.
        threshold: Decision threshold, tuned on validation.
        min_size: Minimum connected-component size in pixels.

    Returns:
        ``(B, 1, H, W)`` float tensor in ``{0.0, 1.0}`` on the input device.
    """
    binary = (probabilities > threshold).detach().cpu().numpy().astype(np.uint8)
    for i in range(binary.shape[0]):
        binary[i, 0] = remove_small_components(binary[i, 0], min_size=min_size)
    return torch.from_numpy(binary).to(device=probabilities.device, dtype=torch.float32)


def sweep_threshold(
    probabilities: list[torch.Tensor],
    targets: list[torch.Tensor],
    grid: np.ndarray | None = None,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Pick the threshold maximising the voxel-level F-beta score.

    Args:
        probabilities: Per-batch probability tensors from the validation split.
        targets: Matching ground-truth tensors.
        grid: Candidate thresholds. Defaults to ``0.05 .. 0.95`` in steps of 0.05.
        beta: ``beta < 1`` favours precision (fewer false positives), ``beta > 1``
            favours recall. ``beta = 1`` is the ordinary F1.

    Returns:
        ``(best_threshold, best_score)``.
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)

    eps = 1e-8
    beta_squared = beta ** 2
    best_threshold, best_score = float(grid[0]), -1.0

    flat_probs = torch.cat([p.reshape(-1) for p in probabilities])
    flat_targets = torch.cat([t.reshape(-1) for t in targets]).to(torch.bool)

    for threshold in grid:
        predicted = flat_probs > float(threshold)
        tp = int((predicted & flat_targets).sum())
        fp = int((predicted & ~flat_targets).sum())
        fn = int((~predicted & flat_targets).sum())

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        score = (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall + eps)
        )

        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)

    return best_threshold, best_score
