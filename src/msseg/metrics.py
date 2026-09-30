"""Segmentation metrics, with an explicit distinction between aggregation levels.

Reporting a single "Dice" number for a segmentation model is ambiguous, because
three different quantities are commonly called by that name:

1. **Voxel-level (micro) Dice** -- pool TP/FP/FN over every voxel of every slice
   of every patient, then compute ``2TP / (2TP + FP + FN)``. Dominated by
   patients with a high lesion load. *Identical to the voxel-level F1 score*:
   the two formulas are algebraically the same for binary masks.
2. **Per-slice mean Dice** -- compute Dice on each slice, then average. Slices
   with no lesion and no prediction score 1.0 by convention, which inflates the
   mean when most slices are negative. Report it only alongside the number of
   positive slices, or restrict it to positive slices.
3. **Per-patient mean Dice** -- compute a voxel-level Dice within each patient,
   then average over patients. Every patient counts equally. This is the most
   clinically meaningful of the three and what most MS lesion papers report.

:class:`ConfusionAccumulator` computes all three, so results are never quoted
without saying which one they are.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["Counts", "metrics_from_counts", "ConfusionAccumulator"]

_EPS = 1e-8


@dataclass
class Counts:
    """Voxel-level confusion matrix entries."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def as_matrix(self) -> np.ndarray:
        """2x2 confusion matrix laid out as ``[[TN, FP], [FN, TP]]``."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)


def metrics_from_counts(c: Counts) -> dict[str, float]:
    """Derive the standard metric set from voxel counts.

    Returns a dict with ``accuracy``, ``precision``, ``recall``, ``specificity``,
    ``f1``, ``dice`` and ``iou``. Note that ``dice == f1`` by construction; both
    are returned because both names appear in the literature.
    """
    tp, fp, fn, tn = c.tp, c.fp, c.fn, c.tn

    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    specificity = tn / (tn + fp + _EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + _EPS)
    f1 = 2 * precision * recall / (precision + recall + _EPS)
    dice = 2 * tp / (2 * tp + fp + fn + _EPS)
    iou = tp / (tp + fp + fn + _EPS)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
    }


@dataclass
class _PatientState:
    counts: Counts = field(default_factory=Counts)
    dice_slice_sum: float = 0.0
    dice_slice_pos_sum: float = 0.0
    n_slices: int = 0
    n_slices_pos: int = 0
    gt_voxels: int = 0
    pred_voxels: int = 0


class ConfusionAccumulator:
    """Accumulates predictions over a whole evaluation run.

    Usage::

        acc = ConfusionAccumulator()
        for imgs, masks, names in loader:
            preds = postprocess(model(imgs))
            acc.update(preds, masks, names)
        report = acc.report()

    The accumulator never holds the images, only counts, so memory stays flat
    regardless of dataset size.
    """

    def __init__(self) -> None:
        self._global = Counts()
        self._per_patient: dict[str, _PatientState] = defaultdict(_PatientState)
        self._dice_slice_sum = 0.0
        self._dice_slice_pos_sum = 0.0
        self._n_slices = 0
        self._n_slices_pos = 0
        self._worst: list[tuple[float, str, int, int]] = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        names: list[str],
        patient_key_fn=None,
    ) -> None:
        """Add one batch.

        Args:
            preds: Binary predictions ``(B, 1, H, W)`` in ``{0, 1}``.
            targets: Binary ground truth, same shape.
            names: One filename per batch element, used to group by patient.
            patient_key_fn: Callable mapping a filename to a patient key.
                Defaults to :func:`msseg.data.naming.patient_key`.
        """
        if patient_key_fn is None:
            from msseg.data.naming import patient_key as patient_key_fn

        b = preds.shape[0]
        p = preds.reshape(b, -1).to(torch.bool)
        t = targets.reshape(b, -1).to(torch.bool)

        tp = (p & t).sum(dim=1)
        fp = (p & ~t).sum(dim=1)
        fn = (~p & t).sum(dim=1)
        tn = (~p & ~t).sum(dim=1)

        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * tp.float() + _EPS) / (union.float() + _EPS)

        for i in range(b):
            tp_i, fp_i, fn_i, tn_i = (
                int(tp[i]),
                int(fp[i]),
                int(fn[i]),
                int(tn[i]),
            )
            dice_i = float(dice[i])
            has_lesion = bool(t[i].any())

            self._global.update(tp_i, fp_i, fn_i, tn_i)
            self._dice_slice_sum += dice_i
            self._n_slices += 1
            if has_lesion:
                self._dice_slice_pos_sum += dice_i
                self._n_slices_pos += 1

            state = self._per_patient[patient_key_fn(names[i])]
            state.counts.update(tp_i, fp_i, fn_i, tn_i)
            state.dice_slice_sum += dice_i
            state.n_slices += 1
            state.gt_voxels += int(t[i].sum())
            state.pred_voxels += int(p[i].sum())
            if has_lesion:
                state.dice_slice_pos_sum += dice_i
                state.n_slices_pos += 1

            self._worst.append((dice_i, names[i], fp_i, fn_i))

    def per_patient(self) -> list[dict]:
        """One row per patient with voxel-level metrics inside that patient."""
        rows = []
        for key, st in self._per_patient.items():
            m = metrics_from_counts(st.counts)
            rows.append(
                {
                    "patient": key,
                    "n_slices": st.n_slices,
                    "n_slices_with_lesion": st.n_slices_pos,
                    "gt_voxels": st.gt_voxels,
                    "pred_voxels": st.pred_voxels,
                    **{k: m[k] for k in ("precision", "recall", "f1", "dice", "iou")},
                    "dice_slice_mean": st.dice_slice_sum / max(st.n_slices, 1),
                    "dice_slice_mean_pos": st.dice_slice_pos_sum / max(st.n_slices_pos, 1),
                }
            )
        rows.sort(key=lambda r: r["patient"])
        return rows

    def worst_slices(self, k: int = 20) -> list[tuple[float, str, int, int]]:
        """The ``k`` slices with the lowest Dice, as ``(dice, name, fp, fn)``."""
        return sorted(self._worst, key=lambda x: x[0])[:k]

    def report(self) -> dict:
        """Full result bundle with all three aggregation levels."""
        rows = self.per_patient()
        patient_dice = np.array([r["dice"] for r in rows], dtype=float)
        patient_f1 = np.array([r["f1"] for r in rows], dtype=float)

        return {
            "counts": {
                "tp": self._global.tp,
                "fp": self._global.fp,
                "fn": self._global.fn,
                "tn": self._global.tn,
            },
            "confusion_matrix": self._global.as_matrix(),
            # Level 1: pooled over every voxel.
            "voxel": metrics_from_counts(self._global),
            # Level 2: averaged over slices.
            "slice": {
                "dice_mean": self._dice_slice_sum / max(self._n_slices, 1),
                "dice_mean_positive_only": self._dice_slice_pos_sum
                / max(self._n_slices_pos, 1),
                "n_slices": self._n_slices,
                "n_slices_with_lesion": self._n_slices_pos,
            },
            # Level 3: averaged over patients, each patient weighted equally.
            "patient": {
                "dice_mean": float(patient_dice.mean()) if patient_dice.size else 0.0,
                "dice_std": float(patient_dice.std(ddof=1)) if patient_dice.size > 1 else 0.0,
                "f1_mean": float(patient_f1.mean()) if patient_f1.size else 0.0,
                "f1_std": float(patient_f1.std(ddof=1)) if patient_f1.size > 1 else 0.0,
                "n_patients": len(rows),
            },
            "per_patient": rows,
        }
