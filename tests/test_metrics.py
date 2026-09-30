"""Metrics, including the properties that make the three aggregation levels differ."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msseg.metrics import ConfusionAccumulator, Counts, metrics_from_counts


def test_perfect_prediction_scores_one():
    # The epsilon in the denominators keeps the scores just under 1.0.
    m = metrics_from_counts(Counts(tp=100, fp=0, fn=0, tn=900))
    assert m["dice"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["iou"] == pytest.approx(1.0)


def test_dice_equals_f1_for_binary_masks():
    """Not a coincidence: 2TP/(2TP+FP+FN) is algebraically the harmonic mean."""
    for tp, fp, fn, tn in [(50, 20, 30, 900), (1, 99, 1, 10), (7, 3, 11, 42)]:
        m = metrics_from_counts(Counts(tp=tp, fp=fp, fn=fn, tn=tn))
        assert np.isclose(m["dice"], m["f1"], atol=1e-6)


def test_accuracy_is_useless_under_extreme_imbalance():
    """Predicting all background: 99.9% accuracy, zero Dice."""
    m = metrics_from_counts(Counts(tp=0, fp=0, fn=100, tn=99_900))
    assert m["accuracy"] == pytest.approx(0.999, abs=1e-4)
    assert m["dice"] < 1e-6


def _batch(pred: np.ndarray, gt: np.ndarray):
    return (
        torch.tensor(pred, dtype=torch.float32)[None, None],
        torch.tensor(gt, dtype=torch.float32)[None, None],
    )


def test_accumulator_separates_aggregation_levels():
    """A high-load patient and a low-load one must move voxel and patient Dice differently."""
    acc = ConfusionAccumulator()

    # Patient A: large lesion, segmented well.
    big_gt = np.zeros((16, 16))
    big_gt[2:14, 2:14] = 1
    big_pred = np.zeros((16, 16))
    big_pred[2:14, 2:13] = 1
    p, g = _batch(big_pred, big_gt)
    acc.update(p, g, ["src_FLAIR_slice_p01_z001.npy"])

    # Patient B: tiny lesion, missed entirely.
    small_gt = np.zeros((16, 16))
    small_gt[5:7, 5:7] = 1
    small_pred = np.zeros((16, 16))
    p, g = _batch(small_pred, small_gt)
    acc.update(p, g, ["src_FLAIR_slice_p02_z001.npy"])

    report = acc.report()

    assert report["patient"]["n_patients"] == 2
    # Voxel level is dominated by the large, well-segmented lesion.
    assert report["voxel"]["dice"] > 0.9
    # Patient level weights both equally, so the missed patient halves it.
    assert report["patient"]["dice_mean"] < 0.6


def test_empty_prediction_on_empty_mask_scores_one_per_slice():
    """Convention, and the reason the all-slice mean Dice is inflated."""
    acc = ConfusionAccumulator()
    empty = np.zeros((8, 8))
    p, g = _batch(empty, empty)
    acc.update(p, g, ["src_FLAIR_slice_p01_z000.npy"])

    report = acc.report()
    assert np.isclose(report["slice"]["dice_mean"], 1.0)
    assert report["slice"]["n_slices_with_lesion"] == 0


def test_confusion_matrix_layout():
    acc = ConfusionAccumulator()
    gt = np.zeros((4, 4))
    gt[0, 0] = 1
    pred = np.zeros((4, 4))
    pred[0, 1] = 1
    p, g = _batch(pred, gt)
    acc.update(p, g, ["src_FLAIR_slice_p01_z000.npy"])

    matrix = acc.report()["confusion_matrix"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 14  # TN
    assert matrix[0, 1] == 1   # FP
    assert matrix[1, 0] == 1   # FN
    assert matrix[1, 1] == 0   # TP
