"""Equivalence with the original thesis implementation.

These tests re-implement the original formulations inline and assert that the
refactored code computes the same numbers. They are the reason it is safe to
claim that this repository reproduces the thesis: the claim is checked on every
commit rather than asserted in prose.

Where the default configuration deliberately departs from the original, the test
pins the *reproduction* path (``legacy_minmax``, ``selection_metric:
soft_dice``) and separately asserts that the default differs, so that a change
to either one is caught.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msseg.data.preprocess import PreprocessConfig, normalise_volume
from msseg.losses import BCEDiceLoss, soft_dice
from msseg.models.unet import UNet2D

# --------------------------------------------------------------------------
# The original implementations, transcribed verbatim from the thesis scripts.
# --------------------------------------------------------------------------

def _original_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """From ``train_unet_simple.py::dice_coefficient``."""
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean()


def _original_bce_dice(logits, target, bce_weight=0.7, pos_weight=None):
    """From ``train_unet_simple.py::BCEDiceLoss.forward``."""
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss = bce(logits, target)
    dice_loss = 1.0 - _original_dice(torch.sigmoid(logits), target)
    return bce_weight * bce_loss + (1.0 - bce_weight) * dice_loss


def _original_minmax(volume: np.ndarray) -> np.ndarray:
    """From ``nature_sanity_checks.py``: whole-volume min-max, no clipping."""
    lo, hi = volume.min(), volume.max()
    return ((volume - lo) / (hi - lo + 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------

def test_soft_dice_matches_the_original():
    torch.manual_seed(0)
    probs = torch.rand(4, 1, 24, 24)
    target = (torch.rand(4, 1, 24, 24) > 0.97).float()
    assert torch.allclose(
        soft_dice(probs, target), _original_dice(probs, target), atol=1e-9
    )


def test_bce_dice_loss_matches_the_original():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 24, 24)
    target = (torch.rand(4, 1, 24, 24) > 0.97).float()
    pos_weight = torch.tensor([3.0])

    mine = BCEDiceLoss(bce_weight=0.7, pos_weight=pos_weight)(logits, target)
    original = _original_bce_dice(logits, target, 0.7, pos_weight)
    assert torch.allclose(mine, original, atol=1e-9)


def test_legacy_normalisation_matches_the_original():
    rng = np.random.default_rng(0)
    volume = rng.random((24, 24, 5)).astype(np.float32) * 400 + 50
    volume[:5] = 0.0
    volume[10, 10, 2] = 5000.0  # a bright outlier, as found near the skull

    mine = normalise_volume(volume, PreprocessConfig(normalisation="legacy_minmax"))
    assert np.allclose(mine, _original_minmax(volume), atol=1e-7)


def test_default_normalisation_deliberately_differs_from_legacy():
    """The default is more robust, and therefore not a reproduction path."""
    rng = np.random.default_rng(1)
    volume = rng.random((24, 24, 5)).astype(np.float32) * 400 + 50
    volume[:5] = 0.0
    volume[10, 10, 2] = 5000.0

    legacy = normalise_volume(volume, PreprocessConfig(normalisation="legacy_minmax"))
    default = normalise_volume(volume, PreprocessConfig(normalisation="minmax"))

    assert not np.allclose(legacy, default)
    brain = volume > 0
    # Percentile clipping stops one bright voxel from squashing the brain.
    assert default[brain].mean() > legacy[brain].mean() * 2


def test_architecture_matches_the_original_parameter_count():
    """7,819,133 parameters, as in the thesis model."""
    model = UNet2D(in_channels=3, out_channels=1, base_channels=32, use_se=True)
    assert model.num_parameters == 7_819_133


def test_state_dict_is_layout_compatible_with_the_original():
    """A checkpoint trained with the original scripts must load unchanged.

    The original defined the same modules under the same attribute names, so the
    state_dict keys must match exactly. If a refactor ever renames a layer, this
    fails and the old weights become unloadable.
    """
    model = UNet2D(in_channels=3, out_channels=1, base_channels=32)
    keys = set(model.state_dict())

    expected_samples = {
        "enc1.block.0.weight",
        "enc1.se.fc.0.weight",
        "bottleneck.block.3.weight",
        "up4.weight",
        "dec4.se.fc.2.bias",
        "out_conv.weight",
        "out_conv.bias",
    }
    assert expected_samples <= keys
    assert len(keys) == 100


def test_thesis_config_pins_the_original_settings():
    from pathlib import Path

    from msseg.config import ExperimentConfig

    path = Path(__file__).resolve().parents[1] / "configs" / "thesis_reproduction.yaml"
    cfg = ExperimentConfig.from_yaml(path)

    assert cfg.train.selection_metric == "soft_dice"
    assert cfg.train.fixed_threshold == 0.7
    assert cfg.train.epochs == 15
    assert cfg.train.early_stopping_patience == 6
    assert cfg.train.scheduler_patience == 2
    assert cfg.train.amp is False
    assert cfg.train.bce_weight == 0.7
    assert cfg.train.pos_weight == 3.0
    assert cfg.train.batch_size == 8
    assert cfg.train.learning_rate == 1e-3
    assert cfg.model.base_channels == 32
    assert cfg.model.use_se is True
    assert cfg.eval.min_component_size == 10


# --------------------------------------------------------------------------
# Legacy checkpoint conversion
# --------------------------------------------------------------------------

def test_legacy_checkpoint_converts_and_reloads(tmp_path):
    """A bare state_dict from the original scripts must survive the round trip."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from convert_legacy_checkpoint import convert

    from msseg.evaluate import load_checkpoint

    original = UNet2D(in_channels=3, out_channels=1, base_channels=32, use_se=True)
    legacy = tmp_path / "old.pth"
    torch.save(original.state_dict(), legacy)

    converted = tmp_path / "new.pt"
    convert(
        input_path=legacy, output_path=converted,
        processed_root="data/processed_legacy",
        manifest="data/manifests/splits_legacy.csv",
        threshold=0.7, base_channels=32, in_channels=3, use_se=True,
        mode="multimodal", min_component_size=10,
    )

    model, cfg, threshold = load_checkpoint(converted, torch.device("cpu"))
    assert threshold == 0.7
    assert cfg.model.base_channels == 32
    assert cfg.data.mode == "multimodal"

    # The weights must be unchanged, not merely loadable.
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    original.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(original(x), model(x))


def test_converting_into_the_wrong_architecture_is_refused(tmp_path):
    """Silently accepting mismatched weights would produce nonsense metrics."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from convert_legacy_checkpoint import convert

    torch.save(
        UNet2D(in_channels=3, base_channels=32).state_dict(), tmp_path / "old.pth"
    )

    with pytest.raises(SystemExit, match="do not fit"):
        convert(
            input_path=tmp_path / "old.pth", output_path=tmp_path / "new.pt",
            processed_root="x", manifest="y", threshold=0.7,
            base_channels=64,  # wrong
            in_channels=3, use_se=True, mode="multimodal", min_component_size=10,
        )
