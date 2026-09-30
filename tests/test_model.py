"""Model, loss and post-processing invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msseg.losses import BCEDiceLoss, soft_dice
from msseg.models.unet import SEBlock, UNet2D
from msseg.postprocess import postprocess_batch, remove_small_components, sweep_threshold


def test_forward_preserves_spatial_dimensions():
    model = UNet2D(in_channels=3, out_channels=1, base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 1, 64, 64)


def test_forward_handles_non_square_input():
    """The real slices are 224x192, not square."""
    model = UNet2D(in_channels=3, out_channels=1, base_channels=8)
    assert model(torch.randn(1, 3, 224, 192)).shape == (1, 1, 224, 192)


def test_input_not_divisible_by_sixteen_fails_loudly():
    """Four max-pools require H and W divisible by 16. Better a crash than a silent crop."""
    model = UNet2D(in_channels=3, out_channels=1, base_channels=8)
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 30, 30))


def test_se_block_preserves_shape_and_gates_within_zero_one():
    block = SEBlock(channels=16, reduction=4)
    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    assert out.shape == x.shape
    gate = (out / (x + 1e-8)).flatten()
    assert torch.isfinite(gate).any()


def test_disabling_se_reduces_the_parameter_count():
    with_se = UNet2D(base_channels=8, use_se=True).num_parameters
    without_se = UNet2D(base_channels=8, use_se=False).num_parameters
    assert without_se < with_se


def test_gradients_flow_to_every_parameter():
    model = UNet2D(in_channels=3, out_channels=1, base_channels=8)
    criterion = BCEDiceLoss(bce_weight=0.7)
    loss = criterion(model(torch.randn(2, 3, 32, 32)), torch.zeros(2, 1, 32, 32))
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"


def test_soft_dice_bounds():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:6, 2:6] = 1.0

    assert soft_dice(target.clone(), target).item() == pytest.approx(1.0, abs=1e-4)
    assert soft_dice(1.0 - target, target).item() < 0.1


def test_bce_dice_loss_is_lower_for_a_better_prediction():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 4:12, 4:12] = 1.0

    good = torch.where(target > 0, 4.0, -4.0)   # confident and correct logits
    bad = torch.where(target > 0, -4.0, 4.0)    # confident and wrong

    criterion = BCEDiceLoss(bce_weight=0.7)
    assert criterion(good, target).item() < criterion(bad, target).item()


def test_bce_weight_must_be_a_valid_proportion():
    with pytest.raises(ValueError):
        BCEDiceLoss(bce_weight=1.5)


def test_remove_small_components_keeps_large_and_drops_small():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:12, 4:12] = 1   # 64 pixels, keep
    mask[20, 20] = 1       # 1 pixel, drop
    mask[25:27, 25:27] = 1 # 4 pixels, drop

    cleaned = remove_small_components(mask, min_size=10)
    assert cleaned.sum() == 64
    assert cleaned[20, 20] == 0


def test_remove_small_components_uses_eight_connectivity():
    """Two diagonally touching pixels are one component of size 2, not two of size 1."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3, 3] = 1
    mask[4, 4] = 1
    assert remove_small_components(mask, min_size=2).sum() == 2
    assert remove_small_components(mask, min_size=3).sum() == 0


def test_min_size_zero_disables_the_filter():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 1
    assert remove_small_components(mask, min_size=0).sum() == 1


def test_postprocess_batch_returns_binary_on_the_input_device():
    probabilities = torch.rand(3, 1, 32, 32)
    out = postprocess_batch(probabilities, threshold=0.5, min_size=0)
    assert out.shape == probabilities.shape
    assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})
    assert out.device == probabilities.device


def test_sweep_threshold_finds_the_separating_value():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 4:12, 4:12] = 1.0
    probabilities = torch.where(target > 0, 0.85, 0.15)

    best, score = sweep_threshold([probabilities], [target])
    assert 0.15 < best <= 0.85
    assert score > 0.99
