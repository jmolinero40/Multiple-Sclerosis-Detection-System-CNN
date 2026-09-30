"""Dataset shapes, augmentation invariants, and the no-leakage guarantee."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msseg.data.dataset import MultimodalMSDataset, train_augment
from msseg.data.preprocess import PreprocessConfig, normalise_volume
from msseg.data.splits import SplitConfig, build_manifest, read_manifest


def test_dataset_returns_three_channels_and_one_mask(processed_root, manifest):
    ds = MultimodalMSDataset(processed_root, manifest, split="train")
    image, mask = ds[0]

    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert image.dtype == torch.float32
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_dataset_can_return_the_filename(processed_root, manifest):
    ds = MultimodalMSDataset(processed_root, manifest, split="train", return_name=True)
    _, _, name = ds[0]
    assert name.endswith(".npy")
    assert "_FLAIR_slice_" in name


def test_splits_are_disjoint_by_patient(processed_root, manifest):
    """The single most important property of the whole pipeline."""
    rows = read_manifest(manifest)
    patients_per_split = {}
    for row in rows:
        patients_per_split.setdefault(row["split"], set()).add(row["pid"])

    splits = list(patients_per_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = patients_per_split[a] & patients_per_split[b]
            assert not overlap, f"Patients {overlap} appear in both {a} and {b}"


def test_overlapping_patient_ranges_are_rejected():
    cfg = SplitConfig(patient_ranges={"train": (1, 10), "val": (10, 15)})
    with pytest.raises(ValueError, match="disjoint"):
        cfg.validate()


def test_negative_subsampling_applies_to_train_only(processed_root):
    """Val and test must keep every slice, including lesion-free ones."""
    cfg = SplitConfig(
        patient_ranges={"train": (3, 3), "val": (4, 4)},
        source="testset",
        negative_keep_ratio=0.0,   # drop every lesion-free training slice
        min_brain_fraction=0.0,
    )
    rows = build_manifest(processed_root, cfg)
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    assert "train" not in by_split          # p03 is lesion-free, all dropped
    assert len(by_split["val"]) == 6        # p04 is lesion-free but fully kept


def test_manifest_is_reproducible_given_a_seed(processed_root):
    cfg_a = SplitConfig(patient_ranges={"train": (1, 4)}, source="testset",
                        negative_keep_ratio=0.5, seed=7, min_brain_fraction=0.0)
    cfg_b = SplitConfig(patient_ranges={"train": (1, 4)}, source="testset",
                        negative_keep_ratio=0.5, seed=7, min_brain_fraction=0.0)
    assert build_manifest(processed_root, cfg_a) == build_manifest(processed_root, cfg_b)


def test_augmentation_keeps_the_mask_binary():
    """Bilinear interpolation on a mask would produce fractional labels."""
    torch.manual_seed(0)
    image = torch.rand(3, 32, 32)
    mask = torch.zeros(1, 32, 32)
    mask[0, 8:20, 8:20] = 1.0

    for _ in range(25):
        _, augmented = train_augment(image, mask)
        assert set(torch.unique(augmented).tolist()).issubset({0.0, 1.0})


def test_augmentation_keeps_intensities_in_range_and_background_at_zero():
    torch.manual_seed(1)
    image = torch.rand(3, 32, 32)
    image[:, :6, :] = 0.0  # background strip
    mask = torch.zeros(1, 32, 32)

    for _ in range(25):
        augmented, _ = train_augment(image, mask)
        assert augmented.min() >= 0.0
        assert augmented.max() <= 1.0


def test_normalise_volume_scales_brain_to_unit_range():
    rng = np.random.default_rng(0)
    volume = rng.random((16, 16, 4)) * 500 + 100
    volume[:2] = 0.0  # background

    out = normalise_volume(volume, PreprocessConfig(normalisation="minmax"))

    assert out.dtype == np.float32
    assert np.allclose(out[:2], 0.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalise_volume_handles_an_empty_volume():
    out = normalise_volume(np.zeros((8, 8, 2)), PreprocessConfig())
    assert np.allclose(out, 0.0)


# --------------------------------------------------------------------------
# Geometry: padding vs resizing
# --------------------------------------------------------------------------

MNI_SLICE_HW = (182, 218)   # MSLesSeg native in-plane size (MNI152 1mm)
CANVAS_HW = (192, 224)      # next multiple of 16 in each direction


def test_pad_center_produces_the_target_shape():
    from msseg.data.preprocess import pad_center

    out = pad_center(np.zeros(MNI_SLICE_HW, dtype=np.float32), CANVAS_HW)
    assert out.shape == CANVAS_HW


def test_pad_center_preserves_every_lesion_voxel():
    """The whole point of padding: the ground truth is not altered."""
    from msseg.data.preprocess import pad_center

    mask = np.zeros(MNI_SLICE_HW, dtype=np.uint8)
    mask[40:52, 60:75] = 1
    original = int(mask.sum())

    padded = pad_center(mask, CANVAS_HW)
    assert int(padded.sum()) == original
    assert set(np.unique(padded).tolist()) == {0, 1}


def test_pad_center_centres_the_content():
    from msseg.data.preprocess import pad_center

    array = np.ones((4, 6), dtype=np.float32)
    out = pad_center(array, (8, 10))

    # (8-4)//2 = 2 rows above, (10-6)//2 = 2 columns left.
    assert np.allclose(out[2:6, 2:8], 1.0)
    assert np.allclose(out[:2], 0.0)
    assert np.allclose(out[6:], 0.0)


def test_pad_center_puts_the_odd_margin_at_the_end():
    """A 1-pixel margin cannot be split, so it goes at the bottom and right.

    Matches the original thesis implementation, where ``pad_top`` is the
    floored half. On the real data the margins are even (182 -> 192 and
    218 -> 224), so this only fixes the convention for other inputs.
    """
    from msseg.data.preprocess import pad_center

    out = pad_center(np.ones((3, 3), dtype=np.float32), (4, 4))
    assert np.allclose(out[0:3, 0:3], 1.0)   # content sits at the top left
    assert np.allclose(out[3, :], 0.0)       # spare row at the bottom
    assert np.allclose(out[:, 3], 0.0)       # spare column at the right


def test_pad_center_refuses_to_shrink():
    from msseg.data.preprocess import pad_center

    with pytest.raises(ValueError, match="larger than the target"):
        pad_center(np.zeros((300, 300)), CANVAS_HW)


def test_resize_changes_the_lesion_voxel_count_but_padding_does_not():
    """The 9% ground-truth discrepancy that identified the wrong geometry."""
    from msseg.data.preprocess import PreprocessConfig, _fit

    mask = np.zeros(MNI_SLICE_HW, dtype=np.uint8)
    mask[40:52, 60:75] = 1
    original = int(mask.sum())

    padded = _fit(mask, PreprocessConfig(target_hw=CANVAS_HW, geometry="pad"), is_mask=True)
    stretched = _fit(mask, PreprocessConfig(target_hw=(224, 192), geometry="resize"), is_mask=True)

    assert int(padded.sum()) == original
    assert int(stretched.sum()) != original


def test_both_geometries_yield_dimensions_the_unet_accepts():
    from msseg.data.preprocess import PreprocessConfig, _fit

    for geometry, target in (("pad", CANVAS_HW), ("resize", (224, 192))):
        out = _fit(
            np.zeros(MNI_SLICE_HW, dtype=np.float32),
            PreprocessConfig(target_hw=target, geometry=geometry),
            is_mask=False,
        )
        assert out.shape[0] % 16 == 0 and out.shape[1] % 16 == 0


def test_unknown_geometry_is_rejected():
    from msseg.data.preprocess import PreprocessConfig, _fit

    with pytest.raises(ValueError, match="Unknown geometry"):
        _fit(np.zeros((8, 8)), PreprocessConfig(geometry="squash"), is_mask=False)
