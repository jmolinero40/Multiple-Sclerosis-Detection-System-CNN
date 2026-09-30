"""Stage 3: the PyTorch ``Dataset`` and the training-time augmentation.

The final system stacks the **three co-registered modalities** of one axial
slice into the channel dimension::

    channel 0 = FLAIR   (lesions are hyperintense; the primary signal)
    channel 1 = T1      (lesions are hypointense; separates them from artefacts)
    channel 2 = T2      (hyperintense, sensitive but less specific than FLAIR)

so one sample is ``(3, H, W)`` with a single ``(1, H, W)`` binary mask.

An earlier variant of this project used the same three-channel shape for a
different purpose: stacking neighbouring slices ``z-1, z, z+1`` of FLAIR alone
("2.5D"), giving the network local through-plane context. That variant is kept
in :class:`AxialContextDataset` for the ablation reported in the README, but it
is **not** the final system.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from msseg.data.naming import mask_filename, slice_filename
from msseg.data.splits import read_manifest

logger = logging.getLogger(__name__)

__all__ = ["MultimodalMSDataset", "AxialContextDataset", "train_augment"]

MODALITIES = ("FLAIR", "T1", "T2")


class MultimodalMSDataset(Dataset):
    """Axial slices with FLAIR / T1 / T2 stacked as channels.

    Args:
        processed_root: Folder containing ``slices/`` and ``masks/``.
        manifest: Path to the split manifest CSV produced by
            :mod:`msseg.data.splits`.
        split: ``"train"``, ``"val"`` or ``"test"``.
        transform: Optional callable ``f(image, mask) -> (image, mask)`` applied
            to the tensors. Pass :func:`train_augment` for training, ``None``
            for validation and test.
        return_name: If ``True``, ``__getitem__`` also returns the FLAIR
            filename. Evaluation needs it to group results by patient.

    Yields:
        ``(image, mask)`` or ``(image, mask, name)`` where ``image`` is
        ``(3, H, W)`` float32 and ``mask`` is ``(1, H, W)`` float32 in
        ``{0.0, 1.0}``.
    """

    def __init__(
        self,
        processed_root: str | Path,
        manifest: str | Path,
        split: str,
        transform=None,
        return_name: bool = False,
    ) -> None:
        self.root = Path(processed_root)
        self.slices_dir = self.root / "slices"
        self.masks_dir = self.root / "masks"
        self.split = split
        self.transform = transform
        self.return_name = return_name

        for d in (self.slices_dir, self.masks_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Missing directory: {d}")

        self.rows = read_manifest(Path(manifest), split=split)
        if not self.rows:
            raise RuntimeError(f"Manifest {manifest} contains no rows for split={split!r}")

        n_positive = sum(r["has_lesion"] for r in self.rows)
        logger.info(
            "split=%-5s  %d slices  (%d with lesion, %.1f%%)",
            split, len(self.rows), n_positive, 100.0 * n_positive / len(self.rows),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _paths(self, row: dict) -> tuple[dict[str, Path], Path]:
        source, pid, z = row["source"], row["pid"], row["z"]
        images = {
            m: self.slices_dir / slice_filename(source, m, pid, z) for m in MODALITIES
        }
        mask = self.masks_dir / mask_filename(source, pid, z)
        return images, mask

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_paths, mask_path = self._paths(row)

        channels = [np.load(image_paths[m]).astype(np.float32) for m in MODALITIES]
        shapes = {c.shape for c in channels}
        if len(shapes) != 1:
            raise ValueError(
                f"{row['pid']} z={row['z']}: modalities have different shapes {shapes}"
            )

        image = torch.from_numpy(np.stack(channels, axis=0))
        mask_np = (np.load(mask_path).astype(np.float32) > 0.5).astype(np.float32)
        mask = torch.from_numpy(mask_np[None])  # (1, H, W)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.return_name:
            return image, mask, image_paths["FLAIR"].name
        return image, mask


class AxialContextDataset(Dataset):
    """2.5D variant: FLAIR slices ``z-1, z, z+1`` stacked as channels.

    Kept for the ablation in the README. At the volume boundary the missing
    neighbour is replaced by the central slice, so the tensor shape is constant.
    """

    def __init__(
        self,
        processed_root: str | Path,
        manifest: str | Path,
        split: str,
        transform=None,
        return_name: bool = False,
        modality: str = "FLAIR",
    ) -> None:
        self.root = Path(processed_root)
        self.slices_dir = self.root / "slices"
        self.masks_dir = self.root / "masks"
        self.transform = transform
        self.return_name = return_name
        self.modality = modality
        self.rows = read_manifest(Path(manifest), split=split)
        if not self.rows:
            raise RuntimeError(f"Manifest {manifest} contains no rows for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def _slice_or_fallback(self, source: str, pid: str, z: int, fallback: np.ndarray) -> np.ndarray:
        path = self.slices_dir / slice_filename(source, self.modality, pid, z)
        if not path.exists():
            return fallback
        return np.load(path).astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        source, pid, z = row["source"], row["pid"], row["z"]

        centre_path = self.slices_dir / slice_filename(source, self.modality, pid, z)
        centre = np.load(centre_path).astype(np.float32)
        previous = self._slice_or_fallback(source, pid, z - 1, centre)
        following = self._slice_or_fallback(source, pid, z + 1, centre)

        image = torch.from_numpy(np.stack([previous, centre, following], axis=0))

        mask_np = (
            np.load(self.masks_dir / mask_filename(source, pid, z)).astype(np.float32) > 0.5
        ).astype(np.float32)
        mask = torch.from_numpy(mask_np[None])

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.return_name:
            return image, mask, centre_path.name
        return image, mask


def train_augment(
    image: torch.Tensor,
    mask: torch.Tensor,
    p_geometric: float = 0.8,
    p_intensity: float = 0.7,
    max_rotation: float = 10.0,
    brightness_delta: float = 0.06,
    contrast_range: tuple[float, float] = (0.90, 1.10),
    gamma_range: tuple[float, float] = (0.90, 1.10),
    noise_std: float = 0.01,
    brain_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmentation for slices normalised to ``[0, 1]`` with a zero background.

    Geometric transforms are applied to the image, the mask **and** a derived
    brain mask, with nearest-neighbour interpolation for the two binary maps so
    that labels stay binary. Intensity transforms are applied to the image only
    and are confined to brain tissue; the background is forced back to exactly 0
    at the end, because a non-zero background would leak a "this slice was
    augmented" cue into the network.

    Rotation is limited to +/- 10 degrees: head position in an MRI is
    constrained by the coil, so larger rotations generate anatomy the model will
    never encounter. Flips are kept despite the brain's left-right asymmetry,
    since MS lesion location is not lateralised in a clinically meaningful way
    and the extra invariance is worth more than the lost asymmetry cue.

    Args:
        image: ``(C, H, W)`` float tensor in ``[0, 1]``.
        mask: ``(1, H, W)`` float tensor in ``{0, 1}``.

    Returns:
        The augmented ``(image, mask)`` pair.
    """
    image = image.float()
    mask = mask.float()

    # Brain mask: max across channels, so a voxel visible in any modality counts.
    brain = (image.max(dim=0).values > brain_threshold).float().unsqueeze(0)

    if random.random() < p_geometric:
        if random.random() < 0.5:
            image, mask, brain = TF.hflip(image), TF.hflip(mask), TF.hflip(brain)
        if random.random() < 0.5:
            image, mask, brain = TF.vflip(image), TF.vflip(mask), TF.vflip(brain)

        angle = random.uniform(-max_rotation, max_rotation)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        brain = TF.rotate(brain, angle, interpolation=InterpolationMode.NEAREST)

    if random.random() < p_intensity:
        image = image + random.uniform(-brightness_delta, brightness_delta)

        # Contrast around the mean intensity of brain tissue, not of the whole
        # image: the large zero background would otherwise pull the mean to ~0
        # and turn the contrast change into a brightness change.
        brain_mean = (image * brain).sum() / (brain.sum() * image.size(0) + 1e-8)
        image = (image - brain_mean) * random.uniform(*contrast_range) + brain_mean

        image = torch.clamp(image, 0.0, 1.0) ** random.uniform(*gamma_range)

        if noise_std > 0:
            image = image + torch.randn_like(image) * noise_std * brain

    image = torch.clamp(image * brain, 0.0, 1.0)
    return image, mask
