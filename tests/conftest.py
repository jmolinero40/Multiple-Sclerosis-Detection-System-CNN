"""Shared fixtures: a tiny synthetic dataset on disk.

The tests never touch the real MRI data. They build a handful of small random
volumes in a temporary directory, which keeps the suite fast enough to run on
every commit and means CI works without access to the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msseg.data.naming import mask_filename, slice_filename

MODALITIES = ("FLAIR", "T1", "T2")
SOURCE = "testset"
HEIGHT, WIDTH = 32, 32


@pytest.fixture
def processed_root(tmp_path: Path) -> Path:
    """A processed pool with 4 patients x 6 slices x 3 modalities.

    Patients p01 and p02 carry a lesion in the middle slices; p03 and p04 are
    lesion-free, so tests can exercise both branches of the split filters.
    """
    slices_dir = tmp_path / "slices"
    masks_dir = tmp_path / "masks"
    slices_dir.mkdir()
    masks_dir.mkdir()

    rng = np.random.default_rng(0)

    for patient in range(1, 5):
        pid = f"p{patient:02d}"
        for z in range(6):
            for modality in MODALITIES:
                image = rng.random((HEIGHT, WIDTH)).astype(np.float32)
                image[:4, :] = 0.0  # a strip of background, so brain fraction < 1
                np.save(slices_dir / slice_filename(SOURCE, modality, pid, z), image)

            mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            if patient <= 2 and 1 <= z <= 3:
                mask[10:18, 10:18] = 1
            np.save(masks_dir / mask_filename(SOURCE, pid, z), mask)

    return tmp_path


@pytest.fixture
def manifest(processed_root: Path) -> Path:
    """A manifest over the synthetic pool, keeping every slice."""
    from msseg.data.splits import SplitConfig, build_manifest, write_manifest

    cfg = SplitConfig(
        patient_ranges={"train": (1, 2), "val": (3, 3), "test": (4, 4)},
        source=SOURCE,
        negative_keep_ratio=1.0,
        min_brain_fraction=0.0,
    )
    path = processed_root / "manifest.csv"
    write_manifest(build_manifest(processed_root, cfg), path)
    return path
