"""Stage 2: build train / val / test manifests from the processed slice pool.

Two things matter here, and both are easy to get wrong.

**Splitting by patient, not by slice.** Adjacent axial slices of the same brain
are nearly identical. Shuffling slices into random splits puts near-duplicates of
the same anatomy on both sides of the train/test boundary, and the reported Dice
becomes an interpolation score rather than a generalisation score. Splits here
are defined by disjoint *patient ranges*, so no patient contributes to more than
one split.

**Filters apply to train only.** The training set drops nearly-empty slices and
subsamples lesion-free slices, because roughly 70% of axial slices in a volume
contain no lesion and the loss is otherwise dominated by trivial negatives.
Validation and test keep **every** slice, including the empty ones: the test
distribution must match what the model would see in deployment, where nobody
pre-filters the slices for it.

Output is a **manifest** (CSV) per split rather than a copy of the data. The
original version of this project copied several gigabytes of ``.npy`` files
three times over; a manifest is a few hundred kilobytes, is diffable, and makes
the exact composition of every split reproducible and reviewable.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from msseg.data.naming import slice_filename

logger = logging.getLogger(__name__)

__all__ = ["SplitConfig", "build_manifest", "write_manifest", "read_manifest", "main"]

MODALITIES = ("FLAIR", "T1", "T2")
MANIFEST_FIELDS = ("split", "source", "pid", "z", "has_lesion", "lesion_voxels")


@dataclass
class SplitConfig:
    """Parameters for stage 2.

    Attributes:
        patient_ranges: Split name -> inclusive ``(first, last)`` patient range.
            Ranges must not overlap; :meth:`validate` enforces this.
        excluded_patients: Patients dropped entirely, e.g. because of a
            corrupted acquisition. Document every exclusion in the README.
        brain_threshold: Intensity above which a pixel counts as brain tissue.
        min_brain_fraction: Training slices with a smaller brain fraction than
            this are dropped (they are mostly neck or empty space above the head).
        negative_keep_ratio: Fraction of lesion-free training slices kept.
        seed: Seed for the negative subsampling, so the split is reproducible.
        source: Dataset tag, must match the one used in stage 1.
    """

    patient_ranges: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"train": (1, 53), "val": (54, 64), "test": (65, 75)}
    )
    excluded_patients: tuple[int, ...] = ()
    brain_threshold: float = 0.05
    min_brain_fraction: float = 0.10
    negative_keep_ratio: float = 0.3
    seed: int = 42
    source: str = "mslesseg"

    def validate(self) -> None:
        """Raise if any patient appears in more than one split."""
        seen: dict[int, str] = {}
        for split, (first, last) in self.patient_ranges.items():
            for p in range(first, last + 1):
                if p in seen:
                    raise ValueError(
                        f"Patient {p} is in both {seen[p]!r} and {split!r}: "
                        "splits must be disjoint to avoid leakage."
                    )
                seen[p] = split

    def split_of(self, patient: int) -> str | None:
        for split, (first, last) in self.patient_ranges.items():
            if first <= patient <= last:
                return split
        return None


def build_manifest(processed_root: Path, cfg: SplitConfig) -> list[dict]:
    """Scan the processed pool and decide which slices belong to which split.

    Args:
        processed_root: Folder containing ``slices/`` and ``masks/``.
        cfg: Split configuration.

    Returns:
        One row per retained slice, with the fields in :data:`MANIFEST_FIELDS`.
    """
    cfg.validate()
    rng = random.Random(cfg.seed)

    slices_dir = processed_root / "slices"
    masks_dir = processed_root / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"No masks/ folder under {processed_root}")

    rows: list[dict] = []
    stats = {
        s: {"kept": 0, "dropped_small_brain": 0, "dropped_negative": 0, "dropped_incomplete": 0}
        for s in cfg.patient_ranges
    }

    for patient in sorted(
        p
        for rng_ in cfg.patient_ranges.values()
        for p in range(rng_[0], rng_[1] + 1)
    ):
        if patient in cfg.excluded_patients:
            logger.info("P%d: excluded by configuration", patient)
            continue

        split = cfg.split_of(patient)
        if split is None:
            continue

        pid = f"p{patient:02d}"
        pattern = f"{cfg.source}_mask_{pid}_z*.npy"

        for mask_path in sorted(masks_dir.glob(pattern)):
            z = int(mask_path.stem.rsplit("_z", 1)[1])

            modality_paths = {
                m: slices_dir / slice_filename(cfg.source, m, pid, z) for m in MODALITIES
            }
            if not all(p.exists() for p in modality_paths.values()):
                stats[split]["dropped_incomplete"] += 1
                continue

            mask = np.load(mask_path)
            lesion_voxels = int((mask > 0).sum())
            has_lesion = lesion_voxels > 0

            if split == "train":
                flair = np.load(modality_paths["FLAIR"])
                brain_fraction = float((flair > cfg.brain_threshold).mean())
                if brain_fraction < cfg.min_brain_fraction:
                    stats[split]["dropped_small_brain"] += 1
                    continue
                if not has_lesion and rng.random() > cfg.negative_keep_ratio:
                    stats[split]["dropped_negative"] += 1
                    continue

            rows.append(
                {
                    "split": split,
                    "source": cfg.source,
                    "pid": pid,
                    "z": z,
                    "has_lesion": int(has_lesion),
                    "lesion_voxels": lesion_voxels,
                }
            )
            stats[split]["kept"] += 1

    for split, s in stats.items():
        logger.info(
            "%-5s kept=%5d  dropped: small_brain=%4d negative=%5d incomplete=%3d",
            split, s["kept"], s["dropped_small_brain"], s["dropped_negative"],
            s["dropped_incomplete"],
        )
    return rows


def write_manifest(rows: list[dict], out_path: Path) -> None:
    """Write manifest rows to CSV, sorted for a stable diff."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["split"], r["source"], r["pid"], r["z"]))
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest written: %s (%d rows)", out_path, len(rows))


def read_manifest(path: Path, split: str | None = None) -> list[dict]:
    """Read a manifest CSV, optionally keeping only one split."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["z"] = int(r["z"])
        r["has_lesion"] = bool(int(r["has_lesion"]))
        r["lesion_voxels"] = int(r["lesion_voxels"])
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/manifests/splits.csv"))
    parser.add_argument("--negative-keep-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="mslesseg")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = SplitConfig(
        negative_keep_ratio=args.negative_keep_ratio,
        seed=args.seed,
        source=args.source,
    )
    write_manifest(build_manifest(args.processed_root, cfg), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
