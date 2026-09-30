"""Stage 1: NIfTI volumes -> normalised 2D axial slices stored as ``.npy``.

Input layout expected (MSLesSeg, "unified" folder)::

    <raw_root>/P1/T1/P1_T1_FLAIR.nii.gz
    <raw_root>/P1/T1/P1_T1_T1.nii.gz
    <raw_root>/P1/T1/P1_T1_T2.nii.gz
    <raw_root>/P1/T1/P1_T1_MASK.nii.gz

Note the naming collision in the source dataset: the ``T1`` folder and the
second ``T1`` in the filename denote **timepoint 1**, while the trailing token
is the **modality**. So ``P1_T1_T2.nii.gz`` is "patient 1, timepoint 1,
T2-weighted".

Output layout::

    <processed_root>/slices/mslesseg_FLAIR_slice_p01_z039.npy   (H, W) float32
    <processed_root>/slices/mslesseg_T1_slice_p01_z039.npy
    <processed_root>/slices/mslesseg_T2_slice_p01_z039.npy
    <processed_root>/masks/mslesseg_mask_p01_z039.npy           (H, W) uint8

Design decisions worth knowing:

* **Target 192x224** (H x W). MSLesSeg is distributed in MNI152 1mm space, so
  every volume is 182x218x182: 182 axial slices of 182x218. Neither 182 nor 218
  is divisible by 16, which the four-level U-Net requires, so each slice is
  centred in a 192x224 canvas -- the next multiple of 16 in each direction.
* **Padding rather than resizing, by default.** Centring the slice leaves every
  original pixel untouched: no interpolation, no deformation, and the lesion
  voxel count stays exactly what the annotator drew. Stretching to the same
  size changes all three. ``--geometry resize`` is available for data that does
  not fit the canvas.
* **Masks are never interpolated.** Under ``resize`` they use nearest-neighbour;
  bilinear on a binary mask produces fractional labels at lesion borders, which
  silently changes the definition of the ground truth.
* **Normalisation is per volume, not per slice.** Per-slice normalisation
  destroys the relative intensity of a slice within its own volume, which is
  precisely the signal that distinguishes a hyperintense lesion from normal
  white matter on FLAIR.
* Intensities are normalised **inside the brain only** (non-zero voxels), and
  the background is forced back to 0 afterwards.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

from msseg.data.naming import mask_filename, slice_filename

logger = logging.getLogger(__name__)

__all__ = ["PreprocessConfig", "normalise_volume", "process_patient", "main"]

MODALITIES = ("FLAIR", "T1", "T2")


@dataclass
class PreprocessConfig:
    """Parameters for stage 1.

    Attributes:
        target_hw: Output spatial size as ``(height, width)``.
        geometry: How a native slice is brought to ``target_hw``.

            ``"pad"``
                Centre the slice in a zero-filled canvas. **Geometry
                preserving**: no interpolation, no deformation, and the lesion
                voxel count is exactly the one the annotator drew. This is what
                the original thesis used, and it is the better choice in
                general. Requires ``target_hw`` to be at least as large as the
                native slice in both dimensions.
            ``"resize"``
                Stretch the slice to ``target_hw`` (bilinear for images,
                nearest for masks). Works for any input size, but deforms the
                anatomy and changes the lesion voxel count.

            A model trained under one of these cannot be evaluated under the
            other: the anatomy it sees is not the anatomy it learned, and the
            metrics drop with no error raised.
        normalisation: One of

            ``"legacy_minmax"``
                Plain min-max over the **whole volume**, background included,
                with no outlier clipping. This is exactly what the original
                thesis used. Reproduces those results bit for bit, and is the
                setting to use with a checkpoint trained under it.
            ``"minmax"``
                Min-max over **brain voxels only**, after clipping at
                ``clip_percentiles``. Default, and the more robust choice.
            ``"zscore"``
                Standardise brain voxels to zero mean and unit variance after
                the same clipping, clip at ``zscore_clip`` sigma, rescale to
                ``[0, 1]``.

            The three are **not interchangeable for an already-trained model**:
            a network learns the intensity distribution it was fed, so a
            checkpoint must be evaluated under the scheme it was trained with.
        clip_percentiles: Percentiles used to clip outliers before computing
            the normalisation statistics. Protects against single bright voxels
            (common near the skull) compressing the whole dynamic range.
            Ignored by ``"legacy_minmax"``.
        zscore_clip: Number of standard deviations kept when
            ``normalisation="zscore"``.
        source: Short dataset tag written into every filename.
    """

    target_hw: tuple[int, int] = (192, 224)
    geometry: str = "pad"
    normalisation: str = "minmax"
    clip_percentiles: tuple[float, float] = (0.5, 99.5)
    zscore_clip: float = 3.0
    source: str = "mslesseg"


def normalise_volume(volume: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Normalise a 3D volume to ``[0, 1]``.

    Args:
        volume: 3D array of raw intensities.
        cfg: Preprocessing configuration, whose ``normalisation`` field selects
            the scheme. See :class:`PreprocessConfig`.

    Returns:
        Float32 array of the same shape, values in ``[0, 1]``.
    """
    vol = volume.astype(np.float32)

    if cfg.normalisation == "legacy_minmax":
        # The original thesis formulation, kept verbatim so that results can be
        # reproduced exactly. Statistics come from the whole volume, background
        # included, and there is no outlier clipping: a single bright voxel near
        # the skull therefore compresses the brain into a narrow band of the
        # output range. That is a real weakness, which is why it is not the
        # default -- but changing it silently would make this repository stop
        # reproducing the numbers it reports.
        lo, hi = float(vol.min()), float(vol.max())
        return ((vol - lo) / (hi - lo + 1e-8)).astype(np.float32)

    brain = vol > 0
    values = vol[brain]

    if values.size == 0:
        return np.zeros_like(vol, dtype=np.float32)

    lo, hi = np.percentile(values, cfg.clip_percentiles)

    if cfg.normalisation == "minmax":
        if hi - lo < 1e-6:
            return np.zeros_like(vol, dtype=np.float32)
        out = (np.clip(vol, lo, hi) - lo) / (hi - lo)

    elif cfg.normalisation == "zscore":
        clipped = np.clip(values, lo, hi)
        mu, sigma = float(clipped.mean()), float(clipped.std())
        if sigma < 1e-6:
            return np.zeros_like(vol, dtype=np.float32)
        z = np.clip((vol - mu) / sigma, -cfg.zscore_clip, cfg.zscore_clip)
        out = (z + cfg.zscore_clip) / (2.0 * cfg.zscore_clip)

    else:
        raise ValueError(f"Unknown normalisation: {cfg.normalisation!r}")

    out[~brain] = 0.0
    return out.astype(np.float32)


def pad_center(array: np.ndarray, target_hw: tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    """Centre a 2D slice inside a constant-filled canvas of ``target_hw``.

    Padding leaves every original pixel untouched, so a binary mask keeps
    exactly the voxel count it had. When the margin is odd, the extra row or
    column goes at the bottom or right.

    Args:
        array: 2D input slice.
        target_hw: Output ``(height, width)``. Must be at least the input size.
        pad_value: Value written into the margin.

    Raises:
        ValueError: If the input is larger than the target in either dimension.
    """
    h, w = array.shape
    out_h, out_w = target_hw
    if h > out_h or w > out_w:
        raise ValueError(
            f"Input slice is {h}x{w}, larger than the target {out_h}x{out_w}. "
            "Padding cannot shrink a slice; either raise target_hw or use "
            "geometry='resize'."
        )

    top = (out_h - h) // 2
    left = (out_w - w) // 2
    return np.pad(
        array,
        ((top, out_h - h - top), (left, out_w - w - left)),
        mode="constant",
        constant_values=pad_value,
    )


def _fit(array: np.ndarray, cfg: PreprocessConfig, is_mask: bool) -> np.ndarray:
    """Bring one 2D slice to ``cfg.target_hw`` using the configured geometry."""
    dtype = np.uint8 if is_mask else np.float32
    array = array.astype(dtype)

    if cfg.geometry == "pad":
        return pad_center(array, cfg.target_hw, pad_value=0)

    if cfg.geometry == "resize":
        # cv2.resize takes (width, height), the reverse of the (h, w) convention
        # used everywhere else in this codebase.
        h, w = cfg.target_hw
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(array, (w, h), interpolation=interpolation)

    raise ValueError(f"Unknown geometry: {cfg.geometry!r}. Expected 'pad' or 'resize'.")


def _patient_paths(raw_root: Path, patient: int, timepoint: int = 1) -> dict[str, Path]:
    """Map ``"FLAIR" | "T1" | "T2" | "MASK"`` to the NIfTI path for one patient."""
    folder = raw_root / f"P{patient}" / f"T{timepoint}"
    prefix = f"P{patient}_T{timepoint}"
    paths = {m: folder / f"{prefix}_{m}.nii.gz" for m in MODALITIES}
    paths["MASK"] = folder / f"{prefix}_MASK.nii.gz"
    return paths


def process_patient(
    raw_root: Path,
    out_root: Path,
    patient: int,
    cfg: PreprocessConfig,
    timepoint: int = 1,
) -> int:
    """Convert one patient's volumes into per-slice ``.npy`` files.

    Returns:
        Number of slices written (0 if the patient was skipped).

    Raises:
        ValueError: If the four volumes do not share the same shape, which would
            mean they are not co-registered and the mask does not describe the
            images.
    """
    paths = _patient_paths(raw_root, patient, timepoint)

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        logger.warning("P%d: skipping, missing %s", patient, ", ".join(sorted(missing)))
        return 0

    volumes = {k: nib.load(str(p)).get_fdata() for k, p in paths.items()}

    shapes = {k: v.shape for k, v in volumes.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"P{patient}: volumes are not co-registered, shapes={shapes}")

    mask_values = np.unique(volumes["MASK"])
    if not set(np.round(mask_values).tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"P{patient}: mask is not binary, unique values={mask_values}")

    images = {m: normalise_volume(volumes[m], cfg) for m in MODALITIES}
    mask_volume = (volumes["MASK"] > 0.5).astype(np.uint8)

    slices_dir = out_root / "slices"
    masks_dir = out_root / "masks"
    slices_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    pid = f"p{patient:02d}"
    n_z = mask_volume.shape[2]

    for z in range(n_z):
        for modality in MODALITIES:
            arr = _fit(images[modality][:, :, z], cfg, is_mask=False)
            np.save(slices_dir / slice_filename(cfg.source, modality, pid, z), arr)

        mask = _fit(mask_volume[:, :, z], cfg, is_mask=True)
        np.save(masks_dir / mask_filename(cfg.source, pid, z), mask)

    logger.info(
        "P%d: wrote %d slices x %d modalities (%dx%d native -> %dx%d by %s)",
        patient, n_z, len(MODALITIES),
        mask_volume.shape[0], mask_volume.shape[1],
        cfg.target_hw[0], cfg.target_hw[1], cfg.geometry,
    )
    return n_z


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Folder containing P1/, P2/, ... from MSLesSeg.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="Destination for slices/ and masks/.")
    parser.add_argument("--patients", type=int, nargs=2, default=(1, 75),
                        metavar=("FIRST", "LAST"), help="Inclusive patient range.")
    parser.add_argument("--height", type=int, default=192,
                        help="Output height. Must be >= the native slice height when padding.")
    parser.add_argument("--width", type=int, default=224,
                        help="Output width. Must be >= the native slice width when padding.")
    parser.add_argument(
        "--geometry", choices=("pad", "resize"), default="pad",
        help="pad centres the slice without deforming it (default, and what the "
             "thesis used); resize stretches it to the target size.",
    )
    parser.add_argument(
        "--normalisation",
        choices=("minmax", "zscore", "legacy_minmax"),
        default="minmax",
        help="legacy_minmax reproduces the original thesis preprocessing exactly.",
    )
    parser.add_argument("--source", default="mslesseg")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = PreprocessConfig(
        target_hw=(args.height, args.width),
        geometry=args.geometry,
        normalisation=args.normalisation,
        source=args.source,
    )

    for side, size in (("height", args.height), ("width", args.width)):
        if size % 16:
            logger.warning(
                "%s=%d is not divisible by 16; the four-level U-Net will fail on it.",
                side, size,
            )

    first, last = args.patients
    total = 0
    for patient in range(first, last + 1):
        total += process_patient(args.raw_root, args.out_root, patient, cfg)

    logger.info("Done. %d slices written to %s", total, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
