"""Single source of truth for the on-disk filename convention.

Every processed slice and mask follows one of two patterns::

    {source}_{modality}_slice_{pid}_z{zzz}.npy    # e.g. mslesseg_FLAIR_slice_p01_z039.npy
    {source}_mask_{pid}_z{zzz}.npy                # e.g. mslesseg_mask_p01_z039.npy

There is exactly **one mask per (patient, slice)**: the ground truth is defined
in a single space shared by the three co-registered modalities, so storing it
per modality would duplicate identical data and invite drift.

Parsing filenames with ad-hoc ``split("_")`` calls scattered across scripts was
the main source of silent bugs in the first version of this project (an index
that was correct for ``nature_slice_p01_z039`` was off by one for
``nature_FLAIR_slice_p01_z039``). Everything now goes through this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SliceId",
    "slice_filename",
    "mask_filename",
    "parse_slice_filename",
    "parse_mask_filename",
    "patient_key",
    "z_index",
]

_SLICE_RE = re.compile(
    r"^(?P<source>[a-z0-9]+)_(?P<modality>[A-Za-z0-9]+)_slice_(?P<pid>p\d+)_z(?P<z>\d+)\.npy$"
)
_MASK_RE = re.compile(r"^(?P<source>[a-z0-9]+)_mask_(?P<pid>p\d+)_z(?P<z>\d+)\.npy$")


@dataclass(frozen=True)
class SliceId:
    """Identity of one 2D slice: which dataset, patient, axial index, modality."""

    source: str
    pid: str
    z: int
    modality: str | None = None

    @property
    def patient_key(self) -> str:
        """Stable key for grouping, e.g. ``"mslesseg_p01"``."""
        return f"{self.source}_{self.pid}"


def slice_filename(source: str, modality: str, pid: str, z: int) -> str:
    """Build the filename for one modality of one slice."""
    return f"{source}_{modality}_slice_{pid}_z{z:03d}.npy"


def mask_filename(source: str, pid: str, z: int) -> str:
    """Build the filename for the (single, modality-agnostic) mask of one slice."""
    return f"{source}_mask_{pid}_z{z:03d}.npy"


def parse_slice_filename(name: str) -> SliceId:
    """Parse a slice filename. Raises ``ValueError`` on anything unexpected."""
    m = _SLICE_RE.match(name)
    if m is None:
        raise ValueError(f"Not a valid slice filename: {name!r}")
    return SliceId(
        source=m["source"],
        pid=m["pid"],
        z=int(m["z"]),
        modality=m["modality"],
    )


def parse_mask_filename(name: str) -> SliceId:
    """Parse a mask filename. Raises ``ValueError`` on anything unexpected."""
    m = _MASK_RE.match(name)
    if m is None:
        raise ValueError(f"Not a valid mask filename: {name!r}")
    return SliceId(source=m["source"], pid=m["pid"], z=int(m["z"]), modality=None)


def patient_key(name: str) -> str:
    """Extract ``"{source}_{pid}"`` from either a slice or a mask filename."""
    for parser in (parse_slice_filename, parse_mask_filename):
        try:
            return parser(name).patient_key
        except ValueError:
            continue
    raise ValueError(f"Cannot extract a patient key from: {name!r}")


def z_index(name: str) -> int:
    """Extract the axial index from either a slice or a mask filename."""
    for parser in (parse_slice_filename, parse_mask_filename):
        try:
            return parser(name).z
        except ValueError:
            continue
    raise ValueError(f"Cannot extract a z index from: {name!r}")
