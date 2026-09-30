"""Filename parsing. These are the bugs that bit hardest in the first version."""

from __future__ import annotations

import pytest

from msseg.data.naming import (
    SliceId,
    mask_filename,
    parse_mask_filename,
    parse_slice_filename,
    patient_key,
    slice_filename,
    z_index,
)


def test_slice_filename_roundtrip():
    name = slice_filename("mslesseg", "FLAIR", "p01", 39)
    assert name == "mslesseg_FLAIR_slice_p01_z039.npy"
    assert parse_slice_filename(name) == SliceId("mslesseg", "p01", 39, "FLAIR")


def test_mask_filename_roundtrip():
    name = mask_filename("mslesseg", "p07", 5)
    assert name == "mslesseg_mask_p07_z005.npy"
    assert parse_mask_filename(name) == SliceId("mslesseg", "p07", 5, None)


def test_z_index_is_zero_padded_to_three_digits():
    # Without padding, sorted() orders z10 before z9, which silently scrambles
    # the neighbour lookup in the 2.5D dataset.
    assert slice_filename("s", "FLAIR", "p01", 9).endswith("z009.npy")
    names = sorted(slice_filename("s", "FLAIR", "p01", z) for z in (2, 10, 9))
    assert [z_index(n) for n in names] == [2, 9, 10]


def test_patient_key_works_for_both_patterns():
    assert patient_key("mslesseg_FLAIR_slice_p12_z100.npy") == "mslesseg_p12"
    assert patient_key("mslesseg_mask_p12_z100.npy") == "mslesseg_p12"


@pytest.mark.parametrize(
    "bad",
    [
        "mslesseg_slice_p01_z039.npy",       # no modality
        "mslesseg_FLAIR_slice_p01_z039.npz", # wrong extension
        "mslesseg_FLAIR_slice_1_z039.npy",   # pid missing the 'p'
        "random_file.npy",
    ],
)
def test_invalid_names_raise(bad):
    with pytest.raises(ValueError):
        parse_slice_filename(bad)


def test_modality_token_does_not_shift_the_patient_index():
    """Regression: an index-based parser confuses these two patterns."""
    assert patient_key("src_FLAIR_slice_p65_z080.npy") == "src_p65"
    assert z_index("src_FLAIR_slice_p65_z080.npy") == 80
