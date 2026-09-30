# Data

This folder is intentionally almost empty. Medical imaging data is not committed
to version control: the volumes are large, they are redistributed under their own
terms, and even de-identified scans are governed by the data use agreement of the
original release.

## What is tracked

- `manifests/` — CSV files listing which slices belong to which split. These are
  small, diffable, and are what makes an experiment reproducible: given the raw
  data and a manifest, anyone can rebuild the exact splits used for a result.

## What is not tracked

- `raw/` — the downloaded NIfTI volumes
- `processed/` — the `.npy` slices produced by `msseg.data.preprocess`

## Getting the data

Download **MSLesSeg** from https://iplab.dmi.unict.it/mfs/ms-les-seg/ and unpack
it into `data/raw/` so that the layout is:

```
data/raw/
├── P1/T1/P1_T1_FLAIR.nii.gz
│        P1_T1_T1.nii.gz
│        P1_T1_T2.nii.gz
│        P1_T1_MASK.nii.gz
├── P2/T1/...
└── ...
```

Then run the preprocessing step from the README.

## Excluded patients

None in the current configuration. If a patient is excluded, record it in the
`excluded_patients` field of the split config and state the reason here, so the
exclusion is part of the repository rather than folklore.
