"""MS lesion segmentation in brain MRI with a 2D U-Net.

Public entry points:

* :mod:`msseg.data.preprocess` -- NIfTI volumes to normalised 2D slices
* :mod:`msseg.data.splits`     -- patient-level train/val/test manifests
* :mod:`msseg.train`           -- training loop
* :mod:`msseg.evaluate`        -- metrics on a held-out split
* :mod:`msseg.figures`         -- report figures
"""

__version__ = "1.0.0"
