# Extended methodology and experiment log

Companion to the README. This file records what was tried, what failed, and why
the final configuration looks the way it does. Keeping a log like this is worth
the effort: without it, the reasons behind a hyperparameter are gone within a
month, and a reviewer cannot tell a deliberate choice from an accident.

---

## 1. Data exploration

### Sanity checks before anything else

Before writing a single line of model code, every volume was checked for:

- **Shape agreement** between image and mask. A mismatch means the mask does not
  describe the image and every metric computed from it is meaningless.
- **Spatial orientation** (`nibabel.orientations.aff2axcodes`). Volumes stored
  with different axis codes get sliced along the wrong anatomical plane, which
  produces sagittal or coronal slices that look plausible in a thumbnail and are
  wrong.
- **Mask binarity.** Interpolated masks contain values other than 0 and 1.
- **Lesion load.** Volumes with an empty mask are either healthy controls or a
  failed annotation, and the two need different handling.

These checks caught real problems. In the Mendeley cohort, patients 53, 54, 56
and 57 were stored with the anatomical up-down direction along a different axis
from the rest, and patients 34, 41 and 48 had annotation problems. All seven
were excluded, and the exclusion is recorded in the split configuration rather
than applied by hand.

### Why a second cohort was dropped

The first version combined two datasets: Mendeley (60 patients, FLAIR only) and
MSLesSeg (75 patients, FLAIR + T1 + T2). Combining them was attractive for
sample size, but:

1. Mendeley has **no T1 or T2**, so the multimodal model could not use it.
2. Per-cohort evaluation showed a large performance gap between the two, and the
   optimal decision threshold differed by roughly 0.5 between them — a clear
   domain shift, not noise.

Training on the union and reporting a single pooled number would have hidden
that gap behind an average. The final system uses **MSLesSeg only**, and the
domain shift is reported as an open limitation instead of being averaged away.
This is a deliberate trade of sample size for an interpretable result.

---

## 2. Preprocessing

### Normalisation

Three schemes were tried:

| Scheme | Description | Outcome |
|---|---|---|
| Whole-volume min–max, unclipped | `(v - v.min()) / (v.max() - v.min())` | **Used in the thesis** (`legacy_minmax`) |
| Brain-only min–max, percentile-clipped | Clip at 0.5/99.5, scale brain to `[0, 1]` | **Current default** (`minmax`) |
| Brain-only z-score | Standardise, clip at ±3σ, rescale | Comparable to the default |
| Per-slice z-score | Same, computed slice by slice | **Rejected** |

The move from the first row to the second is the one place where the refactored
code departs numerically from the thesis. The original computed min and max over
the **entire volume including background** and applied no outlier clipping, so a
single bright voxel near the skull compresses the whole brain into a narrow band
of `[0, 1]`. On a synthetic volume with one such voxel, the mean brain intensity
drops from 0.50 under the clipped scheme to 0.05 under the original — a tenfold
difference in the dynamic range the network actually sees.

Both are available. `legacy_minmax` reproduces the thesis exactly and is what an
existing checkpoint must be evaluated with; the clipped version is the better
default for new runs. A checkpoint is **not** transferable between them.

Per-slice normalisation was rejected on principle rather than on a metric: it
destroys the intensity of a slice *relative to its own volume*, and that relative
hyperintensity is precisely what identifies a lesion on FLAIR. A slice full of
lesions and a slice with none get mapped to the same range, deleting the signal.

Both volume-level schemes clip at the 0.5th and 99.5th percentiles before
computing statistics. Without clipping, a single bright voxel near the skull
compresses the entire brain into a narrow band of the output range.

### Slice geometry

MSLesSeg is distributed in MNI152 1mm space, so every volume is exactly
182 × 218 × 182. Neither in-plane dimension is divisible by 16, which four
max-pooling levels require, so each slice has to be brought to a compatible
size. Two ways were implemented:

| | Operation | Lesion voxels | Anatomy |
|---|---|---|---|
| **`pad`** | centre in a 192 × 224 canvas | unchanged | undeformed |
| `resize` | stretch to 224 × 192 | changed by ~1% | stretched |

**Padding is the default and is what the thesis used.** It leaves every original
pixel untouched: no interpolation, no deformation, and the ground truth keeps
exactly the voxel count the annotator drew. The margin is 5 rows top and bottom,
3 columns left and right — a 2.7% border of zeros, which costs a negligible
amount of capacity and buys exactness.

Resizing remains available for data that does not fit the canvas. Under it,
images use bilinear interpolation and **masks use nearest neighbour**; bilinear
on a binary mask produces fractional values at lesion borders, quietly
redefining the ground truth.

> The two are not interchangeable for a trained model. Evaluating
> padding-trained weights on stretched slices cost about 0.05 Dice here, with no
> error raised. The tell is the **ground-truth lesion voxel count**: it changes
> with the geometry but not with the model or the threshold, so comparing it
> against a previous run localises this class of mismatch immediately.

---

## 3. Splits and class imbalance

### Patient-level splitting

Non-negotiable. Adjacent axial slices of one brain are near-duplicates.
Slice-level random splitting leaks anatomy across the train/test boundary and
inflates Dice substantially. Enforced by `SplitConfig.validate()` and covered by
a test.

### Handling the negatives

Roughly 70% of axial slices contain no lesion at all, and within a positive slice
lesions occupy under 1% of pixels. Two filters, **on training only**:

1. Drop slices whose brain fraction is below 10% — mostly neck and empty space
   above the head.
2. Keep 30% of lesion-free slices, sampled with a fixed seed.

Validation and test keep every slice. Filtering the test set would make the
reported numbers describe an easier problem than the real one.

An adaptive variant of filter 1, using the 20th percentile of non-zero
intensities as the threshold rather than a fixed 0.05, was tried. It behaves
better on volumes with unusual intensity distributions but adds a parameter for a
marginal gain; the fixed threshold was kept for simplicity.

### Manifests instead of copying

The first version physically copied `.npy` files into `train/`, `val/` and
`test/` folders — several gigabytes duplicated, and the exact composition of a
split impossible to recover afterwards. The current pipeline writes a **CSV
manifest** instead: a few hundred kilobytes, version-controlled, diffable, and
enough to reconstruct any split exactly.

---

## 4. Model

### Architecture search

| Variant | Note |
|---|---|
| `base_channels` 16 / 32 / 64 | 32 chosen; 64 gave no gain and doubled memory |
| BatchNorm vs GroupNorm | GroupNorm; BatchNorm unstable at batch size 8 |
| With / without SE | SE kept; see the ablation table in the README |
| 4 vs 5 downsampling levels | 4; at 5 the bottleneck is 14 × 12 and loses detail |

### Loss

`0.7 × BCE(pos_weight=3) + 0.3 × (1 − soft Dice)`.

Plain BCE collapses to the all-background solution. Soft Dice alone trains but is
unstable early, when predictions are near-uniform and the Dice gradient is
poorly conditioned. The mixture gets BCE's stable early gradient and Dice's
resistance to imbalance.

`pos_weight` was swept over {1, 3, 5}. Larger values raise recall and cost more
precision; 3 sat best on the validation F1.

A focal-loss variant with a γ ramped from 0 to 2 over the first six epochs was
implemented and tested. It did not beat BCE-Dice here and added two
hyperparameters, so it was dropped.

---

## 5. Threshold and post-processing

### Threshold selection

The decision threshold is a genuine hyperparameter and moves Dice by several
points. It is swept over `0.05 … 0.95` on the **validation** split after every
epoch, and the best value is stored in the checkpoint.

The sweep optimises F-β. Setting β < 1 favours precision and yields a higher
threshold; the default β = 1 (ordinary F1) is what the reported results use. For
a screening application where a missed lesion is worse than a false alarm, β > 1
is the defensible choice — this is a clinical decision, not a statistical one.

### Small-component removal

Predicted connected components under 10 pixels (8-connectivity) are removed.
These are mostly isolated activations at the grey/white matter boundary. The
filter costs a little recall and returns more precision.

### Axial consistency (tried, not adopted)

A post-processing rule was implemented that keeps a prediction at slice *z* only
if the neighbouring slices *z−1* or *z+1* also predict a lesion at the same
location above a lower support threshold, unless the central prediction is very
confident. The intuition is sound: real lesions are 3D and span several slices,
whereas false positives tend to be isolated.

It reduced false positives but cost three forward passes per slice and did not
beat plain small-component removal by enough to justify the complexity. The code
path is documented here rather than kept in the main pipeline.

---

## 6. What a next iteration should address

In rough order of expected value:

1. **Cross-validation over patients** instead of one fixed split. With 75
   patients, an 11-patient test set gives a wide confidence interval, and the
   per-patient standard deviation already shows this.
2. **A published baseline.** Dice 0.75 means little without nnU-Net or a
   comparable reference trained on the same split.
3. **Multiple seeds**, with mean and standard deviation, so ablation gaps can be
   distinguished from run-to-run variation.
4. **3D or 2.5D-with-3D-loss**, to use through-plane context the 2D model discards.
5. **Lesion-level metrics** (detection rate per lesion, false positives per
   patient) alongside voxel-level ones. A clinician cares whether a lesion was
   found, not how many of its voxels were.
6. **External validation** on a second cohort, reported per cohort.
