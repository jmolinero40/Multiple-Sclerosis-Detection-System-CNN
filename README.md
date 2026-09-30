# Multiple Sclerosis Lesion Segmentation in Brain MRI

A 2D U-Net with Squeeze-and-Excitation blocks that segments multiple sclerosis
lesions from co-registered FLAIR, T1 and T2 brain MRI. Bachelor's thesis in
Mathematics and Statistics, Complutense University of Madrid — graded 9/10.
Predictions were reviewed by a practising neurologist.

![Qualitative results](docs/images/qualitative_panel.png)
*Test-set predictions for three patients: the best case (P69), the hardest
(P68, which carries only 919 lesion voxels) and the heaviest lesion load (P72).
Left to right: FLAIR, ground truth, prediction, and the error overlay with false
positives in red and false negatives in blue.*

---

## Results

Evaluated on **11 held-out patients** (MSLesSeg P65–P75) that appear in no
other split. The decision threshold (0.70) and the minimum connected-component
size (10 px) were fixed before evaluation, not tuned on test.

| Metric | Value |
|---|---|
| Dice = F1, voxel level | **0.72** |
| Dice, mean per slice | 0.75 |
| Dice, per patient (mean ± sd) | 0.66 ± 0.14 |
| Dice, lesion-bearing slices only | 0.58 |
| Precision / Recall, voxel level | 0.71 / 0.72 |
| IoU, voxel level | 0.56 |

These reproduce the original thesis figures: it reported 0.718 voxel-level F1
and 0.750 mean per-slice Dice on this split, against 0.7185 and 0.7493 here.

> The three Dice figures above are the same predictions aggregated three ways,
> and the differences between them are informative rather than noise. The
> per-slice mean is raised by the 1103 lesion-free slices, where predicting
> nothing against nothing scores 1.0 by convention; restricted to slices that
> contain a lesion it is 0.58. The per-patient mean weights each patient
> equally and is the figure most comparable to the MS lesion segmentation
> literature. See [`src/msseg/metrics.py`](src/msseg/metrics.py).

### Ablation: what the SE blocks buy

| Input | SE blocks | Precision | Recall | Dice (voxel) | Dice (per patient) |
|---|---|---|---|---|---|
| FLAIR + T1 + T2 | no | 0.720 | 0.658 | 0.688 | 0.634 ± 0.143 |
| **FLAIR + T1 + T2** | **yes** | 0.714 | **0.723** | **0.718** | **0.664 ± 0.142** |

Identical architecture, data, patient split and decision threshold; the only
difference is the channel-attention gate.

The gain is almost entirely in **recall**: +0.065, against −0.006 in precision.
Channel attention does not make the network more cautious, it makes it find
lesions it was otherwise missing. With the modality axis as the channel axis,
SE learns to reweight FLAIR, T1 and T2 per slice, which is what recovers lesions
that FLAIR alone reads ambiguously.

> Both models were trained once, so the gap is indicative rather than a
> significance claim. The two runs also come from separate training sessions
> whose schedules were not verified identical, so the comparison isolates the
> architecture but not every confound.

### Reproducing the thesis exactly

`configs/multimodal.yaml` is the *improved* configuration. The thesis itself was
run with slightly different settings, preserved in
[`configs/thesis_reproduction.yaml`](configs/thesis_reproduction.yaml):

| | Thesis | Default here |
|---|---|---|
| Intensity normalisation | whole-volume min–max, no clipping | brain-only, percentile-clipped |
| Best-epoch criterion | soft Dice on validation | F1 at the swept threshold |
| Decision threshold | fixed at 0.70 | swept on validation each epoch |
| Epochs / patience | 15 / 6 | 40 / 8 |
| Mixed precision | off | on |

Slice geometry is **the same in both** (centre-padding to 192×224) and is not a
knob to turn casually: see the note below.

```bash
python -m msseg.data.preprocess --raw-root data/raw \
    --out-root data/processed_legacy \
    --geometry pad --height 192 --width 224 \
    --normalisation legacy_minmax
python -m msseg.data.splits --processed-root data/processed_legacy \
    --out data/manifests/splits_legacy.csv --seed 0
python -m msseg.train --config configs/thesis_reproduction.yaml
```

The network itself is **identical** in both: same modules, same 7,819,133
parameters, same `state_dict` keys, so a checkpoint trained with the original
scripts loads into this code unchanged. That equivalence, along with the loss
and the legacy normalisation, is asserted in
[`tests/test_thesis_equivalence.py`](tests/test_thesis_equivalence.py) and
checked on every commit.

> **A checkpoint is bound to its preprocessing, in two independent ways.**
> Weights trained on `legacy_minmax` intensities score poorly on `minmax` data,
> and weights trained on padded slices score poorly on stretched ones. Both
> failures are silent: no error is raised, the metrics simply come out a few
> points low. If a reproduction lands close but not on the mark, compare the
> **ground-truth lesion voxel count** first — it depends on the geometry but not
> on the model or the threshold, so a mismatch there localises the problem
> immediately.

---

## Method

### Data

[**MSLesSeg**](https://iplab.dmi.unict.it/mfs/ms-les-seg/) — 75 patients with
co-registered FLAIR, T1 and T2 volumes and expert consensus lesion masks,
distributed in MNI152 1mm space, so every volume is 182×218×182: 182 axial
slices of 182×218. Each slice is centre-padded to 192×224, the next multiple of
16 in each direction, which is what the four-level U-Net requires.

| Split | Patients | Role |
|---|---|---|
| Train | P1–P53 | Fitting |
| Validation | P54–P64 | Threshold, early stopping, model selection |
| Test | P65–P75 | Reported once, at the end |

Splits are **by patient, never by slice**. Adjacent axial slices of one brain are
near-identical; shuffling slices across splits puts copies of the same anatomy on
both sides of the boundary and turns the reported Dice into an interpolation
score. This is the most common way an MRI segmentation result is quietly
inflated, and the repository has a test that fails if the splits ever overlap
([`tests/test_data.py`](tests/test_data.py)).

### Pipeline

```
NIfTI volumes
     |
     |  msseg.data.preprocess   per-volume normalisation inside the brain,
     |                          axial slicing, centre-pad to 192x224
     v
.npy slices + masks
     |
     |  msseg.data.splits       patient-level manifest; train-only filters
     |                          (drop near-empty slices, subsample negatives)
     v
manifest.csv
     |
     |  msseg.train             U-Net + SE, BCE-Dice loss,
     |                          threshold swept on validation each epoch
     v
best.pt  (weights + config + threshold, all in one file)
     |
     |  msseg.evaluate          metrics at three aggregation levels
     |  msseg.figures           confusion matrix, per-patient spread, error panel
     v
results/ + docs/images/
```

### Design decisions

**Three modalities as channels.** FLAIR shows lesions as hyperintense and is the
primary signal; T1 shows them as hypointense, which separates true lesions from
FLAIR artefacts; T2 is sensitive but less specific. Stacking them lets the first
convolution see all three at once rather than fusing late.

**Squeeze-and-Excitation, not spatial attention.** In this setup the modality
axis *is* the channel axis, so a learned per-channel gate lets the network weight
FLAIR, T1 and T2 differently depending on slice content. That is exactly what SE
does, and it costs about 3% extra parameters.

**GroupNorm, not BatchNorm.** Training runs at batch size 8 on one consumer GPU.
BatchNorm statistics are noisy at that size, and with lesions occupying well under
1% of pixels the running statistics are dominated by background. GroupNorm is
batch-independent and was measurably more stable here.

**BCE + soft Dice, with a positive class weight.** Plain BCE converges to
predicting background everywhere: >99% pixel accuracy, Dice 0. Soft Dice is
scale-invariant in the size of the foreground so it does not vanish for small
lesions; `pos_weight=3.0` inside the BCE term adds recall pressure. The mix is
`0.7 × BCE + 0.3 × (1 − Dice)`, chosen on validation.

**Decision threshold chosen on validation, never on test.** Swept every epoch and
stored inside the checkpoint, so evaluation cannot silently tune it. Passing
`--threshold` to the evaluation script prints a warning for this reason.

**Connected components under 10 pixels are dropped.** The network fires on
isolated pixels at the grey/white matter boundary. A lesion that small is below
what a radiologist would annotate, so removing them trades a little recall for
more precision.

**Filters apply to training only.** Training drops near-empty slices and keeps
30% of lesion-free slices, since roughly 70% of axial slices contain no lesion.
Validation and test keep **every** slice: the evaluation distribution has to match
what the model would see in deployment, where nobody pre-filters its input.

### Known limitations

- **Single cohort.** All 75 patients come from one dataset, so the results do not
  measure robustness to a scanner or protocol the model has not seen. An earlier
  version of this work included a second cohort (Mendeley) and performance
  dropped substantially across sources; that domain gap is the honest open
  problem here, not a solved one.
- **2D, not 3D.** Each slice is segmented independently. A 3D U-Net or through-plane
  consistency post-processing would use anatomy the current model ignores.
- **No inter-rater variability.** The ground truth is one consensus annotation, so
  a Dice of 0.75 cannot be compared against the human ceiling, which for MS lesion
  segmentation is itself typically in the 0.7–0.8 range.
- **Single seed.** Reported numbers are one run. The per-patient spread is wide
  enough that seed variation matters.

---

## Getting started

### Install

```bash
git clone https://github.com/jmolinero40/Multiple-Sclerosis-Detection-System-CNN.git
cd Multiple-Sclerosis-Detection-System-CNN

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Requires Python ≥ 3.10. A GPU is recommended but not required; the code falls
back to CPU.

### Get the data

The dataset is **not** in this repository, and must not be: it is large and is
distributed under its own terms. Download MSLesSeg from
[the project page](https://iplab.dmi.unict.it/mfs/ms-les-seg/) and unpack it so
the layout is:

```
data/raw/
├── P1/T1/P1_T1_FLAIR.nii.gz
│        P1_T1_T1.nii.gz
│        P1_T1_T2.nii.gz
│        P1_T1_MASK.nii.gz
├── P2/T1/...
└── ...
```

The `T1` folder and the middle `T1` in each filename denote **timepoint 1**;
the trailing token is the modality. So `P1_T1_T2.nii.gz` is "patient 1,
timepoint 1, T2-weighted".

### Run

```bash
# 1. NIfTI volumes -> normalised 2D slices
python -m msseg.data.preprocess \
    --raw-root data/raw --out-root data/processed --patients 1 75

# 2. Patient-level split manifest
python -m msseg.data.splits \
    --processed-root data/processed --out data/manifests/splits.csv

# 3. Train
python -m msseg.train --config configs/multimodal.yaml

# 4. Evaluate on the held-out test patients
python -m msseg.evaluate \
    --checkpoint runs/multimodal/best.pt --split test \
    --output-dir results/multimodal

# 5. Figures
python -m msseg.figures \
    --checkpoint runs/multimodal/best.pt \
    --results results/multimodal/metrics.json \
    --output-dir docs/images \
    --panel mslesseg_p68:80 mslesseg_p70:auto mslesseg_p75:110
```

Each step is also available as a console script (`msseg-preprocess`,
`msseg-train`, …) after installation.

### Tests

```bash
pytest                 # 39 tests, a few seconds, no dataset required
ruff check src tests
```

The suite builds small synthetic volumes in a temporary directory, so it runs
anywhere, including CI.

---

## Repository layout

```
├── configs/                 experiment configs (one YAML per run)
├── data/
│   ├── manifests/           split definitions — tracked, small, diffable
│   └── README.md            how to obtain the data
├── docs/
│   ├── METHOD.md            extended methodology and experiment log
│   └── images/              generated figures
├── src/msseg/
│   ├── config.py            typed config loaded from YAML
│   ├── data/
│   │   ├── naming.py        the filename convention, in one place
│   │   ├── preprocess.py    stage 1: NIfTI -> .npy slices
│   │   ├── splits.py        stage 2: patient-level manifests
│   │   └── dataset.py       stage 3: Dataset + augmentation
│   ├── models/unet.py       U-Net with SE blocks
│   ├── losses.py            BCE + soft Dice
│   ├── metrics.py           metrics at three aggregation levels
│   ├── postprocess.py       thresholding, small-component removal
│   ├── train.py             training loop
│   ├── evaluate.py          held-out evaluation
│   └── figures.py           report figures
└── tests/                   39 tests over synthetic data
```

---

## Citation

If this work is useful to you:

```bibtex
@mastersthesis{molinero2026ms,
  author = {Molinero Araguas, Javier},
  title  = {Multiple Sclerosis Lesion Segmentation in Brain MRI
            with Convolutional Neural Networks},
  school = {Complutense University of Madrid},
  year   = {2026}
}
```

Please also cite the MSLesSeg dataset authors if you use the data.

## Licence

Code released under the [MIT Licence](LICENSE). The dataset is **not** covered by
this licence and remains subject to its own terms.

## Acknowledgements

Thesis supervision at the Faculty of Mathematics, Complutense University of
Madrid. Clinical review of the predictions by a practising neurologist.
