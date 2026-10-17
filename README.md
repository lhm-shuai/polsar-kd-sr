# PolSAR Ship Detection by Knowledge-Distillation-Guided Symbolic Regression

Reference implementation of the IEEE Transactions manuscript

> **Interpretable Polarimetric Scattering Representation for PolSAR Ship Detection via Knowledge-Distillation-Guided Symbolic Regression**

Haomiao Liu, Sinong Quan, and Zhihao Cai.

The method distils a small multilayer-perceptron teacher into **closed-form
scattering descriptors**. A lightweight teacher is trained on pixel-level hard
labels to produce a continuous ship-membership surface, and an evolutionary
symbolic-regression student (NSGA-II, via PySR) fits that surface under an
explicit accuracy–complexity trade-off. The selected operator,

```
Pmv = exp( -c / (Pm + Pv) ),    c = 0.20430441
```

is a six-node closed-form mapping. It is strictly monotone in `Pm + Pv`, so it
has the **same AUC as the raw sum**; its value is a bounded, auditable,
unit-interval output that can replace the `Pm` channel of a detector, not a new
ranking of the pixels. The repository prints both `Pmv` and the raw sum so this
property is always visible.

> **Scope of the paper.** The detector comparison is restricted to four YOLO
> variants (YOLOv8n, YOLOv11n, YOLOv12n, YOLO26). The earlier MMDetection
> one/two-stage rows were removed because their per-cell training logs could not
> be archived; no cross-family generality is claimed. See
> `PAPER_CONSISTENCY.md`.

---

## What is in here

| Path | Paper | What it does |
|---|---|---|
| `matlab/decompose_components.m` | Sec. II-A | Yamaguchi four-component decomposition → one `.npy` per scene |
| `matlab/make_hard_labels.m` | Sec. II-A | the pixel-level BT.601 threshold rule → one mask `.png` per scene |
| `src/data.py` | Sec. III-A | pixel sampling and the scene-level split |
| `src/teacher.py` | Sec. II-A | the 3-64-64-1 teacher, 4,481 parameters |
| `src/sr.py` | Sec. II-B | the PySR search and the Pareto selection |
| `src/evaluate.py` | Sec. III-C | pixel metrics, ROC, raw-sum baseline, CFAR reference |
| `src/detector.py` | Sec. III-D | sliced YOLO dataset + single-class training |
| `results/mined_descriptor.txt` | Sec. II-B/III-B | the six-node descriptor and the C=13 reference |
| `results/paper_tables/` | Tables I–VII | the exact numbers used in the manuscript |
| `paper/` | — | the IEEE-formatted manuscript (PDF + DOCX) |

### Script → paper block

| Script | Produces |
|---|---|
| `01_teacher.py` | teacher metrics, soft labels |
| `02_search.py` | Pareto front / candidate expressions (Table II, Fig. 8) |
| `03_seeds.py` | seed-stability study (Table III) |
| `04_evaluate.py` | pixel-level comparison (Table IV, Figs. 9–10) |
| `05_detector.py` | four YOLO variants (Tables V–VII, Figs. 11–14) |
| `06_assist.py` | two-layer pixel extraction (Table VIII; auxiliary) |
| `07_operators.py` | classic-operator comparison (auxiliary; not a main table) |
| `08_transfer.py` | IPSD cross-sensor check (Table IX) with one scalar calibration |
| `09_tcr.py` | target-to-clutter ratios (auxiliary; no result file shipped in v75) |
| `10_sizes.py` | parameter/FLOP/FPS inventory (auxiliary) |

The four `ipsd_*` scripts re-derive the cross-dataset numbers under the
documented erosion / label-screening / operating-point protocols and write
their own JSON. The paper reports the **1:2 balanced IPSD** result and calls it
a cross-sensor consistency check, not a zero-shot transfer: the descriptor's
constant is tied to the power scale, so one scalar calibration is applied to
`Pm+Pv` and the expression is unchanged.

---

## Install

```bash
conda create -n kdsr python=3.9
conda activate kdsr
pip install -r requirements.txt
```

PySR needs Julia; the first run installs it automatically. The reported runs
used PySR 1.5.9 on Julia 1.11.9, with `torch` 2.8+cu128. The detector stage
also needs Ultralytics (AGPL-3.0; see *Licences*).

## Data

The datasets are **not** redistributed here.

- **PSDD** is the benchmark of Gao et al., *PSDD: A benchmark PolSAR dataset for
  CNN performance evaluation of ship detection*, IEEE AESM, 2026. It is the
  dataset used in the paper; obtain it from the authors.
- **IPSD** is the third-party set released with MaOutCNN; see the paper's
  reference list. It is used only for the cross-sensor check.

Every path a script reads or writes comes from `configs/default.yaml`; there are
no hard-coded paths in the tree. Point the `paths:` block at your copy:

```
data/mat/              coherency matrices, one .mat per scene
data/Annotations/      VOC XML, one per scene
data/四成分/            Stage I(a) output: one .npy per scene, (H, W, 3), [Pm, Pv, Ps]
data/分割舰船位置矩阵/    Stage I(b) output: one 8-bit PNG per scene, 255 = ship
data/yolo/<combination>/  tile datasets written by 05_detector.py build
data/masks/            tile-level ground-truth masks
data/IPSD/             optional third-party transfer set
work/                  everything the pipeline writes
```

```bash
python scripts/01_teacher.py
python scripts/07_operators.py --config configs/default.yaml --combination PmPvPs
```

## Reproducing the paper

**Stage I — decomposition and labels (MATLAB).** Run
`matlab/decompose_components.m`, then `matlab/make_hard_labels.m`. The labelling
rule is

```
R, G, B = uint8(round(255 * clip(Pm, Pv, Ps, 0, 1)))
L       = 0.299 R + 0.587 G + 0.114 B
ship   <=>  L >= 40,  inside an annotation box only
```

One constant, `40`, shared by all three sensors and all 23 scenes. Note the
channel order: `[Pm, Pv, Ps]` is `[R, G, B]`; the luminance weights depend on it.

**Stage II — teacher and search.**

```bash
python scripts/01_teacher.py
python scripts/02_search.py                                    # main result
python scripts/02_search.py --target hard --tag direct_sr      # Direct-SR ablation
python scripts/02_search.py --features Pm,Pv --tag dual        # feature-subset ablation
python scripts/02_search.py --operators arithmetic --tag ops_a
python scripts/02_search.py --operators with_exp   --tag ops_b
python scripts/02_search.py --operators full       --tag ops_c
python scripts/03_seeds.py                                     # Table III
```

**Pixel-level evaluation.**

```bash
python scripts/04_evaluate.py --cfar-images data/四成分
```

The output includes `Pm+Pv` next to `Pmv`; the script deliberately prints the
raw sum so the rank equivalence is auditable.

**Detectors (four YOLO variants).**

```bash
python scripts/05_detector.py build --channels Pm,Pv,Ps  --out data/yolo_PmPvPs
python scripts/05_detector.py build --channels Pmv,Pv,Ps --out data/yolo_PmvPvPs
python scripts/05_detector.py train --data-yaml configs/detector_data.yaml \
    --model yolov8n.yaml --name yolov8n_PmvPvPs
```

The manuscript's YOLO protocol is 640×640 input, batch 8, AdamW, 300 epochs
(three-channel) or 200 epochs (single-channel), single class. The released
config trains the YOLO architecture from its YAML (no COCO initialization); the
manuscript reports the released protocol and does not claim COCO initialization.
MMDetection one/two-stage models are intentionally **not** part of the paper's
main comparison; their earlier rows were removed because the final logs were not
archived.

## Licences and data availability

- Code: MIT (see `LICENSE`).
- Ultralytics: AGPL-3.0, used as an installed dependency; it is not vendored.
- PSDD and IPSD: third-party datasets, not redistributed; follow their release
  terms.
- Paper PDF/DOCX: `paper/`.

## Citation

If you use this code, please cite the IEEE manuscript. A machine-readable entry
is in `CITATION.cff`. Earlier titles/abstracts are superseded by the IEEE
manuscript in `paper/`.
