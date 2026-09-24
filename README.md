# PolSAR Ship Detection by Knowledge-Distillation-Guided Symbolic Regression

Reference implementation of

> **Feedback Learning-Based Polarimetric Scattering Representation for PolSAR Ship Detection**

The method distils a small multilayer-perceptron teacher into **closed-form
scattering descriptors**. A lightweight teacher is trained on pixel-level hard
labels to produce a continuous ship-membership surface, and an evolutionary
symbolic-regression student (NSGA-II, via PySR) fits that surface under an
explicit accuracy–complexity trade-off. The selected operator,

```
Pmv = exp( -c / (Pm + Pv) ),    c = 0.20430441
```

is six nodes deep, costs three floating-point operations per pixel, and can be
dropped into an existing detector as a replacement for the `Pm` channel.

---

## What is in here

| Path | Paper | What it does |
|---|---|---|
| `matlab/decompose_components.m` | Sec. II-A | Yamaguchi four-component decomposition → one `.npy` per scene |
| `matlab/make_hard_labels.m` | Sec. II-A, Eqs. (7)-(8) | the pixel-level labelling rule → one mask `.png` per scene |
| `src/data.py` | Sec. III-A | pixel sampling and the scene-level split |
| `src/teacher.py` | Sec. II-B | the 3-64-64-1 teacher, 4,481 parameters |
| `src/sr.py` | Sec. II-C | the PySR search, all four variants used in the paper |
| `src/evaluate.py` | Sec. III-C | pixel-level metrics, ROC, CFAR reference |
| `src/detector.py` | Sec. III-A/D | sliced dataset + Ultralytics training |
| `results/mined_descriptor.txt` | Sec. III-B | the two mined expressions |

Each script under `scripts/` corresponds to one block of results:

| Script | Produces |
|---|---|
| `01_teacher.py` | teacher metrics (Sec. III-B), soft labels for Stage III |
| `02_search.py` | the Pareto front (Fig. 7), Table I |
| `03_seeds.py` | the five-seed stability study, Table II |
| `04_evaluate.py` | Table III, ROC figures |
| `05_detector.py` | Tables IV–VI |

---

## Install

```bash
conda create -n kdsr python=3.9
conda activate kdsr
pip install -r requirements.txt
```

PySR needs Julia; the first run installs it automatically (a few minutes). The
experiments reported in the paper used PySR 1.5.9 on Julia 1.11.9, with
`pysr` 1.5.9, `torch` 2.8+cu128.

The detector stage additionally needs the Ultralytics package — see *Licences*
below.

## Data

The dataset is **not** redistributed here. PSDD, the benchmark used in the paper,
is described in

> Z. Yu et al., *PSDD: A benchmark PolSAR dataset for CNN performance* (see the
> paper's reference list).

You need three directories:

```
data/mat/            coherency-matrix .mat files, one per scene
data/Annotations/    VOC-style XML, one per scene, <object><name>ship</name>
data/四成分/          <- written by Stage I(a)
data/分割舰船位置矩阵/  <- written by Stage I(b)
```

Then edit the `paths:` block at the top of `configs/default.yaml`.

## Reproducing the paper

**Stage I — decomposition and labels (MATLAB).** Run
`matlab/decompose_components.m`, then `matlab/make_hard_labels.m`. The labelling
rule is

```
R, G, B = uint8(round(255 * clip(Pm, Pv, Ps, 0, 1)))    % MATLAB rounds
L       = 0.299 R + 0.587 G + 0.114 B                   % ITU-R BT.601
ship   <=>  L >= 40,  inside an annotation box only
```

One constant, `40`, shared by all three sensors and all 23 scenes. This reproduces
the released masks *pixel for pixel* on all 23 scenes. Note the channel order:
`[Pm, Pv, Ps]` is `[R, G, B]`, and the luminance weights depend on it.

**Stage II — teacher.**

```bash
python scripts/01_teacher.py
```

**Stage III — search.**

```bash
python scripts/02_search.py                                    # main result
python scripts/02_search.py --target hard --tag direct_sr      # Direct-SR ablation
python scripts/02_search.py --features Pm,Pv --tag dual        # feature-subset ablation
python scripts/02_search.py --operators arithmetic --tag ops_a # Fig. 14
python scripts/02_search.py --operators with_exp   --tag ops_b
python scripts/02_search.py --operators full       --tag ops_c
python scripts/03_seeds.py                                     # Table II
```

**Pixel-level evaluation.**

```bash
python scripts/04_evaluate.py --cfar-images data/四成分
```

**Detectors.**

```bash
python scripts/05_detector.py build --channels Pm,Pv,Ps  --out data/yolo_PmPvPs
python scripts/05_detector.py build --channels Pmv,Pv,Ps --out data/yolo_PmvPvPs
python scripts/05_detector.py train --data-yaml configs/detector_data.yaml \
    --model yolov8n.yaml --name yolov8n_PmvPvPs
```

---

## Three things to know before you compare numbers

**1. The low-complexity descriptor has the same AUC as the raw sum.**
`exp(-c/(Pm+Pv))` is strictly monotone in `Pm + Pv`, so it ranks pixels
identically. `scripts/04_evaluate.py` therefore prints a `Pm+Pv` row next to
`Pmv`. The contribution of the descriptor is that it is closed form, costs three
operations, and saturates like a membership probability — not that it separates
the classes better. Quoting an AUC gain over `Pm + Pv` would be wrong; the paper
does not claim one.

**2. The teacher consumes the features in a different order than the rendering.**
The `.npy` files store `[Pm, Pv, Ps]`. The luminance step needs exactly that
order. The teacher, however, was trained on `(Ps, Pm, Pv)` — an arbitrary choice
that does not affect what the network can represent, but which shifts the
reproduced numbers slightly if you permute it. `src/data.py` handles both
orders explicitly.

**3. Background pixels are the whole image minus the labelled ships.** The mask
is written only inside annotation boxes, so `mask == 0` means "outside every box,
plus the sub-threshold pixels inside one". The sampling in `src/data.py` relies
on this, and it is why the background class is defined by the annotations rather
than by the threshold.

## Search settings

Taken from the archived `PySRRegressor` checkpoint of the run behind Fig. 7 and
Table I, not retyped from the manuscript:

```
niterations 300 · populations 100 · population_size 27 · maxsize 30
binary  + - * /          unary  square sqrt log exp sin cos
constraints  {log: 5, sqrt: 5, exp: 5, "/": (-1, 5)}
elementwise_loss L2DistLoss() · batching with batch_size 500 · procs 8
```

The logarithm is in the operator set but is selected in none of the ten archived
fronts, so no reported expression contains it.

## Licences

- **This repository** is MIT (see `LICENSE`).
- **Ultralytics** (YOLO training) is **AGPL-3.0**. It is used here as an
  installed dependency, not vendored; if you copy Ultralytics source into your
  own tree, the AGPL applies to your tree.
- **`matlab/yamaguchi_4components_T3.m`** implements the published Yamaguchi
  four-component decomposition; the method is cited in the file header.
- **PSDD** has its own terms. Do not redistribute the dataset from this
  repository.

## Citation

```bibtex
@article{liu_polsar_kdsr,
  title  = {Feedback Learning-Based Polarimetric Scattering Representation
            for PolSAR Ship Detection},
  author = {Liu, Haomiao and Quan, Sinong and Cai, Zhihao and Xing, Shiqi and
            Li, Yongzhen},
  note   = {Preprint}
}
```

See `CITATION.cff` for a machine-readable copy.
