# Feedback-Learning-Based Polarimetric Scattering Representation for PolSAR Ship Detection

Code and result tables for the paper by Haomiao Liu, Sinong Quan, and Zhihao Cai.

The pipeline has three stages. A small multilayer-perceptron teacher is trained
on pixel-level hard labels and outputs a continuous ship-membership map. An
evolutionary symbolic-regression search (NSGA-II, through PySR) fits that map
under an explicit accuracy–complexity trade-off. The selected low-complexity
descriptor is

```
Pmv = exp(-c / (Pm + Pv)),    c = 0.20430441
```

a six-node closed form. Its value is a bounded, auditable, unit-interval
membership score that replaces the `Pm` channel of a detector. The
complexity-29 descriptor, reported as `Psmv`, is included as the high-complexity
counterpart. The detector experiments use eight COCO-pretrained networks:
YOLOv8, YOLOv11, YOLOv12, YOLO26, RetinaNet, Faster R-CNN, SSD, and PAA.

The paper compares these networks with the baseline `[Pm, Pv, Ps]` input and
with the descriptor-based input. It also reports pixel-level discrimination,
target-to-clutter ratio, zero-training transfer to IPSD, the soft-label and
operator-set ablations, and a two-layer decision that refines detector boxes
with the explicit descriptor. The repository does not claim generality beyond
the eight architectures tested.

## Repository layout

| Path | Paper block | Contents |
|---|---|---|
| `matlab/decompose_components.m` | Sec. III-A | Yamaguchi four-component decomposition, one `.npy` per scene |
| `matlab/make_hard_labels.m` | Sec. III-A | BT.601 luminance rule, one pixel mask per scene |
| `src/data.py` | Sec. III-A | pixel sampling and the scene-level split |
| `src/teacher.py` | Sec. II-A | 3-64-64-1 teacher, 4,481 parameters |
| `src/sr.py` | Sec. II-B | PySR search and Pareto selection |
| `src/evaluate.py` | Sec. III-C | pixel metrics, ROC, raw-sum baseline, CFAR reference |
| `src/detector.py` | Sec. III-D | sliced YOLO dataset and single-class training |
| `src/assist.py` | Sec. II-C/III-E | two-step in-box descriptor decision |
| `results/mined_descriptor.txt` | Sec. II-B/III-B | the complexity-6 and complexity-29 descriptors |
| `results/paper_tables/` | Tables III–VII | table sources; mapping below |

## Scripts and paper items

| Script | Produces |
|---|---|
| `01_teacher.py` | teacher metrics and soft labels (Fig. 6; input to Table III) |
| `02_search.py` | Pareto front and candidate descriptors (Table III, Fig. 7); operator-set ablation (Fig. 15) |
| `03_seeds.py` | five-seed stability (Fig. 7) |
| `04_evaluate.py` | pixel-level comparison on PSDD (Table IV, Fig. 9(a)) |
| `05_detector.py` | YOLO datasets and training (Tables V–VI, Figs. 10–12) |
| `06_assist.py` | two-layer box refinement (Table VII, Fig. 13) |
| `07_operators.py` | conventional-operator rows of Table IV and Fig. 9 |
| `08_transfer.py` | IPSD columns of Table IV and Fig. 9(b), with one scalar calibration |
| `09_tcr.py` | target-to-clutter ratios cited in Sec. III-C (no table in the paper) |
| `10_sizes.py` | parameter, FLOP, and FPS inventory (auxiliary) |

The `ipsd_*` scripts re-derive the cross-dataset numbers under the
documented erosion, label-screening, and operating-point protocols and write
their own JSON. The paper reports the 1:2 balanced IPSD columns and describes
them as zero-training transfer: the descriptor constant is tied to the power
scale, so one scalar calibration is applied to `Pm + Pv` and the expression is
unchanged.

## Install

```bash
conda create -n kdsr python=3.9
conda activate kdsr
pip install -r requirements.txt
```

PySR needs Julia; the first run installs it automatically. The reported runs
used PySR 1.5.9 on Julia 1.11.9 with `torch` 2.8+cu128. The detector stage
uses Ultralytics (AGPL-3.0; see *Licences*).

## Data

The datasets are not redistributed here.

- **PSDD** is the benchmark of Gao et al., *PSDD: A benchmark PolSAR dataset for
  CNN performance evaluation of ship detection*, IEEE AESM, 2026. It contains
  23 scenes, 1,357 patches, and 2,577 ship boxes from GF-3, AIRSAR, and
  Radarsat-2.
- **IPSD** is the third-party set released with MaOutCNN. It contains 934
  images of 256×256 pixels and is used only for the cross-dataset check.

Every path a script reads or writes comes from `configs/default.yaml`. Point the
`paths:` block at your copy:

```
data/mat/                 coherency matrices, one .mat per scene
data/Annotations/         VOC XML, one per scene
data/四成分/               Stage I(a) output: one .npy per scene, [Pm, Pv, Ps]
data/分割舰船位置矩阵/       Stage I(b) output: one 8-bit mask per scene
data/yolo/<combination>/  tile datasets written by 05_detector.py build
data/masks/               tile-level ground-truth masks
data/IPSD/                IPSD transfer set
work/                     pipeline output
```

The auxiliary `ipsd_*` scripts read `PSDD_ROOT` and `IPSD_ROOT`, defaulting to
`data/PSDDv1.0` and `data/IPSD`.

## Reproducing the paper

**Stage I: decomposition and labels (MATLAB).** Run
`matlab/decompose_components.m`, then `matlab/make_hard_labels.m`. The labelling
rule is

```
R, G, B = uint8(round(255 * clip(Pm, Pv, Ps, 0, 1)))
L       = 0.299 R + 0.587 G + 0.114 B
ship   <=>  L >= 40,  inside an annotation box only
```

The threshold 40 is shared by all three sensors and all 23 scenes. The channel
order is `[Pm, Pv, Ps] = [R, G, B]`; the luminance weights depend on it.

**Stage II: teacher and search.**

```bash
python scripts/01_teacher.py
python scripts/02_search.py                                    # main result
python scripts/02_search.py --target hard --tag direct_sr      # Direct-SR ablation
python scripts/02_search.py --features Pm,Pv --tag dual        # feature-subset ablation
python scripts/02_search.py --operators arithmetic --tag ops_a
python scripts/02_search.py --operators with_exp   --tag ops_b
python scripts/02_search.py --operators full       --tag ops_c
python scripts/03_seeds.py                                     # Fig. 7 stability
```

**Pixel-level evaluation.**

```bash
python scripts/04_evaluate.py --cfar-images data/四成分
```

The output includes `Pm + Pv` next to `Pmv`, so the effect of the monotone
mapping can be checked directly.

**Detectors.** The YOLO family is trained from COCO-pretrained weights with
640×640 input, batch 8, AdamW, and 300 epochs for three-channel inputs or 200
epochs for single-channel inputs. The learning rates are 0.002 and 0.0005.
The MMDetection networks use the official configurations, SGD with AMP, batch
4, and 12 epochs. Their metrics are in
`results/paper_tables/mmdet_revalidation.csv`.

```bash
python scripts/05_detector.py build --channels Pm,Pv,Ps  --out data/yolo_PmPvPs
python scripts/05_detector.py build --channels Pmv,Pv,Ps --out data/yolo_PmvPvPs
python scripts/05_detector.py train --data-yaml configs/detector_data.yaml \
    --model yolov8n.yaml --name yolov8n_PmvPvPs
```

**Two-layer refinement.**

```bash
python scripts/06_assist.py --weights runs/detect/pmv/weights/best.pt --split test
```

The generic layer applies a high threshold to find ship seeds, grows them into
the low-threshold descriptor region along the 8-neighborhood, removes connected
components below the area floor, and tightens the box to the remaining pixels.

## Result tables

File names keep their original numbering; the paper numbering below is the
authoritative one.

| File | Paper item | Notes |
|---|---|---|
| `table_ii_pareto.csv` | Table III, Fig. 7 | seed-666 Pareto front; complexity 6 is `Pmv`, complexity 29 is `Psmv` |
| `table_iii_seeds.csv` | Fig. 7 | five official seeds |
| `table_iv_pixel.csv` | Table IV | 12 methods on PSDD and IPSD |
| `table_v_single_channel.csv` | Table V | YOLOv8 single-channel inputs |
| `table_vi_three_channel.csv` | Table V | YOLOv8 three-channel combinations |
| `table_vii_yolo.csv` | Table VI | eight detectors, baseline and descriptor input |
| `table_viii_two_layer.csv` | Table VII | three two-layer schemes reported in the paper |
| `table_ipsd_balanced.json` | Table IV, Fig. 9(b) | IPSD balanced-set scores and the scalar calibration |
| `detector_revalidation.csv` | source | YOLO runs, including the raw-sum row |
| `mmdet_revalidation.csv` | source | RetinaNet, Faster R-CNN, SSD, and PAA runs |

## Licences and data availability

- Code: MIT (see `LICENSE`).
- Ultralytics: AGPL-3.0, used as an installed dependency and not vendored.
- PSDD and IPSD: third-party datasets, not redistributed; follow their release
  terms.

## Citation

If you use this code, cite the paper. A machine-readable entry is in
`CITATION.cff`; journal volume and page numbers will be added after publication.
