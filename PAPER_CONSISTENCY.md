# Paper ↔ repository consistency (v75)

This repository is aligned with the IEEE manuscript
`paper/PolSAR_paper_English_v75_IEEE.pdf` (14 pages, 2026-10-17). The mapping
below is the one to use when checking a number, a table or a figure.

## 1. What the paper claims

The figure set is now the Chinese v93 set 1–16, including the combined PR figure (English Fig. 13) and the two-layer comparison (English Fig. 14). The English manuscript has 14 pages and nine tables.

| Paper item | Value / scope | Repository source |
|---|---|---|
| Descriptor | `Pmv = exp(-c/(Pm+Pv))`, 6 nodes, c = 0.20430441 | `src/evaluate.py::pmv`, `results/mined_descriptor.txt` |
| Rank equivalence | AUC 0.9772 for both `Pmv` and `Pm+Pv` | `results/paper_tables/table_iv_pixel.csv`; `scripts/04_evaluate.py` prints both |
| High-complexity reference | `Psmv = cos(A^2)`, C=13, accuracy 95.34%, AUC 0.9867 | `src/evaluate.py::psmv`, `results/paper_tables/table_iv_pixel.csv` |
| Table I | Data + YOLO protocol + SR/teacher settings | `configs/default.yaml`, training `args.yaml` records |
| Table II | Pareto points / expressions | `results/paper_tables/table_ii_pareto.csv` |
| Table III | Five converged seeds; two additional seeds failed | `results/paper_tables/table_iii_seeds.csv`; see §3 |
| Table IV | Pixel accuracy/F1/AUC, 1:2 balanced sample | `results/paper_tables/table_iv_pixel.csv` |
| Table V | Single-channel YOLO, four inputs + raw sum | `results/paper_tables/table_v_single_channel.csv` |
| Table VI | Three-channel YOLO, four combinations | `results/paper_tables/table_vi_three_channel.csv` |
| Table VII | Four YOLO variants, baseline vs `Pmv` (P/R and mAP/F1 shown as pairs to fit the IEEE column; all values are in the CSV) | `results/paper_tables/table_vii_yolo.csv` |
| Table VIII | Two-layer pixel extraction | `results/paper_tables/table_viii_two_layer.csv` |
| Table IX | PSDD/IPSD cross-sensor AUC | `results/paper_tables/table_ix_cross_dataset.csv` |
| IPSD check | Pmv 97.5%, Psmv 96.9%, one scalar calibration | `results/paper_tables/table_ipsd_balanced.json` |
| Figures 1–6 | pipeline / schematic / dataset / teacher / SR / detector | `matlab/`, `src/`, manuscript assets |
| Figure 7 | soft labels + teacher output | `scripts/01_teacher.py` outputs |
| Figure 8 | Pareto front | `results/paper_tables/table_ii_pareto.csv` |
| Figures 9–10 | representation comparison / ROC | `results/paper_tables/table_iv_pixel.csv`, `scripts/04_evaluate.py` |
| Figures 11–14 | detection panels / combined PR / two-layer decision | `results/paper_tables/table_v*`–`table_viii*` |
| Figures 15–16 | soft-label / operator ablations | `scripts/02_search.py --target hard`, `--operators ...` |

## 2. Deliberate scope decisions

### 2.1 Four YOLO variants, not eight detectors

The manuscript's detector comparison is **four YOLO variants** (YOLOv8n,
YOLOv11n, YOLOv12n, YOLO26), all at 640×640 with 300/200 epochs. The earlier
MMDetection one/two-stage rows (RetinaNet, SSD, PAA, Faster R-CNN) were removed
because their final per-cell training logs were not archived and their training
budgets could not be matched. The paper states this explicitly and does not claim
cross-family generality. The repository's `configs/default.yaml` therefore lists
only the four YOLO architectures.

### 2.2 Psmv is C=13; C=29 is legacy

The final paper uses the **13-node** expression

```
Psmv = cos(A^2),  A = exp((-0.6271455 - Ps) * Pv) / (Pm^2 + 0.7921049)
```

for the pixel-level reference row. The older C=29 transcription is kept in
`src/evaluate.py::psmv_c29_legacy` only to replay earlier logs; it is **not** the
paper's Psmv. The previous README/mined-descriptor text that called C=29 "the
paper's Psmv" is superseded.

### 2.3 Pmv is a calibration, not a ranking gain

`Pmv` is strictly monotone in `Pm+Pv`; the raw sum has the same AUC. The paper
therefore reports the raw sum explicitly (Table IV) and attributes the detector
gain to the bounded, closed-form output scale. The code prints both `Pmv` and
`Pm+Pv`; do not describe the descriptor as "more discriminative" than the sum.

### 2.4 IPSD is a cross-sensor check with one scalar calibration

The descriptor's constant is tied to the power scale. The IPSD experiment matches
the median of `Pm+Pv` between PSDD and IPSD with **one scalar**; the expression
and its constant are unchanged. This is not zero-shot transfer and the paper says
so. `scripts/08_transfer.py` and `ipsd_balanced_v55` implement the calibration.

## 3. Known limitations that remain in the paper

These are stated in the manuscript's Limitations section; they are not hidden in
the code:

1. Pixel metrics use a 1:2 balanced sample, not the natural prior (~0.86%).
2. Detection uses the benchmark tile-level split; 20 of 23 scenes appear in both
   training and test tiles (scene leakage).
3. PSDD and IPSD pixel labels use the same BT.601 threshold rule; the IPSD check
   is cross-sensor consistency, not independent-ground-truth generalization.
4. Detection results are single-run.
5. Architecture coverage is limited to four YOLO variants.
6. The C=13 Psmv is only a pixel-level reference; it is not used in detection.
7. The two additional SR seeds (456, 1024) converged to a different local
   optimum (AUC 0.9598); they are documented, not used for the reported
   descriptor.

## 4. How to regenerate the table files

`results/paper_tables/` contains the exact files used in the manuscript:

- `table_ii_pareto.csv` — copied from the archived `T1_pareto.csv`
- `table_iii_seeds.csv` — copied from `T2_stability.csv`
- `table_iv_pixel.csv` — copied from `all_metrics_with_fast_cfar.csv`
- `table_v_single_channel.csv`, `table_vi_three_channel.csv`,
  `table_vii_yolo.csv` — derived from `detector_revalidation.csv`
  (the derivation is simple percentage conversion; see the CSV headers)
- `table_ipsd_balanced.json` — copied from `ipsd_balanced_v55.json`

If you change a number in the paper, change the corresponding file here and
re-run the manuscript assembly; do not edit the PDF directly.
