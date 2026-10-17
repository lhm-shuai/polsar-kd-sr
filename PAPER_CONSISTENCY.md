# Paper ↔ repository consistency (v96)

This repository is aligned with the IEEE manuscript
`paper/PolSAR_paper_English_v96_IEEE.pdf` (14 pages, 2026-10-18). The v75
manuscript (Psmv C=13, four YOLO only) has been removed. The mapping below is
the one to use when checking a number, a table or a figure.

## 1. What the paper claims

The figure set is the Chinese v93/v95 set 1–16, including the combined PR figure
(English Fig. 13) and the two-layer comparison (English Fig. 14). The English
manuscript has 14 pages and nine tables.

| Paper item | Value / scope | Repository source |
|---|---|---|
| Descriptor | `Pmv = exp(-c/(Pm+Pv))`, 6 nodes, c = 0.20430441 | `src/evaluate.py::pmv`, `results/mined_descriptor.txt` |
| Rank equivalence | AUC 0.9772 for both `Pmv` and `Pm+Pv` | `results/paper_tables/table_iv_pixel.csv`; `scripts/04_evaluate.py` prints both |
| High-complexity reference | `Psmv` = C29, Eq. (23), pixel-level only | `src/evaluate.py::psmv_c29`, `results/paper_tables/table_iv_pixel.csv` |
| Table I | Dataset, training and detector settings | `configs/default.yaml`, training `args.yaml` records |
| Table II | Pareto points / expressions | `results/paper_tables/table_ii_pareto.csv` |
| Table III | Five converged seeds; two additional seeds failed | `results/paper_tables/table_iii_seeds.csv`; see §3 |
| Table IV | Pixel accuracy/F1/AUC, 1:2 balanced sample | `results/paper_tables/table_iv_pixel.csv` |
| Table V | Single-channel YOLO | `results/paper_tables/table_v_single_channel.csv` |
| Table VI | Three-channel YOLO | `results/paper_tables/table_vi_three_channel.csv` |
| Table VII | Eight detectors: four YOLO + four MMDetection | `results/paper_tables/table_vii_yolo.csv`, `mmdet_revalidation.csv` |
| Table VIII | Two-layer pixel extraction, six rows | `results/paper_tables/table_viii_two_layer.csv` |
| Table IX | PSDD/IPSD cross-sensor AUC, 12 methods | `results/paper_tables/table_ix_cross_dataset.csv` |
| TCR paragraph | n=115; Pmv 26.0 dB, Psmv 17.1 dB | `05_结果与中间数据/tcr_common_v62.json` in the full package |
| IPSD check | Pmv 97.5%, Psmv 96.9%, one scalar calibration | `results/paper_tables/table_ipsd_balanced.json` |

## 2. Deliberate scope decisions

### 2.1 Eight detectors

The manuscript compares four YOLO variants (YOLOv8n, YOLOv11n, YOLOv12n,
YOLO26) and four MMDetection detectors (RetinaNet, Faster R-CNN, SSD, PAA).
This matches the Chinese v95 paper. The v75 restriction to four YOLO variants
is superseded. Detection results remain single-run and are controlled
within-architecture; no broader family generality is claimed.

### 2.2 Psmv is C=29

The high-complexity reference is the 29-node expression of Eq. (23):

```
A = exp((-0.72197366 - Ps) * Pv) / (Pm^2 + 1.0250757)
B = sin(sin(sin(cos(A^8 + 0.48293668))))
Psmv = [B^2 / (0.61469054 * 0.49011374)]^8
```

It is reported only at the pixel level and is not used in detection. The C=13
transcription used in v75 is obsolete and has been removed from the manuscript.

### 2.3 Pmv is a calibration, not a ranking gain

`Pmv` is strictly monotone in `Pm+Pv`; the raw sum has the same AUC. The paper
therefore reports the raw sum explicitly and attributes the detector gain to the
bounded, closed-form output scale. The code prints both `Pmv` and `Pm+Pv`.

### 2.4 IPSD is a cross-sensor check with one scalar calibration

The descriptor's constant is tied to the power scale. The IPSD experiment
matches the median of `Pm+Pv` between PSDD and IPSD with one scalar; the
expression and its constant are unchanged. This is not zero-shot transfer.

## 3. Known limitations

1. Pixel metrics use a 1:2 balanced sample, not the natural prior (~0.86%).
2. Detection uses the benchmark tile-level split; shared scenes occur.
3. PSDD and IPSD pixel labels use the same BT.601 threshold rule; the IPSD check
   is cross-sensor consistency, not independent-ground-truth generalization.
4. Detection results are single-run.
5. Architecture coverage is eight detectors; no PolSAR-specific detector is
   compared under the same protocol.
6. C29 Psmv is only a pixel-level reference.
7. The two additional SR seeds (456, 1024) converged to a different local
   optimum (AUC 0.9598); they are documented, not used for the descriptor.

## 4. Table files

`results/paper_tables/` contains the files used in the manuscript:

- `table_ii_pareto.csv` — seed-666 front
- `table_iii_seeds.csv` — five converged seeds + two failed seeds
- `table_iv_pixel.csv` — 1:2 balanced pixel metrics
- `table_v_single_channel.csv`, `table_vi_three_channel.csv`,
  `table_vii_yolo.csv` — detector results
- `table_viii_two_layer.csv` — six-row two-layer extraction
- `table_ix_cross_dataset.csv` — 12-method cross-dataset AUC/F1
- `mmdet_revalidation.csv` — four MMDetection configurations
- `table_ipsd_balanced.json` — IPSD balanced metrics

If you change a number in the paper, change the corresponding file here and
re-run the manuscript assembly; do not edit the PDF directly.
