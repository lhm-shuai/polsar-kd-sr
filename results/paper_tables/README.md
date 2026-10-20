# Paper table sources (v100)

These files support the tables of the final manuscript
(`08_论文/PolSAR_paper_EN_投稿版_IEEE_v100_审查修订.docx`). The paper uses the C29
descriptor as Psmv and the eight-detector comparison.

| File | Paper table | Notes |
|---|---|---|
| `table_ii_pareto.csv` | Table III / search front | full seed-666 Pareto front; complexity 6 is the selected Pmv; complexity 29 is the canonical Psmv (C29). The C13 row is an intermediate Pareto point, not Psmv. |
| `table_iii_seeds.csv` | Sec. III-B stability | five official seeds (42, 123, 211, 666, 789); seeds 456 and 1024 were excluded from the paper and are archived outside this repository. |
| `table_iv_pixel.csv` | Table IV | pixel-level evaluation including the teacher and raw-sum rows; the paper reports the 12-method comparison subset. |
| `table_v_single_channel.csv` | Table VI (single channel) | YOLOv8 single-channel inputs; values in percent. |
| `table_vi_three_channel.csv` | Table VI (three channel) | YOLOv8 three-channel combinations; values in percent. |
| `table_vii_yolo.csv` | Table VII | eight detectors: four YOLO variants plus RetinaNet, Faster R-CNN, SSD, and PAA. |
| `table_viii_two_layer.csv` | Table VIII | two-layer pixel extraction; six rows, operating points as in the paper. |
| `table_ix_cross_dataset.csv` | Table IX | PSDD/IPSD AUC and IPSD operating-point metrics. |
| `table_ipsd_balanced.json` | Cross-dataset paragraph | IPSD balanced-set scores; one scalar calibration. |
| `detector_revalidation.csv` | raw source | YOLO runs, including the raw-sum row `PmPv_raw`. |
| `mmdet_revalidation.csv` | raw source | RetinaNet/Faster R-CNN/SSD/PAA runs with checkpoint paths. |

The paper's Table V (target-to-clutter ratio) is produced by `scripts/09_tcr.py`;
the corresponding per-run values are kept with the reproduction package rather
than duplicated here.
