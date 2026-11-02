# Paper table sources

These files contain the values used in the manuscript. The paper has seven
content tables (I–VII); the CSV names retain their earlier numbering, and the
mapping below is the authoritative one.

| File | Paper item | Notes |
|---|---|---|
| `table_ii_pareto.csv` | Table III, Fig. 7 | seed-666 Pareto front; complexity 6 is `Pmv`; complexity 29 is `Psmv` |
| `table_iii_seeds.csv` | Fig. 7 | five official seeds (42, 123, 211, 666, 789) |
| `table_iv_pixel.csv` | Table IV | 12 methods on PSDD and IPSD; PSDD and IPSD metrics side by side |
| `table_v_single_channel.csv` | Table V | YOLOv8 single-channel inputs |
| `table_vi_three_channel.csv` | Table V | YOLOv8 three-channel combinations |
| `table_vii_yolo.csv` | Table VI | eight detectors, baseline and descriptor input |
| `table_viii_two_layer.csv` | Table VII | three two-layer schemes reported in the paper |
| `table_ipsd_balanced.json` | Table IV, Fig. 9(b) | IPSD balanced-set scores and the scalar calibration |
| `detector_revalidation.csv` | source | YOLO runs, including the raw-sum row `PmPv_raw` |
| `mmdet_revalidation.csv` | source | RetinaNet, Faster R-CNN, SSD, and PAA runs with checkpoint paths |

The detector tables use the same runs as the paper. The YOLO family was
initialized from COCO-pretrained weights; the MMDetection networks use the
official configurations. `table_iv_pixel.csv` reports the 12-method comparison
that appears in the paper. The raw-sum detector run used for the rank check is
kept in `detector_revalidation.csv` as `PmPv_raw`.

TCR values are cited in Sec. III-C and produced by `scripts/09_tcr.py`; the
per-run values are kept with the reproduction package rather than duplicated
here.
