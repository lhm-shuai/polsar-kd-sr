# Paper table sources (v75)

These are the exact files behind Tables II–VII of the IEEE manuscript
`paper/PolSAR_paper_English_v75_IEEE.pdf`.

| File | Paper table | Notes |
|---|---|---|
| `table_ii_pareto.csv` | Table II | archived `T1_pareto.csv`; C=6 is the selected descriptor, C=13 the Psmv reference |
| `table_iii_seeds.csv` | Table III | archived `T2_stability.csv`; five converged seeds; two extra seeds failed (see `PAPER_CONSISTENCY.md`) |
| `table_iv_pixel.csv` | Table IV | `all_metrics_with_fast_cfar.csv`; 1:2 balanced sample; the raw-sum row is `92.25/0.8877/0.9772` |
| `table_v_single_channel.csv` | Table V | derived from `detector_revalidation.csv`; values in percent |
| `table_vi_three_channel.csv` | Table VI | derived from `detector_revalidation.csv`; values in percent |
| `table_vii_yolo.csv` | Table VII | derived from `detector_revalidation.csv`; four YOLO variants only |
| `table_viii_two_layer.csv` | Table VIII | two-layer pixel extraction; operating points in the paper |
| `table_ix_cross_dataset.csv` | Table IX | PSDD/IPSD AUC and IPSD F1 |
| `table_ipsd_balanced.json` | IPSD paragraph | `ipsd_balanced_v55.json`; one scalar calibration |
| `detector_revalidation.csv` | raw source | YOLO runs; includes the raw-sum row `PmPv_raw` |

The four MMDetection rows that appeared in earlier drafts are **not** included:
their final logs were not archived. The manuscript and repository both restrict
the comparison to four YOLO variants.

`table_iv_pixel.csv` includes a CFAR row; the manuscript's Table IV reports the
seven rows listed in the paper (Ps, Pm, Pv, Teacher, Pmv, Psmv C=13, raw sum).
