# Manuscript

`PolSAR_paper_English_v75_IEEE.pdf` / `.docx` are the IEEE Transactions
manuscript aligned with this repository.

- 16 pages, IEEE two-column format.
- Four YOLO variants (YOLOv8n, YOLOv11n, YOLOv12n, YOLO26).
- `Pmv = exp(-c/(Pm+Pv))`, C=6; `Psmv = cos(A^2)`, C=13 reference (pixel-level only).
- Tables II–VII are traceable to `results/paper_tables/`.
- The MMDetection rows from earlier drafts are intentionally excluded; see
  `../PAPER_CONSISTENCY.md`.

If you change code/results, update the manuscript's corresponding table and
`results/paper_tables/` together. Do not edit the PDF directly.
