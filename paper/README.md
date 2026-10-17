# Manuscript

`PolSAR_paper_English_v96_IEEE.pdf` / `.docx` are the current IEEE Transactions
manuscript, aligned with the Chinese v95 paper and this repository.

- 14 pages, IEEE two-column format.
- Eight detectors: four YOLO variants (YOLOv8n, YOLOv11n, YOLOv12n, YOLO26) and
  four MMDetection detectors (RetinaNet, Faster R-CNN, SSD, PAA).
- `Pmv = exp(-c/(Pm+Pv))`, C=6; `Psmv` is the C=29 pixel-level reference
  (Eq. 23), not used in detection.
- Tables I–IX are traceable to `results/paper_tables/`.
- The superseded v75 manuscript (Psmv C=13, four YOLO only) has been removed.

If you change code/results, update the manuscript's corresponding table and
`results/paper_tables/` together. Do not edit the PDF directly.
