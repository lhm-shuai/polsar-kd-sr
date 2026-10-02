# -*- coding: utf-8 -*-
"""Parameters, GFLOPs and forward-only FPS of the YOLO family (Table XIII).

The MMDetection models (RetinaNet / Faster R-CNN / PAA / SSD300) are measured
with the same protocol inside their own environment; the numbers reported in the
paper are in results/size_speed_{yolo,mmdet,fps}.json, which this script
refreshes for the YOLO rows.  Checkpoints are looked up under --runs, the output
tree of scripts/05_detector.py:

    python scripts/10_sizes.py --runs runs
"""
import os
import sys
import time

import numpy as np
import torch


YOLOS = {
    # label: (checkpoint relative to --runs, architecture yaml, input size)
    "YOLOv8n (detector)": ("pmvpvps_300/weights/best.pt", "yolov8n.yaml", 640),
    "YOLOv8n-seg (pixel branch)": ("pmvpvps_seg_300/weights/best.pt", "yolov8n-seg.yaml", 640),
    "YOLOv11n": ("yolo11n/weights/best.pt", "yolo11n.yaml", 640),
    "YOLOv12n": ("yolo12n/weights/best.pt", "yolov12n.yaml", 640),
    "YOLO26": ("yolo26/weights/best.pt", "yolo26.yaml", 640),
}


def measure_efficiency():
    from thop import profile
    from ultralytics import YOLO
    out = {}
    for label, (w, yml, imgsz) in YOLOS.items():
        src = w if os.path.exists(w) else yml
        model = YOLO(src)
        net = model.model.float()
        n_par = sum(p.numel() for p in net.parameters())
        net.eval()
        x = torch.zeros(1, 3, imgsz, imgsz)
        macs, _ = profile(net, inputs=(x,), verbose=False)
        gflops = macs / 1e9
        # end-to-end single-image FPS on the GPU, decode+NMS included
        model.to("cuda")
        img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(8):
            model.predict(img, verbose=False, device=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        n = 60
        for _ in range(n):
            model.predict(img, verbose=False, device=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fps = n / (time.time() - t0)
        out[label] = (n_par / 1e6, gflops, fps, src)
        print("%-26s %.2f M  %.1f GFLOPs  %.1f FPS   <- %s"
              % (label, n_par / 1e6, gflops, fps, os.path.basename(os.path.dirname(os.path.dirname(src)))))
    return out




if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs",
                    help="training-run root the YOLOS table is resolved against")
    a = ap.parse_args()
    for label, (rel, yml, imgsz) in list(YOLOS.items()):
        YOLOS[label] = (os.path.join(a.runs, rel), yml, imgsz)
    measure_efficiency()
