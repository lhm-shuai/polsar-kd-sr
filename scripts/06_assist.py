# -*- coding: utf-8 -*-
"""06_assist.py - the intelligent-assists-general experiment.

For every test image the script

  1. runs the detector trained on [P_mv, P_v, P_s] and keeps its boxes
     (the intelligent layer),
  2. refines each box with the explicit descriptor P_mv >= T_d, 8-connected
     components and the minimum area A_min (the general layer),
  3. evaluates mAP@0.5 for the detector alone and for the refined pipeline.

Usage:
    python scripts/06_assist.py --config configs/default.yaml \
        --weights runs/detect/pmv/weights/best.pt --split test

The descriptor map of an image is P_mv = exp(-c / (P_m + P_v)) computed from the
Yamaguchi components, so no extra training is involved.  T_d and A_min are read
from the config (they are fixed on the training set: T_d by the Youden criterion
and A_min as the smallest annotated ship area).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.assist import average_precision, refine_detections      # noqa: E402


def descriptor_map(components: np.ndarray, c: float) -> np.ndarray:
    """P_mv = exp(-c / (P_m + P_v)) from a (..., 3) = (P_m, P_v, P_s) array."""
    pm, pv = components[..., 0], components[..., 1]
    s = np.maximum(pm + pv, 1e-9)
    return np.exp(-c / s)


def load_split(cfg: dict, split: str):
    """Yield (image_path, components, gt_boxes) for one split."""
    root = cfg['paths']['components']
    img_dir = os.path.join(root, split, 'images')
    comp_dir = os.path.join(root, split, 'components')
    for name in sorted(os.listdir(comp_dir)):
        stem = os.path.splitext(name)[0]
        comp = np.load(os.path.join(comp_dir, name))
        gt_path = os.path.join(root, split, 'labels', stem + '.txt')
        gt = []
        if os.path.exists(gt_path):
            with open(gt_path) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) == 5:                       # class cx cy w h (normalized)
                        _, cx, cy, bw, bh = (float(v) for v in p)
                        h, w = comp.shape[:2]
                        gt.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                                   (cx + bw / 2) * w, (cy + bh / 2) * h))
        yield os.path.join(img_dir, stem + '.png'), comp, gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--describe-only', action='store_true')
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    c = float(cfg['assist']['c'])
    t_d = float(cfg['assist']['threshold'])
    a_min = int(cfg['assist']['min_area'])

    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics is required: pip install ultralytics')
        return 1

    model = YOLO(args.weights)
    preds_raw, preds_ref, gts = [], [], []
    for img_path, comp, gt in load_split(cfg, args.split):
        desc = descriptor_map(comp.astype(np.float32), c)
        res = model.predict(img_path, conf=args.conf, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        rb, rs = refine_detections(desc, boxes, scores, t_d, a_min)
        preds_raw.append({'boxes': [tuple(b[:4]) for b in boxes], 'scores': list(scores)})
        preds_ref.append({'boxes': rb, 'scores': rs})
        gts.append({'boxes': gt})
        if args.describe_only:
            print('%s: %d boxes -> %d kept' % (os.path.basename(img_path), len(boxes), len(rb)))

    m_raw = average_precision(preds_raw, gts)
    m_ref = average_precision(preds_ref, gts)
    print('detector alone      mAP@0.5 = %.4f' % m_raw)
    print('assisted pipeline   mAP@0.5 = %.4f  (%+.4f)' % (m_ref, m_ref - m_raw))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
