# -*- coding: utf-8 -*-
"""Intelligent-assists-general detection.

The intelligent layer (a detector trained on [P_mv, P_v, P_s]) proposes boxes.
The generic layer then refines each box with the explicit descriptor:

    seed = (P_mv >= T_high) inside the box
    grow = (P_mv >= T_low) connected to the seed by 8-neighborhood dilation
    components of grow with area >= A_min form the in-box decision
    box' = bounding box of those components

The high threshold selects strong ship cores, and the low threshold recovers
weak returns attached to them. Components without a strong core, and components
below the area floor, are discarded. The rule has no learned parameter; T_high,
T_low, and A_min come from `configs/default.yaml`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

try:                                   # scipy is optional; a BFS fallback is used
    from scipy import ndimage as ndi
except Exception:                      # pragma: no cover
    ndi = None


Box = Tuple[float, float, float, float]          # x1, y1, x2, y2
S8 = np.ones((3, 3), dtype=int)


@dataclass
class Refined:
    box: Box
    score: float
    kept: bool
    area: int


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if ndi is not None:
        lab, n = ndi.label(mask, structure=S8)
        return lab, int(n)
    # BFS fallback
    h, w = mask.shape
    lab = np.zeros_like(mask, dtype=int)
    n = 0
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or lab[y0, x0]:
                continue
            n += 1
            stack = [(y0, x0)]
            lab[y0, x0] = n
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = n
                            stack.append((yy, xx))
    return lab, n


def _keep_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    lab, n = _components(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if areas[i] >= min_area]
    if not keep:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(lab, keep)


def refine_box(descriptor: np.ndarray, box: Box, high_threshold: float,
               low_threshold: float, min_area: int) -> Refined:
    """Tighten one detector box with the two-step descriptor rule.

    `descriptor` is the P_mv map on the 0-255 scale used by the paper.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = descriptor.shape
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return Refined(box, 0.0, False, 0)
    crop = descriptor[y1:y2, x1:x2]
    seed = crop >= high_threshold
    if not seed.any():
        return Refined(box, 0.0, False, 0)
    low = crop >= low_threshold
    if ndi is not None:
        grow = low & ndi.binary_dilation(seed, structure=S8, iterations=3)
    else:
        grow = seed
    comp = _keep_components(grow, min_area)
    area = int(comp.sum())
    if area == 0:
        return Refined(box, 0.0, False, 0)
    ys, xs = np.nonzero(comp)
    new_box = (x1 + float(xs.min()), y1 + float(ys.min()),
               x1 + float(xs.max()) + 1.0, y1 + float(ys.max()) + 1.0)
    return Refined(new_box, float(crop[comp].mean()), True, area)


def refine_detections(descriptor: np.ndarray, boxes: Sequence[Sequence[float]],
                      scores: Sequence[float], high_threshold: float,
                      low_threshold: float, min_area: int,
                      keep_unrefined: bool = False) -> Tuple[list, list]:
    """Refine a whole image; returns (boxes, scores) for the next evaluation."""
    out_b, out_s = [], []
    for b, s in zip(boxes, scores):
        r = refine_box(descriptor, tuple(b[:4]), high_threshold,
                       low_threshold, min_area)
        if r.kept:
            out_b.append(r.box)
            out_s.append(float(s))          # the detector score is kept
        elif keep_unrefined:
            out_b.append(tuple(b[:4]))      # fall back to the detector box
            out_s.append(float(s))
    return out_b, out_s


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(preds: Iterable[dict], gts: Iterable[dict],
                      iou_thr: float = 0.5) -> float:
    """mAP@0.5 for a single class.

    preds / gts: dicts with 'boxes' and (for preds) 'scores'.
    """
    preds, gts = list(preds), list(gts)
    npos = sum(len(g['boxes']) for g in gts)
    if npos == 0:
        return 0.0
    scored = []
    for i, p in enumerate(preds):
        for b, s in zip(p['boxes'], p['scores']):
            scored.append((float(s), i, b))
    scored.sort(key=lambda t: -t[0])
    used = [set() for _ in gts]
    tp = np.zeros(len(scored))
    fp = np.zeros(len(scored))
    for k, (_, i, b) in enumerate(scored):
        best, best_j = iou_thr - 1e-9, -1
        for j, g in enumerate(gts):
            if j != i:
                continue
            for gi, gb in enumerate(g['boxes']):
                if gi in used[j]:
                    continue
                v = iou(b, gb)
                if v > best:
                    best, best_j = v, gi
        if best_j >= 0:
            tp[k] = 1
            used[i].add(best_j)
        else:
            fp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
