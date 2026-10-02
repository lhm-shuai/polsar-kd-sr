# -*- coding: utf-8 -*-
"""Intelligent-assists-general detection.

The intelligent layer (a learned detector on [P_mv, P_v, P_s]) proposes boxes;
the general layer then extracts the ship pixels inside each box with the
explicit descriptor of Section II-B.  The extraction is a threshold on the
descriptor followed by 8-connected components and an area filter, so it needs
no training and every decision it makes can be reviewed.

Refining a detection:
    mask  = (P_mv >= T_d) inside the box
    comps = 8-connected components of mask
    keep  = largest component with |comp| >= A_min
    box'  = bounding box of keep          (drop the detection if none)

Both the descriptor threshold T_d and the minimum area A_min are fixed on the
training set (T_d by the Youden criterion, A_min by the smallest annotated
ship in the training set), so the refinement introduces no learned parameter.
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


@dataclass
class Refined:
    box: Box
    score: float
    kept: bool
    area: int


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component of a boolean mask."""
    if not mask.any():
        return mask
    if ndi is not None:
        lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        if n <= 1:
            return mask
        sizes = ndi.sum(mask, lab, index=range(1, n + 1))
        return lab == (int(np.argmax(sizes)) + 1)
    # BFS fallback
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best, best_size = None, -1
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack, comp = [(y0, x0)], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
            if len(comp) > best_size:
                best_size, best = len(comp), comp
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def refine_box(descriptor: np.ndarray, box: Box, threshold: float,
               min_area: int) -> Refined:
    """Tighten one detector box with the explicit descriptor.

    `descriptor` is the P_mv map of the image, values in [0, 1].
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = descriptor.shape
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return Refined(box, 0.0, False, 0)
    window = descriptor[y1:y2, x1:x2] >= threshold
    comp = _largest_component(window)
    area = int(comp.sum())
    if area < min_area:
        return Refined(box, 0.0, False, area)
    ys, xs = np.nonzero(comp)
    new_box = (x1 + float(xs.min()), y1 + float(ys.min()),
               x1 + float(xs.max()) + 1.0, y1 + float(ys.max()) + 1.0)
    return Refined(new_box, float(descriptor[y1:y2, x1:x2][comp].mean()), True, area)


def refine_detections(descriptor: np.ndarray, boxes: Sequence[Sequence[float]],
                      scores: Sequence[float], threshold: float,
                      min_area: int, keep_unrefined: bool = True) -> Tuple[list, list]:
    """Refine a whole image; returns (boxes, scores) for the next evaluation."""
    out_b, out_s = [], []
    for b, s in zip(boxes, scores):
        r = refine_box(descriptor, tuple(b[:4]), threshold, min_area)
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
