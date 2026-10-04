# -*- coding: utf-8 -*-
"""Apply the paper's own mask-to-image consistency screening to IPSD and
re-score the zero-shot transfer on the surviving tiles.

Screen (same rule as section III-A on PSDD): on the label-defining luminance
map L = 0.299*render(Pm) + 0.587*render(Pv) + 0.114*render(Ps) (BT.601 weights,
each component rendered by clipping at its own 99th percentile), a tile is kept
only if the mean luminance inside the (3-px-eroded) boxes is not below the
background mean.  Scoring then follows the documented protocol: positives =
eroded box pixels, negatives = outside an 8-px guard band, one median-match
calibration per tile."""
import glob
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy.ndimage as ndi
import tifffile
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gapfill"))
from yamaguchi import c3_to_t3, yamaguchi4_mode0          # noqa: E402

IPSD = r"D:\雷达\数据库\IPSD"
OUT = os.path.join(HERE, "analysis", "out", "ipsd_screened.json")
REF = 1.244077813118314e-3
C = 0.21
GUARD = 8
ERODE = 3
W = (0.299, 0.587, 0.114)


def decompose(tif):
    a = tifffile.imread(tif).astype(np.float64)
    T11, T12, T13, T22, T23, T33 = c3_to_t3(
        a[..., 0], a[..., 1] + 1j * a[..., 2], a[..., 3] + 1j * a[..., 4],
        a[..., 5], a[..., 6] + 1j * a[..., 7], a[..., 8])
    Podd, Pdbl, Pvol, Phel = yamaguchi4_mode0(T11, T12, T13, T22, T23, T33)
    return T11 + T22 + T33, Pdbl + Phel, Pvol, Podd


def render(x):
    v = np.percentile(x, 99)
    return np.clip(x / max(v, 1e-30), 0, 1)


def main():
    feats = ("Pmv", "SPAN", "Pm", "Pv", "Ps")
    pos = {k: [] for k in feats}
    neg = {k: [] for k in feats}
    tiles, kept = [], []
    xmls = sorted(glob.glob(os.path.join(IPSD, "VOC", "Annotations", "*.xml")))
    for i, xml in enumerate(xmls):
        base = os.path.basename(xml)[:-4]
        tif = os.path.join(IPSD, "VOC", "TIFImages", base + ".tif")
        if not os.path.exists(tif):
            continue
        r = ET.parse(xml).getroot()
        boxes = []
        for o in r.findall(".//object"):
            b = o.find("bndbox")
            if b is not None:
                boxes.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
        if not boxes:
            continue
        span, Pm, Pv, Ps = decompose(tif)
        box = np.zeros(span.shape, bool)
        for x1, y1, x2, y2 in boxes:
            box[int(max(0, y1)):int(y2) + 1, int(max(0, x1)):int(x2) + 1] = True
        pos_m = ndi.binary_erosion(box, iterations=ERODE)
        if pos_m.sum() < 16:
            continue
        guard = ndi.binary_dilation(box, iterations=GUARD)
        neg_m = ~guard
        if neg_m.sum() < 16:
            continue
        L = (W[0] * render(Pm) + W[1] * render(Pv) + W[2] * render(Ps))
        ok = L[pos_m].mean() >= L[neg_m].mean()
        tiles.append(dict(tile=base, lum_in=float(L[pos_m].mean()),
                          lum_out=float(L[neg_m].mean()), kept=bool(ok)))
        if not ok:
            continue
        kept.append(base)
        S = Pm + Pv
        k = REF / max(np.median(S.ravel()), 1e-30)
        vals = dict(SPAN=span, Pm=Pm, Pv=Pv, Ps=Ps,
                    Pmv=np.exp(-C / np.maximum(k * S, 1e-30)))
        for key in feats:
            pos[key].append(vals[key][pos_m].astype(np.float32))
            neg[key].append(vals[key][neg_m].astype(np.float32))
        if (i + 1) % 100 == 0:
            print("%d/%d kept=%d" % (i + 1, len(xmls), len(kept)), flush=True)

    y = np.r_[np.ones(sum(len(a) for a in pos["Pmv"])),
              np.zeros(sum(len(a) for a in neg["Pmv"]))]
    res = {}
    for key in feats:
        s = np.concatenate(pos[key] + neg[key])
        res[key] = dict(auc=float(roc_auc_score(y, s)))
    out = dict(pooled=res, kept=kept, tiles=tiles,
               protocol=dict(erode=ERODE, guard=GUARD, c=C, screen="BT601 luminance"))
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("tiles kept %d / %d" % (len(kept), len(tiles)))
    for key in feats:
        print("%-5s AUC %.4f" % (key, res[key]["auc"]))


if __name__ == "__main__":
    main()
