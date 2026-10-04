# -*- coding: utf-8 -*-
"""Re-run the IPSD transfer evaluation under the protocol the paper documents:
positives = pixels inside the boxes ERODED by 3 px, negatives = pixels outside
an 8-px guard band.  The committed ipsd_transfer.py skipped the erosion step,
which does not match Table XV/XIII.  Saves pooled AUCs and per-tile AUCs (for
bootstrap CIs) with the erosion in place."""
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
OUT = os.path.join(HERE, "analysis", "out", "ipsd_eroded.json")
REF = 1.244077813118314e-3          # PSDD median(Pm+Pv), from ipsd_transfer.json
C = 0.21
GUARD = 8
ERODE = 3


def decompose(tif):
    a = tifffile.imread(tif).astype(np.float64)
    T11, T12, T13, T22, T23, T33 = c3_to_t3(
        a[..., 0], a[..., 1] + 1j * a[..., 2], a[..., 3] + 1j * a[..., 4],
        a[..., 5], a[..., 6] + 1j * a[..., 7], a[..., 8])
    Podd, Pdbl, Pvol, Phel = yamaguchi4_mode0(T11, T12, T13, T22, T23, T33)
    return T11 + T22 + T33, Pdbl + Phel, Pvol, Podd


def main():
    feats = ("Pmv", "SPAN", "Pm", "Pv", "Ps")
    pos = {k: [] for k in feats}
    neg = {k: [] for k in feats}
    tiles = []
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
        S = Pm + Pv
        k = REF / max(np.median(S.ravel()), 1e-30)
        vals = dict(SPAN=span, Pm=Pm, Pv=Pv, Ps=Ps,
                    Pmv=np.exp(-C / np.maximum(k * S, 1e-30)))
        for key in feats:
            pos[key].append(vals[key][pos_m].astype(np.float32))
            neg[key].append(vals[key][neg_m].astype(np.float32))
        tiles.append(dict(tile=base,
                          auc_span=float(roc_auc_score(
                              np.r_[np.ones(pos_m.sum()), np.zeros(neg_m.sum())],
                              np.r_[span[pos_m], span[neg_m]])),
                          auc_pmv=float(roc_auc_score(
                              np.r_[np.ones(pos_m.sum()), np.zeros(neg_m.sum())],
                              np.r_[vals["Pmv"][pos_m], vals["Pmv"][neg_m]]))))
        if (i + 1) % 100 == 0:
            print("%d/%d" % (i + 1, len(xmls)), flush=True)

    y = np.r_[np.ones(sum(len(a) for a in pos["Pmv"])),
              np.zeros(sum(len(a) for a in neg["Pmv"]))]
    res = {}
    for key in feats:
        s = np.concatenate(pos[key] + neg[key])
        res[key] = dict(auc=float(roc_auc_score(y, s)),
                        n_pos=int(sum(len(a) for a in pos[key])),
                        n_neg=int(sum(len(a) for a in neg[key])))
    out = dict(pooled=res, tiles=tiles, protocol=dict(erode=ERODE, guard=GUARD, c=C))
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)
    for key in feats:
        print("%-5s AUC %.4f" % (key, res[key]["auc"]))


if __name__ == "__main__":
    main()
