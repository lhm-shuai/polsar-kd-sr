# -*- coding: utf-8 -*-
"""Zero-training DEPLOYMENT comparison (operating-point metrics).

Motivation: AUC is threshold-free, so it lets a statistic implicitly adapt to
IPSD's power scale; the zero-training claim of the paper is that the operating
point selected on PSDD transfers unchanged.  This script therefore

  1. on PSDD (23 full scenes, box-proxy protocol: positives = pixels inside
     boxes eroded 3 px, negatives = pixels outside an 8-px guard band), finds
     each representation's pooled Youden-optimal threshold (the descriptor uses
     its documented operating point T = 168/255 in value space);
  2. on IPSD (934 tiles, same protocol, one median-match calibration per tile
     as everywhere in the paper), applies each PSDD threshold to the calibrated
     values and reports precision/recall/F1/IoU.
"""
import glob
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
import tifffile
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gapfill"))
from yamaguchi import c3_to_t3, yamaguchi4_mode0          # noqa: E402

PSDD = r"D:\雷达\数据库\PSDDv1.0"
IPSD = r"D:\雷达\数据库\IPSD"
OUT = os.path.join(HERE, "analysis", "out", "ipsd_operating_point.json")
C = 0.21
GUARD = 8
ERODE = 3
T_PMV = 168 / 255.0


def decompose_t3(T11, T12, T13, T22, T23, T33):
    Podd, Pdbl, Pvol, Phel = yamaguchi4_mode0(T11, T12, T13, T22, T23, T33)
    return dict(SPAN=T11 + T22 + T33, Pm=Pdbl + Phel, Pv=Pvol, Ps=Podd,
                Pmv=np.exp(-C / np.maximum(Pdbl + Phel + Pvol, 1e-30)))


def boxes_of(xml_path):
    r = ET.parse(xml_path).getroot()
    out = []
    for o in r.findall(".//object"):
        b = o.find("bndbox")
        if b is not None:
            out.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def masks_of(boxes, shape):
    box = np.zeros(shape, bool)
    for x1, y1, x2, y2 in boxes:
        box[int(max(0, y1)):int(y2) + 1, int(max(0, x1)):int(x2) + 1] = True
    pos = ndi.binary_erosion(box, iterations=ERODE)
    neg = ~ndi.binary_dilation(box, iterations=GUARD)
    return pos, neg


RNG = __import__('numpy').random.RandomState(7)


def subsample(arr, cap=200000):
    if len(arr) <= cap:
        return arr
    return arr[RNG.choice(len(arr), cap, replace=False)]


def psdd_thresholds():
    """Pooled Youden-optimal thresholds per representation on PSDD."""
    xmls = sorted(glob.glob(os.path.join(PSDD, "Annotations全景极化对应标签", "*.xml")))
    pos = {k: [] for k in ("SPAN", "Pm", "Pv", "Ps")}
    neg = {k: [] for k in ("SPAN", "Pm", "Pv", "Ps")}
    for xml in xmls:
        base = os.path.basename(xml)[:-4]
        mat = os.path.join(PSDD, "mat全景极化", base + ".mat")
        if not os.path.exists(mat):
            continue
        d = sio.loadmat(mat)
        T11, T22, T33 = d["T11"], d["T22"], d["T33"]
        T12 = d["T12_real"] + 1j * d["T12_imag"]
        T13 = d["T13_real"] + 1j * d["T13_imag"]
        T23 = d["T23_real"] + 1j * d["T23_imag"]
        vals = decompose_t3(T11, T12, T13, T22, T23, T33)
        bx = boxes_of(xml)
        if not bx:
            continue
        pos_m, neg_m = masks_of(bx, T11.shape)
        if pos_m.sum() < 16 or neg_m.sum() < 16:
            continue
        for key in pos:
            pos[key].append(subsample(vals[key][pos_m]).astype(np.float32))
            neg[key].append(subsample(vals[key][neg_m]).astype(np.float32))
        print("PSDD", base, flush=True)
    y = np.r_[np.ones(sum(len(a) for a in pos["SPAN"])),
              np.zeros(sum(len(a) for a in neg["SPAN"]))]
    thr = {}
    for key in pos:
        s = np.concatenate(pos[key] + neg[key])
        fpr, tpr, t = roc_curve(y, s)
        thr[key] = float(t[int(np.argmax(tpr - fpr))])
    return thr


def ipsd_scores(thr):
    """F1/IoU on IPSD with PSDD thresholds transferred through the same
    per-tile median-match calibration the descriptor uses."""
    feats = ("SPAN", "Pm", "Pv", "Ps")
    tp = {k: 0 for k in feats + ("Pmv",)}
    fp = {k: 0 for k in tp}
    fn = {k: 0 for k in tp}
    xmls = sorted(glob.glob(os.path.join(IPSD, "VOC", "Annotations", "*.xml")))
    for i, xml in enumerate(xmls):
        base = os.path.basename(xml)[:-4]
        tif = os.path.join(IPSD, "VOC", "TIFImages", base + ".tif")
        if not os.path.exists(tif):
            continue
        boxes = boxes_of(xml)
        if not boxes:
            continue
        a = tifffile.imread(tif).astype(np.float64)
        T11, T12, T13, T22, T23, T33 = c3_to_t3(
            a[..., 0], a[..., 1] + 1j * a[..., 2], a[..., 3] + 1j * a[..., 4],
            a[..., 5], a[..., 6] + 1j * a[..., 7], a[..., 8])
        vals = decompose_t3(T11, T12, T13, T22, T23, T33)
        pos_m, neg_m = masks_of(boxes, T11.shape)
        if pos_m.sum() < 16 or neg_m.sum() < 16:
            continue
        S = vals["Pm"] + vals["Pv"]
        k = 1.244077813118314e-3 / max(np.median(S.ravel()), 1e-30)
        # y = 1 inside the guard band; positive class = eroded box pixels
        ymap = np.zeros(T11.shape, np.int8)      # 0 neg, 1 pos
        ymap[pos_m] = 1
        ymap[~ndi.binary_dilation(
            ndi.binary_dilation(pos_m, iterations=ERODE), iterations=GUARD)] = 0
        inband = ~((ymap == 0) & neg_m)
        for key in feats:
            pred = (k * vals[key]) >= thr[key]
            pred = pred & inband
            tp[key] += int((pred & (ymap == 1)).sum())
            fp[key] += int((pred & (ymap == 0)).sum())
            fn[key] += int((~pred & (ymap == 1)).sum())
        pred = (vals["Pmv"] >= T_PMV) & inband
        tp["Pmv"] += int((pred & (ymap == 1)).sum())
        fp["Pmv"] += int((pred & (ymap == 0)).sum())
        fn["Pmv"] += int((~pred & (ymap == 1)).sum())
        if (i + 1) % 100 == 0:
            print("IPSD %d/%d" % (i + 1, len(xmls)), flush=True)
    res = {}
    for key in tp:
        P = tp[key] / max(tp[key] + fp[key], 1)
        R = tp[key] / max(tp[key] + fn[key], 1)
        res[key] = dict(prec=round(P, 4), rec=round(R, 4),
                        f1=round(2 * P * R / max(P + R, 1e-12), 4),
                        iou=round(tp[key] / max(tp[key] + fp[key] + fn[key], 1), 4))
    return res


def main():
    thr = psdd_thresholds()
    print("PSDD operating points:", thr)
    res = ipsd_scores(thr)
    out = dict(psdd_thresholds=thr, t_pmv=T_PMV, ipsd=res)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)
    for key, v in res.items():
        print("%-5s F1 %.4f  IoU %.4f  (P %.3f R %.3f)"
              % (key, v["f1"], v["iou"], v["prec"], v["rec"]))


if __name__ == "__main__":
    main()
