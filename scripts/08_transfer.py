# -*- coding: utf-8 -*-
"""Cross-dataset check (Table XV): apply the mined descriptor to IPSD (MaOutCNN's released
dataset) with zero retraining, using the group's own decomposition.

The decomposition is the ported MATLAB code (gapfill/yamaguchi.py), which was
verified against the original .m to 3e-18.  IPSD ships 9-band float32 TIFs whose
bands are the covariance elements in the order the group's main_four.m expects:
[C11, Re C12, Im C12, Re C13, Im C13, C22, Re C23, Im C23, C33].

Because the descriptor exp(-c/(Pm+Pv)) was mined in PSDD's power scale, one
scalar per-dataset calibration is applied: the median of (Pm+Pv) over the
dataset is matched to PSDD's.  The descriptor's form and constant are unchanged.

Evaluation: positives are pixels inside the VOC boxes, negatives are pixels
outside the boxes after an 8-pixel guard band.  Thresholds are never fitted on
IPSD -- AUC is threshold-free and the F1 row uses the operating point selected
on PSDD's validation split.
"""
import glob
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io as sio
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomposition import c3_to_t3, yamaguchi4_mode0     # noqa: E402
from classic_ops import tile_origins                    # noqa: E402

IPSD = r"D:\雷达\数据库\IPSD"
PSDD = r"D:\雷达\数据库\PSDDv1.0"
OUT = os.path.dirname(os.path.abspath(__file__))
GUARD = 8
C_PSDD = 0.21


def decompose_tif(path):
    a = tifffile.imread(path).astype(np.float64)        # (256, 256, 9)
    C11 = a[..., 0]
    C12 = a[..., 1] + 1j * a[..., 2]
    C13 = a[..., 3] + 1j * a[..., 4]
    C22 = a[..., 5]
    C23 = a[..., 6] + 1j * a[..., 7]
    C33 = a[..., 8]
    T11, T12, T13, T22, T23, T33 = c3_to_t3(C11, C12, C13, C22, C23, C33)
    Podd, Pdbl, Pvol, Phel = yamaguchi4_mode0(T11, T12, T13, T22, T23, T33)
    return dict(Ps=Podd, Pm=Pdbl + Phel, Pv=Pvol, Phel=Phel,
                SPAN=T11 + T22 + T33)


def psdd_reference_median():
    """Median of (Pm+Pv) over PSDD test tiles, the mining scale reference."""
    vals = []
    for scene in sorted({os.path.basename(p)[:-4] for p in
                         glob.glob(os.path.join(PSDD, "mat全景极化", "*.mat"))}):
        d = sio.loadmat(os.path.join(PSDD, "mat全景极化", scene + ".mat"))
        T11, T22, T33 = d["T11"], d["T22"], d["T33"]
        T12 = d["T12_real"] + 1j * d["T12_imag"]
        T13 = d["T13_real"] + 1j * d["T13_imag"]
        T23 = d["T23_real"] + 1j * d["T23_imag"]
        cs = (T11.shape[0] // 2, T11.shape[1] // 2)
        sl = lambda A: A[cs[0]:cs[0] + 256, cs[1]:cs[1] + 256]
        try:
            Podd, Pdbl, Pvol, Phel = yamaguchi4_mode0(
                sl(T11), sl(T12), sl(T13), sl(T22), sl(T23), sl(T33))
        except Exception as exc:                        # pragma: no cover
            print("  skip", scene, exc)
            continue
        vals.append(np.median((Pdbl + Phel + Pvol).ravel()))
    v = np.array(vals)
    print("PSDD reference: median(Pm+Pv) over %d scene centres = %.6g (spread %.3g-%.3g)"
          % (len(v), np.median(v), v.min(), v.max()))
    return float(np.median(v))


def boxes_of(xml_path):
    r = ET.parse(xml_path).getroot()
    out = []
    for o in r.findall(".//object"):
        b = o.find("bndbox")
        if b is None:
            continue
        out.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return np.array(out, float) if out else np.zeros((0, 4))


def main():
    ref = psdd_reference_median()

    xmls = sorted(glob.glob(os.path.join(IPSD, "VOC", "Annotations", "*.xml")))
    print("IPSD: %d annotated tiles" % len(xmls))
    accum = {k: ([], []) for k in ("Pmv", "Pm", "Pv", "Ps", "Pm+Pv", "SPAN")}
    stat, diag = [], []
    for i, xml in enumerate(xmls):
        base = os.path.basename(xml)[:-4]
        tif = os.path.join(IPSD, "VOC", "TIFImages", base + ".tif")
        jpg = os.path.join(IPSD, "VOC", "JPEGImages", base + ".jpg")
        if not os.path.exists(tif):
            continue
        comp = decompose_tif(tif)
        bx = boxes_of(xml)
        if not len(bx):
            continue
        gm = np.zeros(comp["SPAN"].shape, bool)
        for x1, y1, x2, y2 in bx:
            gm[int(max(0, y1)):int(y2) + 1, int(max(0, x1)):int(x2) + 1] = True
        if gm.sum() < 16:
            continue
        guard = ndimage.binary_dilation(gm, iterations=GUARD)
        neg = ~guard
        if neg.sum() < 16:
            continue
        S = comp["Pm"] + comp["Pv"]
        stat.append(np.median(S.ravel()))
        k = ref / max(np.median(S.ravel()), 1e-30)
        comp["Pmv"] = np.exp(-C_PSDD / np.maximum(k * S, 1e-30))
        comp["Pm+Pv"] = k * S
        for key in accum:
            v = comp[key]
            accum[key][0].append(v[gm].astype(np.float32))
            accum[key][1].append(v[neg].astype(np.float32))
        if i < 3:
            jp = np.asarray(Image.open(jpg).convert("L")).astype(np.float32)
            diag.append((base, spearmanr(comp["SPAN"].ravel(), jp.ravel()).statistic,
                         float(gm.mean())))
        if (i + 1) % 200 == 0:
            print("  ... %d/%d" % (i + 1, len(xmls)))

    print("\nsanity: rho(SPAN, Pauli-render) on first tiles: %s"
          % ", ".join("%s %+0.3f" % (b, r) for b, r, _ in diag))
    print("calibration: PSDD median %.4g ; IPSD median (per tile) %.4g (median of medians)"
          % (ref, float(np.median(stat))))

    res = {}
    print("\n%-7s %8s %8s %8s %8s %8s" % ("repr", "AUC", "prec", "rec", "F1", "IoU"))
    # operating points: PSDD-selected values on the descriptor's channel scale.
    # Table XII row 1 uses the 8-bit render T=168 with A_min=16; here the same
    # point is expressed through its threshold in value space (168/255) and no
    # area filter, so the number transfers as-is.
    ops = {"Pmv": 168 / 255.0}
    for key in ("Pmv", "Pm", "Pv", "Ps", "Pm+Pv", "SPAN"):
        pos = np.concatenate(accum[key][0])
        neg = np.concatenate(accum[key][1])
        y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        s = np.concatenate([pos, neg])
        auc = roc_auc_score(y, s)
        if key in ops:
            thr = ops[key]
        else:
            fpr, tpr, t = roc_curve(y, s)
            thr = float(t[int(np.argmax(tpr - fpr))])
        pred = s >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        IoU = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        res[key] = dict(auc=auc, prec=P, rec=R, f1=F1, iou=IoU, thr=thr,
                        n_pos=int(len(pos)), n_neg=int(len(neg)))
        print("%-7s %8.4f %8.3f %8.3f %8.3f %8.3f" % (key, auc, P, R, F1, IoU))
    res["_meta"] = dict(psdd_median=ref, ipsd_median=float(np.median(stat)),
                        tiles=len(xmls), guard_px=GUARD, c=C_PSDD)
    with io.open(os.path.join(OUT, "ipsd_transfer.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("wrote ipsd_transfer.json")


if __name__ == "__main__":
    main()
