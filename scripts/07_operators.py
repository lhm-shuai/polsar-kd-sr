# -*- coding: utf-8 -*-
"""Classic polarimetric operators on the paper's test tiles (Table VI).

Mapping tile -> full-scene pixels is the row-major 256/30 scan of the dataset
builder (切片划分数据集一条龙.py) over the full-pol annotations, which was
verified this round: the tile and the scene image agree to mean|diff| ~0.12
(JPEG requantisation) at the predicted origin.

Operators, all computed from the scene's coherency matrix T at the mapped
block, with the clutter covariance estimated from the whole scene:
  SPAN      trace(T)
  PMS       largest eigenvalue of T                        (de Graff 1988)
  Rank-1    lambda1 / trace(T)                             (Cloude, GRSL 2021)
  MPWF      trace(Cc^-1 T)                                 (Liu et al., TGRS 1998)
  PNF       lambda_max(Cc^-1 T)                            (Liu et al., GRSL 2022)
  RS        (|T12| + |T23|) / trace(T)   reflection-symmetry break, normalised
  CA-CFAR   two-parameter CFAR on SPAN, 31x31 cell average
Reference rows (Ps, Pm, Pv, Pmv) come from the tile images themselves.
Thresholds are picked on the validation split (Youden), then applied to the
label-consistent test tiles -- the same protocol as the paper's Table XI.
"""
import collections, glob, io, json, os, re, sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io as sio
from PIL import Image

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)
DS = os.environ.get('PSDD_COMPOSITE', r'D:\雷达\数据库\PSDDv1.0\论文所用数据集\四成分jpg')
MASK = os.environ.get('PSDD_MASKS', r'D:\雷达\数据库\PSDDv1.0\论文所用数据集\位置矩阵\总')

# --------------------------------------------------------------------------- #
# minimal dataset helpers (the release keeps the experiment layout flat: one
# directory of component images per input combination, plus a mask directory and
# train/val/test lists of absolute image paths)
# --------------------------------------------------------------------------- #
def load_split(name):
    with io.open(os.path.join(os.path.dirname(DS), "%s.txt" % name), encoding="utf-8") as fh:
        return [ln.strip().replace("\\", "/") for ln in fh if ln.strip()]


def consistent_tiles(cache, margin=1.0):
    from combo_v2 import imread_u8
    out = []
    for name, (ch, bx, cf, gm) in cache.items():
        for split in ("test", "val", "train"):
            p = os.path.join(DS, "images", split, name + ".jpg")
            if os.path.exists(p):
                break
        else:
            continue
        c = imread_u8(p).astype(float)
        L = 0.114 * c[:, :, 0] + 0.587 * c[:, :, 1] + 0.299 * c[:, :, 2]
        if L[gm].mean() / max(L[~gm].mean(), 1e-6) >= margin:
            out.append(name)
    return out



ROOT = r"D:\雷达\数据库\PSDDv1.0"
T_DIR = os.path.join(ROOT, "mat全景极化")
XML_DIR = os.path.join(ROOT, "Annotations全景极化对应标签")
TILES = os.path.join(ROOT, "论文所用数据集")
SIZE, OVERLAP, MIN_T = 256, 30, 3
OUT = os.path.dirname(os.path.abspath(__file__))


def scene_of(base):
    return re.match(r"(.+)_slice_\d+$", base).group(1)


def labels_of(scene):
    r = ET.parse(os.path.join(XML_DIR, scene + ".xml")).getroot()
    sz = r.find(".//size")
    w, h = float(sz.find("width").text), float(sz.find("height").text)
    out = []
    for o in r.findall(".//object"):
        b = o.find("bndbox")
        x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        out.append(((x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h))
    return out, int(h), int(w)


def tile_origins(scene):
    labels, h, w = labels_of(scene)
    step = SIZE - OVERLAP
    rows = max(1, (h - SIZE + step - 1)//step + 1)
    cols = max(1, (w - SIZE + step - 1)//step + 1)
    out = []
    for r0 in range(rows):
        for c0 in range(cols):
            y1 = min(r0*step, h - SIZE); x1 = min(c0*step, w - SIZE)
            for cx, cy, bw, bh in labels:
                px, py, pw, ph = cx*w, cy*h, bw*w, bh*h
                ox1, oy1 = max(px-pw/2, x1), max(py-ph/2, y1)
                ox2, oy2 = min(px+pw/2, x1+SIZE), min(py+ph/2, y1+SIZE)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                if (ox2-ox1)*(oy2-oy1)/(pw*ph) <= 0.5:
                    continue
                if (pw/SIZE)*SIZE > MIN_T and (ph/SIZE)*SIZE > MIN_T:
                    out.append((y1, x1)); break
    return out


def load_T(path):
    d = sio.loadmat(path)
    shape = d["T11"].shape
    T = np.zeros(shape + (3, 3), np.complex64)
    T[..., 0, 0] = d["T11"]; T[..., 1, 1] = d["T22"]; T[..., 2, 2] = d["T33"]
    T[..., 0, 1] = T[..., 1, 0] = d["T12_real"] + 1j*d["T12_imag"]
    T[..., 0, 2] = T[..., 2, 0] = d["T13_real"] + 1j*d["T13_imag"]
    T[..., 1, 2] = T[..., 2, 1] = d["T23_real"] + 1j*d["T23_imag"]
    return T


def ops(block, Cc_inv):
    ev = np.linalg.eigvalsh(block)
    span = np.trace(block, axis1=-2, axis2=-1).real
    lam1 = ev[..., 2].real
    out = {
        "SPAN": span,
        "PMS": lam1,
        "Rank-1": lam1/np.maximum(span, 1e-12),
        "MPWF": np.trace(Cc_inv @ block, axis1=-2, axis2=-1).real,
        "PNF": np.linalg.eigvalsh(Cc_inv @ block)[..., 2].real,
        "RS": (np.abs(block[..., 0, 1]) + np.abs(block[..., 1, 2]))/np.maximum(span, 1e-12),
    }
    mu = ndimage.uniform_filter(span, size=31)
    mu2 = ndimage.uniform_filter(span**2, size=31)
    sd = np.sqrt(np.maximum(mu2 - mu**2, 0))
    out["CA-CFAR"] = (span - mu)/np.maximum(sd, 1e-12)
    return out


def main():
    per = collections.defaultdict(dict)
    for split in ("val", "test"):
        for name in load_split(split):
            base = os.path.basename(name)[:-4]
            per[scene_of(base)][base] = split
    store = {}
    for scene, items in sorted(per.items()):
        orig = tile_origins(scene)
        T = load_T(os.path.join(T_DIR, scene + ".mat"))
        Cc = T.reshape(-1, 3, 3).mean(0)          # scene clutter covariance
        Cc_inv = np.linalg.inv(Cc)
        n_ok = 0
        for base, split in sorted(items.items()):
            idx = int(base.rsplit("_", 1)[1])
            if idx >= len(orig):
                continue
            mp = os.path.join(MASK, base + ".png")
            if not os.path.exists(mp):
                continue
            gm = np.asarray(Image.open(mp)) > 0
            if gm.shape != (SIZE, SIZE) or gm.sum() == 0:
                continue
            y1, x1 = orig[idx]
            blk = T[y1:y1+SIZE, x1:x1+SIZE]
            st = ops(blk, Cc_inv)
            img = np.asarray(Image.open(os.path.join(TILES, "四成分jpg", "images", split,
                                                     base + ".jpg")).convert("RGB"))
            ref = {"Ps": img[:, :, 2].astype(np.float32)/255,
                   "Pm": img[:, :, 0].astype(np.float32)/255,
                   "Pv": img[:, :, 1].astype(np.float32)/255}
            ref["Pmv"] = np.exp(-0.21/np.maximum(ref["Pm"]+ref["Pv"], 1e-6))
            store[base] = dict(split=split, gm=gm, stats=st, refs=ref)
            n_ok += 1
        print("%-14s origins %3d  tiles used %3d" % (scene, len(orig), n_ok))
    print("tiles stored: %d" % len(store))
    cache = {n: (np.zeros(v["gm"].shape, np.uint8), None, None, v["gm"]) for n, v in store.items()}
    keep = set(consistent_tiles(cache))
    print("label-consistent: %d" % len(keep))

    names = list(next(iter(store.values()))["stats"].keys()) + ["Ps", "Pm", "Pv", "Pmv"]
    thr, auc_val = {}, {}
    for n in names:
        v, l = [], []
        for base, rec in store.items():
            if rec["split"] != "val":
                continue
            src = rec["stats"] if n in rec["stats"] else rec["refs"]
            v.append(src[n].ravel()); l.append(rec["gm"].ravel().astype(np.int8))
        v = np.concatenate(v); l = np.concatenate(l)
        fpr, tpr, t = roc_curve(l, v)
        j = int(np.argmax(tpr - fpr))
        auc_val[n], thr[n] = float(roc_auc_score(l, v)), float(t[j])
    print()
    print("%-8s %9s %7s %7s %7s %7s" % ("method", "val AUC", "prec", "rec", "F1", "IoU"))
    out = {}
    for n in names:
        tp = fp = fn = 0
        for base, rec in store.items():
            if base not in keep or rec["split"] != "test":
                continue
            src = rec["stats"] if n in rec["stats"] else rec["refs"]
            pred = src[n] >= thr[n]
            tp += int((pred & rec["gm"]).sum()); fp += int((pred & ~rec["gm"]).sum())
            fn += int((~pred & rec["gm"]).sum())
        P = tp/(tp+fp) if tp+fp else 0.0
        R = tp/(tp+fn) if tp+fn else 0.0
        F1 = 2*P*R/(P+R) if P+R else 0.0
        IoU = tp/(tp+fp+fn) if tp+fp+fn else 0.0
        out[n] = dict(auc_val=auc_val[n], thr=thr[n], prec=P, rec=R, f1=F1, iou=IoU)
        print("%-8s %9.4f %7.3f %7.3f %7.3f %7.3f" % (n, auc_val[n], P, R, F1, IoU))
    with io.open(os.path.join(OUT, "classic_ops.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote classic_ops.json")


if __name__ == "__main__":
    main()
