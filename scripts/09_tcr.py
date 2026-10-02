# -*- coding: utf-8 -*-
"""Target-to-clutter ratio of every representation (Table VII).

TCR is the metric the group's TGRS paper reports per scene (their Table I), and
it is the quantitative form of this paper's qualitative claim that the mined
descriptor suppresses sea clutter.  Everything comes from data the paper
already uses:

  * the per-channel tile datasets (论文所用数据集/<channel>/images/<split>/),
    which hold the same 256x256 tiles as the composite set, one Yamaguchi
    component or descriptor per tile;
  * the ground-truth masks and the label-consistency screen.

TCR(tile) = 10 lg( mean power inside the mask / mean power in the tile outside
a guard band around the mask ), reported as the median over test tiles so a
single saturated pixel cannot dominate.
"""
import collections
import glob
import io
import json
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)
DS = os.environ.get('PSDD_COMPOSITE', r'D:\雷达\数据库\PSDDv1.0\论文所用数据集\四成分jpg')
MASK = os.environ.get('PSDD_MASKS', r'D:\雷达\数据库\PSDDv1.0\论文所用数据集\位置矩阵\总')

def load_split(name):
    with io.open(os.path.join(os.path.dirname(DS), "%s.txt" % name), encoding="utf-8") as fh:
        return [ln.strip().replace("\\", "/") for ln in fh if ln.strip()]


def consistent_tiles(cache, margin=1.0):
    out = []
    for name, (ch, bx, cf, gm) in cache.items():
        for split in ("test", "val", "train"):
            p = os.path.join(DS, "images", split, name + ".jpg")
            if os.path.exists(p):
                break
        else:
            continue
        c = np.asarray(Image.open(p).convert("RGB")).astype(float)
        L = 0.114 * c[:, :, 0] + 0.587 * c[:, :, 1] + 0.299 * c[:, :, 2]
        if L[gm].mean() / max(L[~gm].mean(), 1e-6) >= margin:
            out.append(name)
    return out



ROOT = r"D:\雷达\数据库\PSDDv1.0\论文所用数据集"
GUARD = 8                      # pixels of guard band between target and clutter
OUT = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ["Ps", "Pm", "Pv", "Pmv", "Psmv"]


def sensor_of(base):
    if base.startswith("GF_3"):
        return "GF-3"
    if base.startswith("AIRSAR"):
        return "AIRSAR"
    return "Radarsat-2"


def tile_dir(ch):
    return os.path.join(ROOT, ch, "images")


def main():
    names = {}
    for split in ("val", "test"):
        for name in load_split(split):
            base = os.path.basename(name)[:-4]
            names.setdefault(split, {})[base] = True

    # locate each channel's tile file per split
    files = {ch: {} for ch in CHANNELS}
    for ch in CHANNELS:
        for split in ("val", "test"):
            for f in glob.glob(os.path.join(tile_dir(ch), split, "*.jpg")):
                files[ch][os.path.basename(f)[:-4]] = f
        print("%-5s tiles: val %d, test %d"
              % (ch, sum(1 for b in files[ch] if b in names.get("val", {})),
                 sum(1 for b in files[ch] if b in names.get("test", {}))))

    # label-consistency screen, using the composite channel (Pm) tiles
    cache = {}
    for base in names["test"]:
        mp = os.path.join(MASK, base + ".png")
        if not os.path.exists(mp):
            continue
        img = np.asarray(Image.open(os.path.join(DS, "images", "test", base + ".jpg")))
        gm = np.asarray(Image.open(mp)) > 0
        if gm.shape != img.shape[:2] or gm.sum() == 0:
            continue
        cache[base] = (img[:, :, 0], None, None, gm)
    keep = set(consistent_tiles(cache))
    print("test tiles with masks: %d ; label-consistent: %d" % (len(cache), len(keep)))

    res = {}
    print()
    print("%-6s %8s %8s %8s %8s" % ("repr", "TCR med", "TCR mean", "p25", "p75"))
    for ch in CHANNELS:
        vals, by_sensor = [], collections.defaultdict(list)
        for base in sorted(keep):
            p = files[ch].get(base)
            if p is None:
                continue
            a = np.asarray(Image.open(p).convert("L")).astype(np.float32) / 255.0
            gm = cache[base][3]
            if a.shape != gm.shape:
                continue
            guard = ndimage.binary_dilation(gm, iterations=GUARD)
            clutter = ~guard
            if gm.sum() < 16 or clutter.sum() < 16:
                continue
            pt, pc = a[gm].mean(), a[clutter].mean()
            if pc <= 1e-6 or pt <= 1e-6:
                continue
            v = 10 * np.log10(pt / pc)
            vals.append(v)
            by_sensor[sensor_of(base)].append(v)
        vals = np.array(vals)
        res[ch] = dict(n=len(vals), median=float(np.median(vals)), mean=float(vals.mean()),
                       p25=float(np.percentile(vals, 25)), p75=float(np.percentile(vals, 75)),
                       by_sensor={k: float(np.median(v)) for k, v in by_sensor.items()},
                       n_by_sensor={k: len(v) for k, v in by_sensor.items()})
        print("%-6s %8.2f %8.2f %8.2f %8.2f" % (ch, res[ch]["median"], res[ch]["mean"],
                                                res[ch]["p25"], res[ch]["p75"]))
    print()
    for ch in CHANNELS:
        print("%-6s by sensor: %s" % (ch, {k: round(v, 2) for k, v in res[ch]["by_sensor"].items()}))
    with io.open(os.path.join(OUT, "tcr.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("wrote tcr.json")


if __name__ == "__main__":
    main()
