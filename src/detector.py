"""Detection stage: build the sliced dataset, then train a detector on it.

The paper's detector experiments use Ultralytics for the YOLO family and
MMDetection for RetinaNet / SSD / PAA / Faster R-CNN. Both are ordinary
third-party trainers driven from here; nothing in this file reimplements them.

Two things to be aware of when you run this:

* Ultralytics is AGPL-3.0. This repository calls it as an installed dependency
  and does not vendor its source, which is what the licence expects. If you copy
  Ultralytics files into your own tree, the AGPL applies to your tree too.
* The channel layout is [Pm, Pv, Ps] (R, G, B) for the baseline, and the
  descriptors replace the P_m channel: [Pmv, Pv, Ps] and so on. The dataset
  builder below writes those jpg triplets; the yaml points at them.
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# dataset: slice scenes into tiles, keeping only tiles that contain a target
# --------------------------------------------------------------------------- #
def _parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find(".//size")
    if size is None:
        return []
    w, h = int(size.find("width").text), int(size.find("height").text)
    if w == 0 or h == 0:
        return []
    out = []
    for obj in root.findall(".//object"):
        if obj.find("name").text != "ship":
            continue
        b = obj.find("bndbox")
        if b is None:
            continue
        x1 = float(b.find("xmin").text)
        y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text)
        y2 = float(b.find("ymax").text)
        out.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
    return out


def slice_scene(img, labels, size=256, overlap=30, min_target=3):
    """Tiles of `size` with `overlap`, keeping tiles with >=1 usable target."""
    h, w = img.shape[:2]
    step = size - overlap
    rows = max(1, (h - size + step - 1) // step + 1)
    cols = max(1, (w - size + step - 1) // step + 1)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y1 = min(r * step, h - size)
            x1 = min(c * step, w - size)
            y2, x2 = y1 + size, x1 + size
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] != size or crop.shape[1] != size:
                crop = cv2.copyMakeBorder(crop, 0, size - crop.shape[0],
                                          0, size - crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            kept = []
            for cx, cy, bw, bh in labels:
                px, py = cx * w, cy * h
                pw, ph = bw * w, bh * h
                ox1, oy1 = max(px - pw / 2, x1), max(py - ph / 2, y1)
                ox2, oy2 = min(px + pw / 2, x2), min(py + ph / 2, y2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                inter = (ox2 - ox1) * (oy2 - oy1)
                if inter / (pw * ph) <= 0.5:
                    continue
                nw, nh = pw / size, ph / size
                if nw * size > min_target and nh * size > min_target:
                    kept.append([0, np.clip((px - x1) / size, .001, .999),
                                 np.clip((py - y1) / size, .001, .999),
                                 np.clip(nw, .001, .999), np.clip(nh, .001, .999)])
            if kept:
                tiles.append((crop, kept))
    return tiles


def build_yolo_dataset(components_dir, annotations_dir, out_dir, channels,
                       size=256, overlap=30, ratios=(0.7, 0.2, 0.1), seed=42):
    """Write a YOLO dataset whose three channels are `channels`, e.g.
    ("Pm", "Pv", "Ps") or ("Pmv", "Pv", "Ps").

    `components_dir` holds the Stage I(a) .npy files. Channels named Pmv/Psmv are
    computed from the components on the fly.
    """
    import random
    from glob import glob
    random.seed(seed)

    tmp = os.path.join(out_dir, "_tmp")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    pairs = []
    for p in sorted(glob(os.path.join(components_dir, "*.npy"))):
        scene = os.path.basename(p)[:-4]
        xml = os.path.join(annotations_dir, scene + ".xml")
        if os.path.exists(xml):
            pairs.append((p, xml, scene))

    written = []
    for npy_path, xml_path, scene in pairs:
        arr = np.load(npy_path)
        Ps, Pm, Pv = arr[:, :, 2], arr[:, :, 0], arr[:, :, 1]
        from .evaluate import pmv, psmv
        src = {"Ps": Ps, "Pm": Pm, "Pv": Pv,
               "Pmv": pmv(Ps, Pm, Pv), "Psmv": psmv(Ps, Pm, Pv)}
        stack = np.stack([src[c] for c in channels], axis=2)
        # every channel is a power component in [0, 1] (the descriptors are
        # bounded too), so the rendering is a straight 8-bit rescale
        img = np.round(255 * np.clip(stack, 0, 1)).astype(np.uint8)
        labels = _parse_voc(xml_path)
        if not labels:
            continue
        for i, (crop, kept) in enumerate(slice_scene(img, labels, size, overlap)):
            stem = f"{scene}_slice_{i:04d}"
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            buf.tofile(os.path.join(tmp, "images", stem + ".jpg"))
            with open(os.path.join(tmp, "labels", stem + ".txt"), "w") as f:
                for row in kept:
                    f.write(" ".join(str(v) for v in row) + "\n")
            written.append(stem)

    random.shuffle(written)
    n = len(written)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    splits = {"train": written[:n_tr],
              "val": written[n_tr:n_tr + n_va],
              "test": written[n_tr + n_va:]}
    for sub, stems in splits.items():
        for kind in ("images", "labels"):
            d = os.path.join(out_dir, kind, sub)
            os.makedirs(d, exist_ok=True)
            for stem in stems:
                ext = ".jpg" if kind == "images" else ".txt"
                shutil.copy2(os.path.join(tmp, kind, stem + ext),
                             os.path.join(d, stem + ext))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"tiles: {n} -> train {len(splits['train'])}, "
          f"val {len(splits['val'])}, test {len(splits['test'])}")
    return splits


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_detector(cfg: dict, data_yaml: str, model: str | None = None,
                   name: str | None = None):
    """Train one Ultralytics detector with the paper's protocol."""
    from ultralytics import YOLO

    dcfg = cfg["detector"]
    model = model or dcfg["model"]
    yolo = YOLO(model)
    return yolo.train(
        data=data_yaml,
        epochs=dcfg["epochs"],
        imgsz=dcfg["imgsz"],
        batch=dcfg["batch"],
        optimizer=dcfg["optimizer"],
        single_cls=dcfg["single_cls"],
        device=dcfg["device"],
        seed=dcfg["seed"],
        name=name or os.path.splitext(os.path.basename(model))[0],
    )


def evaluate_detector(weights: str, data_yaml: str, cfg: dict):
    from ultralytics import YOLO
    return YOLO(weights).val(data=data_yaml, imgsz=cfg["detector"]["imgsz"],
                             single_cls=cfg["detector"]["single_cls"])
