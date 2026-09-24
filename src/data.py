"""Dataset construction and the scene-level split.

Ported from the scripts that produced the paper's experiments. The two things
worth knowing about this module:

*Channel order.* The Stage I(a) .npy files store the components as
``[Pm, Pv, Ps]`` in channels ``[0, 1, 2]``, which is also ``[R, G, B]`` of the
component rendering. That order is load-bearing: the BT.601 luminance weights in
Stage I(b) multiply R, G, B. The teacher, however, consumes the features in the
order ``(Ps, Pm, Pv)``, which is the order the archived models were trained with.
Both orders are handled explicitly below rather than implicitly.

*Split level.* The split is over scenes, not pixels, so no scene contributes to
two splits.
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_mask(png_path: str) -> np.ndarray:
    """Read a Stage I(b) mask as a 0/1 array."""
    if not os.path.exists(png_path):
        raise FileNotFoundError(png_path)
    img = Image.open(png_path)
    if img.mode != "L":
        img = img.convert("L")
    return (np.array(img) > 0).astype(np.uint8)


def load_components(npy_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a Stage I(a) file, returning (Ps, Pm, Pv) - the teacher's order."""
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr.shape}")
    pm = arr[:, :, 0].astype(np.float32)   # R
    pv = arr[:, :, 1].astype(np.float32)   # G
    ps = arr[:, :, 2].astype(np.float32)   # B
    return ps, pm, pv


def sample_from_scene(ps, pm, pv, mask, max_ship: int, bg_ratio: float,
                      rng: np.random.RandomState):
    """Draw up to `max_ship` ship pixels and `bg_ratio` times as many background
    pixels. Every pixel outside an annotation box is background, because the mask
    is zero there by construction.

    Returns (X, y, coords); coords are the (row, col) of each sample, which the
    evaluation stage needs in order to look up a per-pixel score such as CFAR.
    """
    ship_rows, ship_cols = np.where(mask == 1)
    n_ship = len(ship_rows)
    if n_ship == 0:
        return None
    if n_ship > max_ship:
        idx = rng.choice(n_ship, max_ship, replace=False)
        ship_rows, ship_cols = ship_rows[idx], ship_cols[idx]
        n_ship = max_ship

    X_ship = np.column_stack([ps[ship_rows, ship_cols],
                              pm[ship_rows, ship_cols],
                              pv[ship_rows, ship_cols]])
    y_ship = np.ones(n_ship, dtype=np.int8)
    c_ship = np.column_stack([ship_rows, ship_cols])

    bg_rows, bg_cols = np.where(mask == 0)
    n_bg = min(int(n_ship * bg_ratio), len(bg_rows))
    if n_bg == 0:
        return X_ship, y_ship, c_ship
    bg_idx = rng.choice(len(bg_rows), n_bg, replace=False)
    bg_rows, bg_cols = bg_rows[bg_idx], bg_cols[bg_idx]
    X_bg = np.column_stack([ps[bg_rows, bg_cols],
                            pm[bg_rows, bg_cols],
                            pv[bg_rows, bg_cols]])
    y_bg = np.zeros(n_bg, dtype=np.int8)
    c_bg = np.column_stack([bg_rows, bg_cols])

    X = np.vstack([X_ship, X_bg])
    y = np.hstack([y_ship, y_bg])
    coords = np.vstack([c_ship, c_bg])
    order = rng.permutation(len(y))
    return X[order], y[order], coords[order]


def build_dataset(cfg: dict):
    """Collect per-pixel samples from every scene that has both a mask and a
    component file. Returns X, y, sample→scene name, and sample→(row, col)."""
    samp = cfg["sampling"]
    rng = np.random.RandomState(samp["seed"])
    comp_dir = cfg["paths"]["components_dir"]
    mask_dir = cfg["paths"]["masks_dir"]

    Xs, ys, names, coords = [], [], [], []
    scenes = []
    for mask_path in sorted(glob(os.path.join(mask_dir, "*.png"))):
        scene = os.path.basename(mask_path).replace(".png", "")
        comp_path = os.path.join(comp_dir, scene + ".npy")
        if not os.path.exists(comp_path):
            continue
        mask = load_mask(mask_path)
        ps, pm, pv = load_components(comp_path)
        if ps.shape != mask.shape:
            continue
        drawn = sample_from_scene(ps, pm, pv, mask, samp["max_ship_per_scene"],
                                  samp["bg_ratio"], rng)
        if drawn is None:
            continue
        X_img, y_img, c_img = drawn
        Xs.append(X_img)
        ys.append(y_img)
        coords.append(c_img)
        names.extend([scene] * len(y_img))
        scenes.append(scene)

    if not Xs:
        raise RuntimeError(
            f"no usable scenes under {mask_dir!r} / {comp_dir!r}; "
            "see the README for how to run Stage I"
        )
    X = np.vstack(Xs)
    y = np.hstack(ys)
    names = np.array(names)
    coords = np.vstack(coords)
    print(f"scenes: {len(scenes)} | pixels: {len(y)} | ship: {int(y.sum())}")
    return X, y, names, coords


def scene_split(names: np.ndarray, cfg: dict):
    """Split by scene, then carve a validation slice out of the training pixels."""
    samp = cfg["sampling"]
    scenes = np.unique(names)
    train_scenes, test_scenes = train_test_split(
        scenes, test_size=samp["test_size"], random_state=samp["seed"])
    print(f"train scenes: {len(train_scenes)} | test scenes: {len(test_scenes)}")
    return train_scenes, test_scenes


def pixel_split(X, y, names, train_scenes, cfg: dict):
    """Train/val/test pixel arrays for the given training scenes."""
    samp = cfg["sampling"]
    tr = np.isin(names, train_scenes)
    X_tr, y_tr = X[tr], y[tr]
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=samp["val_split"], stratify=y_tr,
        random_state=samp["seed"])
    return X_train, y_train, X_val, y_val
