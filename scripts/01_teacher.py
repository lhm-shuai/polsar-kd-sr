#!/usr/bin/env python
"""Stage II: train the teacher and export the soft labels.

    python scripts/01_teacher.py

Writes into <work_dir>:
    teacher_predictions.npz   X, y, soft, scene names for train and test
    teacher_history.npz       per-epoch losses and validation accuracy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data, teacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = config.load(args.config)
    work = cfg["paths"]["work_dir"]

    print("building the pixel dataset ...")
    X, y, names, coords = data.build_dataset(cfg)
    train_scenes, test_scenes = data.scene_split(names, cfg)
    X_train, y_train, X_val, y_val = data.pixel_split(X, y, names, train_scenes, cfg)

    te = np.isin(names, test_scenes)
    X_test, y_test = X[te], y[te]
    test_scene_of_pixel = names[te]          # lets 04_evaluate.py score CFAR
    coords_test = coords[te]                 # (row, col) of every test pixel
    print(f"train {len(y_train)} | val {len(y_val)} | test {len(y_test)}")

    print("training the teacher ...")
    predict, history, _ = teacher.train_teacher(X_train, y_train, X_val, y_val, cfg)

    soft_train = predict(X_train)
    soft_test = predict(X_test)
    print(f"soft labels: train range [{soft_train.min():.3f}, {soft_train.max():.3f}]")

    np.savez_compressed(
        os.path.join(work, "teacher_predictions.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        soft_train=soft_train, soft_test=soft_test,
        train_scenes=train_scenes, test_scenes=test_scenes,
        test_scene_of_pixel=test_scene_of_pixel,
        coords_test=coords_test,
    )
    np.savez_compressed(os.path.join(work, "teacher_history.npz"), **history)
    print("wrote", os.path.join(work, "teacher_predictions.npz"))


if __name__ == "__main__":
    main()
