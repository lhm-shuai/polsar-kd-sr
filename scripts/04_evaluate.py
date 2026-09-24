#!/usr/bin/env python
"""Pixel-level comparison: the mined descriptors against the handcrafted
components, the teacher, and the CFAR reference.

    python scripts/04_evaluate.py

Produces the metric table of Table III on stdout and, next to the run, a CSV of
ROC points for every method (the data behind the paper's ROC figures).

Note on two baselines that are easy to leave out and should not be:

  * P_m + P_v      The low-complexity descriptor is a strictly monotone function
                   of this sum, so it has the *same* AUC and essentially the same
                   accuracy. The value of the descriptor is that it is closed
                   form, costs three floating-point operations, and saturates
                   like a membership probability - not that it ranks better.
  * P_SPAN         The total scattering power, the classic handcrafted baseline.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cfar-images", default=None,
                    help="directory of component .npy files; adds the CFAR row")
    args = ap.parse_args()

    cfg = config.load(args.config)
    work = cfg["paths"]["work_dir"]
    d = np.load(os.path.join(work, "teacher_predictions.npz"), allow_pickle=True)

    X_test, y_test = d["X_test"], d["y_test"]
    teacher_pred = d["soft_test"]

    cfar = None
    if args.cfar_images:
        from glob import glob
        # CFAR runs per scene on SPAN; each test pixel then takes the score at
        # its own coordinates. 01_teacher.py stores both the scene and the
        # coordinates, which is what makes this lookup possible.
        if "coords_test" not in d:
            raise SystemExit("re-run scripts/01_teacher.py to store pixel "
                             "coordinates, or drop --cfar-images")
        print("computing the CFAR reference ...")
        scene_of = d["test_scene_of_pixel"]
        coords = d["coords_test"]
        cfar = np.zeros(len(y_test), dtype=np.float64)
        for p in sorted(glob(os.path.join(args.cfar_images, "*.npy"))):
            scene = os.path.basename(p)[:-4]
            sel = np.where(scene_of == scene)[0]
            if sel.size == 0:
                continue
            arr = np.load(p)
            span = arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]
            score = evaluate.cfar_score(span)
            r = np.clip(coords[sel, 0], 0, score.shape[0] - 1)
            c = np.clip(coords[sel, 1], 0, score.shape[1] - 1)
            cfar[sel] = score[r, c]
        print(f"  CFAR: {int((cfar > 0).sum())} non-zero scores")

    results = evaluate.evaluate_all(X_test, y_test, teacher_pred, cfar)
    print()
    print(evaluate.format_table(results))

    out = os.path.join(work, "roc_points.csv")
    evaluate.save_roc_csv(out, y_test, {
        "Ps": X_test[:, 0], "Pm": X_test[:, 1], "Pv": X_test[:, 2],
        "Pm+Pv": X_test[:, 1] + X_test[:, 2],
        "Teacher": teacher_pred,
        "Pmv": evaluate.pmv(*X_test.T),
        "Psmv": evaluate.psmv(*X_test.T),
    })
    print("\nwrote", out)


if __name__ == "__main__":
    main()
