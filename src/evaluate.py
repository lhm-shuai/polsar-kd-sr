"""Pixel-level evaluation: the mined descriptors, the handcrafted baselines and
the CFAR reference.

Everything is scored the same way: ROC over the test pixels, threshold taken at
the Youden point (argmax of TPR - FPR), then accuracy and F1 at that threshold.
That is the protocol behind Table III and the ROC panels of Fig. 8.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             roc_curve)


# --------------------------------------------------------------------------- #
# the mined descriptors. Constants are the ones reported in the paper: the
# low-complexity operator uses the five-seed mean 0.20430441, the high-complexity
# one is reproduced verbatim from the archived run.
# --------------------------------------------------------------------------- #
def pmv(Ps, Pm, Pv):
    """exp(-c / (Pm + Pv)) with c = 0.20430441. Three floating-point ops."""
    denom = np.where((Pm + Pv) == 0, 1e-8, Pm + Pv)
    return np.exp(-0.20430441 / denom)


def psmv(Ps, Pm, Pv):
    """The high-complexity descriptor Psmv (Section III-B)."""
    term1 = Pv * (-0.72197366 - Ps)
    term2 = np.exp(term1)
    term3 = np.square(Pm) + 1.0250757
    term4 = term2 / term3
    term5 = np.square(np.square(np.square(term4)))
    term6 = term5 + 0.48293668
    term7 = np.cos(term6)
    term8 = np.sin(term7)
    term9 = np.sin(term8)
    term10 = np.sin(term9)
    term11 = term10 / 0.61469054
    term12 = np.square(term11)
    term13 = term12 / 0.49011374
    return np.square(np.square(np.square(term13)))


# --------------------------------------------------------------------------- #
# CFAR reference
# --------------------------------------------------------------------------- #
def cfar_score(image, guard_win=3, bg_win=10, pfa=1e-4):
    """Vectorised cell-averaging CFAR. Returns the detection score per pixel,
    with the border windows zeroed as in the reference implementation."""
    h, w = image.shape
    total = guard_win + bg_win
    size = 2 * total + 1

    mask = np.ones((size, size))
    mask[total - guard_win:total + guard_win + 1,
         total - guard_win:total + guard_win + 1] = 0
    n_valid = mask.sum()

    s1 = uniform_filter(image, size=size, mode="constant", cval=0) * size * size
    s2 = uniform_filter(image ** 2, size=size, mode="constant", cval=0) * size * size
    mu = s1 / n_valid
    sigma = np.sqrt(np.maximum(s2 / n_valid - mu ** 2, 0.0))

    mu = np.nan_to_num(mu)
    sigma[sigma == 0] = 1e-8
    thr = mu + sigma * np.sqrt(2 * np.log(1 / pfa))
    score = image / (thr + 1e-8)

    score[:total, :] = 0
    score[-total:, :] = 0
    score[:, :total] = 0
    score[:, -total:] = 0
    return score


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(y_true, y_score):
    """(accuracy %, F1, AUC) at the Youden-optimal threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    k = int(np.argmax(tpr - fpr))
    pred = (y_score >= thr[k]).astype(int)
    return (accuracy_score(y_true, pred) * 100.0,
            f1_score(y_true, pred),
            roc_auc_score(y_true, y_score))


def evaluate_all(X_test, y_test, teacher_pred, include_cfar_image=None):
    """Metric table for every method compared in Table III."""
    Ps, Pm, Pv = X_test[:, 0], X_test[:, 1], X_test[:, 2]
    methods = {
        "Ps": Ps,
        "Pm": Pm,
        "Pv": Pv,
        "Pm+Pv": Pm + Pv,
        "SPAN": Ps + Pm + Pv,
        "Teacher MLP": teacher_pred,
        "Pmv": pmv(Ps, Pm, Pv),
        "Psmv": psmv(Ps, Pm, Pv),
    }
    if include_cfar_image is not None:
        methods["CFAR"] = include_cfar_image
    return {name: score(y_test, s) for name, s in methods.items()}


def format_table(results) -> str:
    lines = [f"{'method':<14}{'Acc (%)':>9}{'F1':>9}{'AUC':>9}"]
    lines.append("-" * 41)
    for name, (acc, f1, auc) in results.items():
        lines.append(f"{name:<14}{acc:>9.2f}{f1:>9.4f}{auc:>9.4f}")
    return "\n".join(lines)


def roc_points(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr


def save_roc_csv(path, y_true, curves: dict):
    """One row per ROC point, as used to draw the paper's ROC figures."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "fpr", "tpr", "threshold", "auc"])
        for name, s in curves.items():
            auc = roc_auc_score(y_true, s)
            fpr, tpr, thr = roc_curve(y_true, s)
            for a, b, c in zip(fpr, tpr, thr):
                w.writerow([name, a, b, c, auc])
