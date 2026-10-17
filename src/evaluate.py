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
# low-complexity operator uses c = 0.20430441, the seed-666 constant that also
# rendered the detector inputs (the five seeds give 0.204-0.214; c does not
# affect any ranking-based metric), and the high-complexity one is reproduced
# verbatim from the archived run.
# --------------------------------------------------------------------------- #
def pmv(Ps, Pm, Pv):
    """exp(-c / (Pm + Pv)) with c = 0.20430441. Three floating-point ops."""
    denom = np.where((Pm + Pv) == 0, 1e-8, Pm + Pv)
    return np.exp(-0.20430441 / denom)


def psmv(Ps, Pm, Pv):
    """The paper's 13-node high-complexity reference descriptor (Sec. III-C).

    With A = exp( (-0.6271455 - Ps) * Pv ) / ( Pm^2 + 0.7921049 ):

        Psmv = cos( A^2 )

    This is the C=13 point of the seed-666 Pareto front and the expression
    behind the paper's Table IV row "Psmv (C=13, ref.)" (accuracy 95.34%,
    AUC 0.9867).  It is reported only as a pixel-level reference and is not
    used in the detector experiments.  The older C=29 transcription is kept
    below as psmv_c29_legacy for traceability of earlier logs.
    """
    inner = np.exp((-0.6271455 - Ps) * Pv) / (np.square(Pm) + 0.7921049)
    return np.cos(np.square(inner))


def psmv_c13(Ps, Pm, Pv):
    """Alias of psmv (C=13) kept for backward compatibility."""
    return psmv(Ps, Pm, Pv)


def psmv_c29_legacy(Ps, Pm, Pv):
    """Legacy C=29 transcription, retained only to replay earlier logs."""
    A = np.exp(Pv * (-0.72197366 - Ps)) / (np.square(Pm) + 1.0250757)
    t = np.cos(np.square(np.square(np.square(A)) + 0.48293668))
    t = np.sin(t) / 0.61469054
    t = np.sin(np.sin(t))
    t = np.square(t) / 0.49011374
    return np.power(np.sin(t), 8)


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
