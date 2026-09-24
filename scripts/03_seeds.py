#!/usr/bin/env python
"""Repeat the Stage III search over the five random seeds of Table II.

    python scripts/03_seeds.py

For the low-complexity operator the paper reports that all five runs return the
same functional form, that the fitted constant spans 0.204-0.214 with mean
0.2098, and that the test AUC is identical (0.9772) in all five runs. This script
reproduces that: it fits each seed, extracts the smallest expression of the
reported complexity, and prints the constant and AUC per run.

Runtime is the sum of five full searches (each 100 populations x 300
generations), so budget accordingly.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, evaluate, sr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--complexity", type=int, default=6,
                    help="the low-complexity inflection point (paper: 6)")
    args = ap.parse_args()

    cfg = config.load(args.config)
    work = cfg["paths"]["work_dir"]
    d = np.load(os.path.join(work, "teacher_predictions.npz"), allow_pickle=True)
    X_train, y_train, X_test, y_test = (d["X_train"], d["soft_train"],
                                        d["X_test"], d["y_test"])

    rows = []
    for seed in cfg["symbolic_regression"]["seeds"]:
        run_dir = os.path.join(work, "sr_runs", f"seed_{seed}")
        print(f"\n=== seed {seed} ===")
        model = sr.search(X_train, y_train, cfg, variable_names=["Ps", "Pm", "Pv"],
                          operator_set="full_with_log", seed=seed, run_dir=run_dir)
        best = sr.best_within(model, args.complexity)
        if best is None:
            print("  no expression at or below the complexity cap")
            continue
        c, loss, eq = best
        auc = evaluate.score(y_test, model.predict(X_test))[2]
        rows.append((seed, int(c), loss, eq, auc))
        print(f"  complexity {int(c)} | loss {loss:.6f} | test AUC {auc:.4f}\n  {eq}")

    if not rows:
        raise SystemExit("no runs completed")

    print("\nseed  complexity      loss   test AUC   expression")
    for seed, c, loss, eq, auc in rows:
        print(f"{seed:>4}  {c:>10}  {loss:.6f}    {auc:.4f}   {eq}")

    structures = {eq for _, _, _, eq, _ in rows}
    aucs = [a for *_, a in rows]
    print(f"\ndistinct structures: {len(structures)}")
    print(f"test AUC: identical in all runs = {np.ptp(aucs) < 1e-9} "
          f"(spread {np.ptp(aucs):.2e})")

    import csv
    out = os.path.join(work, "seeds_summary.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "complexity", "loss", "test_auc", "expression"])
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
