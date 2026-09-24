#!/usr/bin/env python
"""Stage III: run the symbolic regression search and print the Pareto front.

    # the main result: fit the teacher's soft labels
    python scripts/02_search.py

    # the Direct-SR ablation: fit the hard labels instead
    python scripts/02_search.py --target hard --tag direct_sr

    # the [Pm, Pv] feature-subset ablation
    python scripts/02_search.py --target soft --features Pm,Pv --tag dual_feature

    # the operator-set ablation behind Fig. 14
    python scripts/02_search.py --operators arithmetic --tag ops_arithmetic
    python scripts/02_search.py --operators with_exp   --tag ops_with_exp
    python scripts/02_search.py --operators full       --tag ops_full

Requires 01_teacher.py to have run for the soft-label targets.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, sr  # noqa: E402

FEATURE_ORDER = {"Ps": 0, "Pm": 1, "Pv": 2}   # columns of the saved X


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", choices=["soft", "hard"], default="soft",
                    help="fit the teacher's soft labels, or the hard labels")
    ap.add_argument("--features", default="Ps,Pm,Pv",
                    help="comma-separated subset, e.g. Pm,Pv")
    ap.add_argument("--operators", default="full_with_log",
                    choices=sorted(sr.OPERATOR_SETS),
                    help="operator set (see src/sr.py)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None, help="output sub-directory name")
    args = ap.parse_args()

    cfg = config.load(args.config)
    work = cfg["paths"]["work_dir"]
    tag = args.tag or f"{args.target}_{args.operators}"
    run_dir = os.path.join(work, "sr_runs", tag)

    npz = os.path.join(work, "teacher_predictions.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"{npz} not found - run scripts/01_teacher.py first")
    d = np.load(npz, allow_pickle=True)

    feats = [f.strip() for f in args.features.split(",")]
    cols = [FEATURE_ORDER[f] for f in feats]
    X_train = d["X_train"][:, cols]
    X_test = d["X_test"][:, cols]

    if args.target == "soft":
        y_fit = d["soft_train"]
    else:
        y_fit = d["y_train"].astype(float)

    print(f"search: target={args.target} features={feats} "
          f"operators={args.operators} n={len(y_fit)}")
    model = sr.search(X_train, y_fit, cfg, variable_names=feats,
                      operator_set=args.operators, seed=args.seed, run_dir=run_dir)

    print("\nPareto front (complexity, loss, equation):")
    for c, loss, eq in sr.pareto_front(model):
        print(f"  {int(c):>3}  {loss:.6f}  {eq}")

    best = sr.best_within(model, cfg["symbolic_regression"]["maxsize"])
    if best:
        c, loss, eq = best
        print(f"\nselected: complexity {int(c)}, loss {loss:.6f}\n  {eq}")

    try:
        model.equations_.to_csv(os.path.join(run_dir, "hall_of_fame.csv"), index=False)
    except Exception as exc:                       # pragma: no cover
        print("could not write hall_of_fame.csv:", exc)

    # keep the test-set scores alongside the expression for later evaluation
    if best:
        np.savez_compressed(os.path.join(run_dir, "prediction.npz"),
                            X_test=X_test, y_test=d["y_test"],
                            y_hat=model.predict(X_test), complexity=int(best[0]),
                            equation=str(best[2]))
    print("run written to", run_dir)


if __name__ == "__main__":
    main()
