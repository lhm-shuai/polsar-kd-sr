#!/usr/bin/env python
"""Build the sliced detector dataset, then train and evaluate one detector.

    # build the baseline dataset [Pm, Pv, Ps]
    python scripts/05_detector.py build --channels Pm,Pv,Ps --out data/yolo_PmPvPs

    # the same scenes with the mined descriptor in place of P_m
    python scripts/05_detector.py build --channels Pmv,Pv,Ps --out data/yolo_PmvPvPs

    # train
    python scripts/05_detector.py train --data-yaml configs/detector_data.yaml \\
        --model yolov8n.yaml --name yolov8n_PmvPvPs

Note that Ultralytics is installed as a dependency and is not vendored here;
it is AGPL-3.0, see the README.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="slice scenes and write a YOLO dataset")
    b.add_argument("--config", default=None)
    b.add_argument("--channels", default="Pm,Pv,Ps")
    b.add_argument("--out", required=True)
    b.add_argument("--size", type=int, default=256)
    b.add_argument("--overlap", type=int, default=30)

    t = sub.add_parser("train", help="train one detector")
    t.add_argument("--config", default=None)
    t.add_argument("--data-yaml", required=True)
    t.add_argument("--model", default=None)
    t.add_argument("--name", default=None)

    v = sub.add_parser("val", help="evaluate a trained detector")
    v.add_argument("--config", default=None)
    v.add_argument("--weights", required=True)
    v.add_argument("--data-yaml", required=True)

    args = ap.parse_args()
    cfg = config.load(args.config)

    if args.cmd == "build":
        detector.build_yolo_dataset(
            cfg["paths"]["components_dir"], cfg["paths"]["annotations_dir"],
            args.out, tuple(args.channels.split(",")),
            size=args.size, overlap=args.overlap)
    elif args.cmd == "train":
        detector.train_detector(cfg, args.data_yaml, args.model, args.name)
    else:
        detector.evaluate_detector(args.weights, args.data_yaml, cfg)


if __name__ == "__main__":
    main()
