"""Load the YAML configuration and make paths absolute."""
from __future__ import annotations

import os
import yaml

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "configs", "default.yaml")


def load(path: str | None = None) -> dict:
    with open(path or DEFAULT, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(os.path.dirname(os.path.abspath(path or DEFAULT)))
    for key, val in list(cfg["paths"].items()):
        if not os.path.isabs(val):
            cfg["paths"][key] = os.path.normpath(os.path.join(root, val))
    os.makedirs(cfg["paths"]["work_dir"], exist_ok=True)
    return cfg
