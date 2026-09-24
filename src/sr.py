"""Stage III - the symbolic regression search.

One entry point covers all four settings used in the paper, because they differ
only in what is fitted and which operators are allowed:

    target="soft"  operators="full"    -> the main result (Section III-B/III-C)
    target="soft"  operators="dual"    -> [Pm, Pv] feature-subset ablation
    target="hard"  operators="full"    -> Direct-SR ablation (Section III-E)
    target="soft"  operators=<set>     -> operator-set ablation (Fig. 14)

Both the KD and the Direct-SR searches optimise the same bi-objective problem:
fitting error against expression size. The Pareto front is what is returned.
"""
from __future__ import annotations

import os

import numpy as np
from pysr import PySRRegressor

# the three operator sets behind Fig. 14
OPERATOR_SETS = {
    "arithmetic": {"binary": ["+", "-", "*", "/"], "unary": []},
    "with_exp": {"binary": ["+", "-", "*", "/"], "unary": ["exp"]},
    "full": {"binary": ["+", "-", "*", "/"],
             "unary": ["exp", "sin", "cos", "square", "sqrt"]},
    "full_with_log": {"binary": ["+", "-", "*", "/"],
                      "unary": ["square", "sqrt", "log", "exp", "sin", "cos"]},
}


def build_regressor(cfg: dict, variable_names, operator_set="full",
                    seed=None, run_dir=None) -> PySRRegressor:
    """PySR instance with the settings read back from the archived checkpoint."""
    sr = cfg["symbolic_regression"]
    ops = OPERATOR_SETS[operator_set]
    seed = sr["random_state"] if seed is None else seed

    kwargs = dict(
        niterations=sr["niterations"],
        populations=sr["populations"],
        population_size=sr["population_size"],
        maxsize=sr["maxsize"],
        binary_operators=ops["binary"],
        unary_operators=ops["unary"],
        elementwise_loss=sr["elementwise_loss"],
        model_selection=sr["model_selection"],
        batching=sr["batching"],
        batch_size=sr["batch_size"],
        procs=sr["procs"],
        random_state=seed,
        progress=True,
    )
    if sr.get("maxdepth"):
        kwargs["maxdepth"] = sr["maxdepth"]
    # per-operator nesting limits, restricted to the operators actually allowed
    cons = {k: tuple(v) if isinstance(v, list) else v
            for k, v in (sr.get("constraints") or {}).items()
            if k in (ops["binary"] + ops["unary"])}
    if cons:
        kwargs["constraints"] = cons
    if run_dir:
        kwargs["output_directory"] = run_dir
        os.makedirs(run_dir, exist_ok=True)

    return PySRRegressor(**kwargs)


def search(X, y, cfg: dict, variable_names, operator_set="full",
           seed=None, run_dir=None):
    """Run one search and return the fitted PySR model.

    `variable_names` is a data-dependent parameter in PySR 1.5+, so it is passed
    to fit rather than to the constructor.
    """
    model = build_regressor(cfg, variable_names, operator_set, seed, run_dir)
    names = list(variable_names)
    try:
        model.fit(X, y, variable_names=names)
    except TypeError:                       # older PySR
        model.set_params(variable_names=names)
        model.fit(X, y)
    return model


def pareto_front(model: PySRRegressor):
    """(complexity, loss, equation) triples sorted by complexity."""
    eqs = model.equations_
    cols = {c.lower(): c for c in eqs.columns}
    comp = eqs[cols["complexity"]].to_numpy()
    loss = eqs[cols["loss"]].to_numpy()
    text = eqs[cols["equation"]].astype(str).to_numpy()
    order = np.argsort(comp)
    return list(zip(comp[order], loss[order], text[order]))


def best_within(model: PySRRegressor, max_complexity: int):
    """Lowest-loss expression at or below a size cap - the 'complexity
    inflection point' rule the paper uses to pick the general-purpose operator."""
    front = [row for row in pareto_front(model) if row[0] <= max_complexity]
    return min(front, key=lambda r: r[1]) if front else None
