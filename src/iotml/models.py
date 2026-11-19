"""Model factory and the random hyperparameter search."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

TREE_MODELS = ("decision_tree", "random_forest", "extra_trees", "xgboost", "lightgbm")


class ContiguousLabels:
    """Wraps a classifier so it sees labels 0..k-1 even when some classes are absent from the
    training sample (XGBoost requires this); predictions are mapped back."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.classes_: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> ContiguousLabels:
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if sample_weight is None:
            self.model.fit(x, y_idx)
        else:
            self.model.fit(x, y_idx, sample_weight=sample_weight)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.asarray(self.model.predict(x)).astype(int).ravel()]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))


def make_model(name: str, params: dict[str, Any], seed: int, n_jobs: int) -> ContiguousLabels:
    """A fresh, unfitted model with the given hyperparameters."""
    params = dict(params)
    if name == "decision_tree":
        model: Any = DecisionTreeClassifier(random_state=seed, **params)
    elif name == "random_forest":
        model = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    elif name == "extra_trees":
        model = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, **params)
    elif name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(random_state=seed, n_jobs=n_jobs, **params)
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(random_state=seed, n_jobs=n_jobs, **params)
    else:
        raise ValueError(f"Unknown model '{name}'")
    return ContiguousLabels(model)


def sample_params(space: dict[str, list[Any]], n: int,
                  rng: np.random.Generator) -> list[dict[str, Any]]:
    """`n` distinct random settings from a grid-shaped search space (fewer if the grid is
    smaller than `n`)."""
    keys = sorted(space)
    sizes = [len(space[k]) for k in keys]
    total = int(np.prod(sizes)) if keys else 1
    picks = rng.choice(total, size=min(n, total), replace=False)
    settings = []
    for flat in picks:
        combo, rest = {}, int(flat)
        for key, size in zip(keys, sizes, strict=True):
            combo[key] = space[key][rest % size]
            rest //= size
        settings.append(combo)
    return settings
