"""Row sampling inside a split partition.

Order used everywhere: split the full data first (by protocol), then sample each partition.
Training samples may be capped per label (a form of undersampling, training data only).
Test samples are capped per label too, but each row carries a weight
(rows of its label in the partition / rows sampled), so weighted metrics estimate the metric
on the whole partition with its natural class mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sample:
    """Global row ids (sorted) and the weight of each sampled row."""

    row_ids: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.row_ids.size)


def cap_per_class(row_ids: np.ndarray, classes: np.ndarray, cap: int | None,
                  rng: np.random.Generator) -> Sample:
    """At most `cap` random rows of each class (all rows when fewer), with inverse weights."""
    row_ids = np.asarray(row_ids)
    classes = np.asarray(classes)
    if row_ids.shape != classes.shape:
        raise ValueError("row_ids and classes must have the same length")
    picked, weights = [], []
    for c in np.unique(classes):
        ids = row_ids[classes == c]
        if cap is not None and ids.size > cap:
            chosen = rng.choice(ids, cap, replace=False)
            weights.append(np.full(cap, ids.size / cap))
        else:
            chosen = ids
            weights.append(np.ones(ids.size))
        picked.append(chosen)
    ids_all = np.concatenate(picked)
    w_all = np.concatenate(weights)
    order = np.argsort(ids_all, kind="stable")
    return Sample(ids_all[order], w_all[order])


def uniform(row_ids: np.ndarray, n: int | None, rng: np.random.Generator) -> Sample:
    """A plain random sample of `n` rows (natural class mix); every weight is equal."""
    row_ids = np.asarray(row_ids)
    keep_all = n is None or n >= row_ids.size
    chosen = np.sort(row_ids if keep_all else rng.choice(row_ids, n, replace=False))
    scale = row_ids.size / max(chosen.size, 1)
    return Sample(chosen, np.full(chosen.size, scale))


def seed_rng(seed: int, *names: str) -> np.random.Generator:
    """A generator for one (seed, purpose) pair, so adding a new purpose never shifts others."""
    key = [seed] + [sum(ord(ch) * (i + 1) for i, ch in enumerate(n)) for n in names]
    return np.random.default_rng(key)
