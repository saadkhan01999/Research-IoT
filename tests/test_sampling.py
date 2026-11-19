"""Tests for per-class capping, weights and seeded generators."""

from __future__ import annotations

import numpy as np

from iotml.sampling import cap_per_class, seed_rng, uniform


def test_cap_keeps_small_classes_and_weights_big_ones() -> None:
    ids = np.arange(1000)
    cls = np.array([0] * 900 + [1] * 90 + [2] * 10)
    s = cap_per_class(ids, cls, cap=50, rng=np.random.default_rng(0))
    picked = cls[s.row_ids]
    assert np.bincount(picked).tolist() == [50, 50, 10]
    assert np.all(np.diff(s.row_ids) > 0)                      # sorted, no repeats
    # weights add back to the partition size of every class
    for c, total in ((0, 900), (1, 90), (2, 10)):
        assert np.isclose(s.weights[picked == c].sum(), total)


def test_weighted_accuracy_estimates_natural_mix() -> None:
    rng = np.random.default_rng(1)
    cls = np.array([0] * 9000 + [1] * 1000)
    correct = np.concatenate([np.ones(9000), np.zeros(1000)])  # class 1 always wrong
    s = cap_per_class(np.arange(10000), cls, cap=500, rng=rng)
    weighted = np.average(correct[s.row_ids], weights=s.weights)
    assert np.isclose(weighted, 0.9)                           # natural accuracy
    assert np.isclose(correct[s.row_ids].mean(), 0.5)          # unweighted is balanced


def test_uniform_and_seed_rng_are_reproducible() -> None:
    a = uniform(np.arange(100), 10, seed_rng(42, "train")).row_ids
    b = uniform(np.arange(100), 10, seed_rng(42, "train")).row_ids
    c = uniform(np.arange(100), 10, seed_rng(42, "test")).row_ids
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist()
