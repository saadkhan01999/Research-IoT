"""Tests for calibration error, grouped bootstrap and model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from iotml.stats import (
    compare_models,
    grouped_bootstrap,
    holm,
    macro_f1_from_confusion,
    reliability,
)


def test_ece_is_zero_for_perfect_calibration_and_large_for_overconfidence() -> None:
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    ece, table = reliability(correct, conf, None)
    assert np.isclose(ece, 0.0) and len(table) == 1
    ece_bad, _ = reliability(np.array([1] * 5 + [0] * 5), np.array([0.99] * 10), None)
    assert np.isclose(ece_bad, 0.49)


def test_bootstrap_point_estimate_matches_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 900)
    pred = np.where(rng.random(900) < 0.8, y, rng.integers(0, 3, 900))
    groups = np.repeat(np.arange(30), 30)
    res = grouped_bootstrap(y, pred, np.ones(900), groups, k=3, n_boot=200, seed=1)
    assert np.isclose(res["macro_f1"], f1_score(y, pred, average="macro"))
    assert res["ci_low"] <= res["macro_f1"] <= res["ci_high"] and res["groups"] == 30


def test_macro_f1_from_confusion_ignores_absent_classes() -> None:
    cm = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=float)
    assert macro_f1_from_confusion(cm) == 1.0


def test_holm_adjustment() -> None:
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_compare_models_ranks_and_tests() -> None:
    blocks = pd.DataFrame({"a": [0.9, 0.8, 0.85, 0.7, 0.95, 0.88],
                           "b": [0.8, 0.7, 0.75, 0.6, 0.85, 0.78],
                           "c": [0.7, 0.75, 0.6, 0.65, 0.8, 0.7]})
    res = compare_models(blocks)
    assert res["best"] == "a" and res["mean_ranks"]["a"] == 1.0
    assert "friedman_p" in res and res["friedman_p"] < 0.05
    assert len(res["wilcoxon_vs_best"]) == 2


def test_bootstrap_keeps_every_class_when_a_class_has_one_file() -> None:
    """D-040: one file per group, one class per file. Class 2 has a single file and is hard
    (half its rows wrong). A plain group resample leaves that file out in about a third of the
    resamples, averages over two easy classes only and pushes the interval above the point.
    Resampling within classes keeps class 2 in every resample."""
    rng = np.random.default_rng(0)
    groups, y, pred = [], [], []
    for g in range(40):                       # 20 files of class 0, 19 of class 1, 1 of class 2
        c = 0 if g < 20 else (1 if g < 39 else 2)
        n = 50
        yt = np.full(n, c)
        wrong = rng.random(n) < (0.5 if c == 2 else 0.05)
        yp = np.where(wrong, (yt + 1) % 3, yt)
        groups.append(np.full(n, g))
        y.append(yt)
        pred.append(yp)
    y, pred, groups = np.concatenate(y), np.concatenate(pred), np.concatenate(groups)
    res = grouped_bootstrap(y, pred, np.ones(len(y)), groups, k=3, n_boot=300, seed=2)
    assert res["strata"] == 3 and res["single_group_strata"] == 1 and res["mixed_groups"] == 0
    assert res["ci_low"] <= res["macro_f1"] <= res["ci_high"]
    # every resample still contains the hard class, so no resample reaches the easy-class score
    easy_only = f1_score(y[y < 2], pred[y < 2], labels=[0, 1], average="macro")
    assert res["ci_high"] < easy_only
