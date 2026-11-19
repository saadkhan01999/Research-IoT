"""Tests for the imbalance treatments (training data only)."""

from __future__ import annotations

import numpy as np

from iotml.imbalance import LogScaler, balanced_weights, smote_tomek, undersample


def test_balanced_weights_give_every_class_equal_total_weight() -> None:
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = balanced_weights(y)
    totals = [w[y == c].sum() for c in range(3)]
    assert np.allclose(totals, totals[0])
    assert np.isclose(w.sum(), len(y))


def test_undersample_caps_only_big_classes() -> None:
    y = np.array([0] * 100 + [1] * 5)
    keep = undersample(y, 20, np.random.default_rng(0))
    assert np.bincount(y[keep]).tolist() == [20, 5]
    assert np.all(np.diff(keep) > 0)


def test_log_scaler_round_trip() -> None:
    x = np.array([[0.0, 1e8, -3.0], [5.0, 2.0, 0.5], [1.0, 3e5, -1.0]])
    s = LogScaler().fit(x)
    assert np.allclose(s.inverse(s.transform(x)), x, rtol=1e-4, atol=1e-4)


def test_smote_tomek_raises_small_classes_and_keeps_shape() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(5, 1, (12, 3))])
    y = np.array([0] * 200 + [1] * 12)
    xr, yr = smote_tomek(x, y, target=100, seed=0)
    counts = np.bincount(yr)
    assert counts[1] >= 95                # raised to about the target (Tomek may drop a few)
    assert counts[0] <= 200               # Tomek only removes majority rows
    assert xr.shape[1] == 3 and xr.dtype == np.float32


def test_smote_tomek_gives_the_same_rows_when_run_twice() -> None:
    rng = np.random.default_rng(3)
    x = np.abs(rng.normal(size=(600, 5))) * np.array([1, 10, 100, 1, 1])
    y = np.array([0] * 500 + [1] * 60 + [2] * 40)
    a = smote_tomek(x, y, target=100, seed=7)
    b = smote_tomek(x, y, target=100, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_treat_reuses_smote_tomek_rows_for_the_next_model_only_on_the_same_data(
        monkeypatch) -> None:
    import logging

    import iotml.experiment as experiment
    from iotml.experiment import Experiment, RunSpec

    calls = []
    real = experiment.smote_tomek

    def counted(x, y, target, seed):
        calls.append(len(y))
        return real(x, y, target, seed)

    monkeypatch.setattr(experiment, "smote_tomek", counted)
    exp = Experiment.__new__(Experiment)          # no data files needed for treat()
    exp.exp = {"imbalance": {"target_per_class": 100}}
    exp.log = logging.getLogger("test")
    exp._smote_cache = None
    exp._treatment_seconds = None
    rng = np.random.default_rng(3)
    x = np.abs(rng.normal(size=(600, 5)))
    y = np.array([0] * 500 + [1] * 60 + [2] * 40)

    def spec(model: str) -> RunSpec:
        return RunSpec("random_dedup", "L8", model, "imb_smotetomek", "tuned_from_random", 7,
                       "quick")

    x1, y1, w1 = exp.treat(spec("xgboost"), x, y)
    first_seconds = exp._treatment_seconds
    x2, y2, w2 = exp.treat(spec("random_forest"), x, y)
    assert len(calls) == 1                         # the second model reused the rows
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert w1 is None and w2 is None
    assert exp._treatment_seconds == first_seconds  # the time of the treatment itself
    fresh = real(x, y, 100, 7)
    assert np.array_equal(x2, fresh[0]) and np.array_equal(y2, fresh[1])
    exp.treat(spec("random_forest"), x + 1.0, y)    # other rows: made again
    assert len(calls) == 2
