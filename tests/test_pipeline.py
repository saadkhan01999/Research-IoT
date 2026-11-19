"""Tests for splits, the leakage rules of preprocessing, metrics and the model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from iotml.features import Preprocessor
from iotml.metrics import evaluate
from iotml.models import ContiguousLabels, sample_params
from iotml.splits import TEST, TRAIN, VAL, random_split

SHARES = {"train": 0.7, "val": 0.1, "test": 0.2}


# ------------------------------------------------------------------ splits
def test_random_split_covers_every_row_once_and_is_stratified() -> None:
    strata = np.array([0] * 1000 + [1] * 100 + [2] * 10)
    parts = random_split(strata, SHARES, np.random.default_rng(0))
    assert set(np.unique(parts)) == {TRAIN, VAL, TEST}          # nothing unused
    for value, size in ((0, 1000), (1, 100), (2, 10)):
        p = parts[strata == value]
        assert (p == TRAIN).sum() == round(0.7 * size)
        assert (p == TEST).sum() == size - round(0.8 * size)


def test_random_split_is_reproducible_and_seed_dependent() -> None:
    strata = np.repeat(np.arange(5), 50)
    a = random_split(strata, SHARES, np.random.default_rng(1))
    b = random_split(strata, SHARES, np.random.default_rng(1))
    c = random_split(strata, SHARES, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_split_shares_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        random_split(np.zeros(10), {"train": 0.7, "val": 0.1, "test": 0.1},
                     np.random.default_rng(0))


# ------------------------------------------------------------------ leakage rules
def test_preprocessor_learns_only_from_training_rows() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.inf], "copy": [9.0, 9.0, 9.0, 9.0]})
    test = pd.DataFrame({"a": [np.nan, 100.0, 100.0, 100.0], "copy": [0.0] * 4})
    prep = Preprocessor(drop=["copy"]).fit(train)
    assert prep.columns == ["a"]                                # copy column dropped
    out = prep.transform(test)
    assert out.dtype == np.float32
    assert out[0, 0] == 2.0                    # filled with the TRAIN median (of 1, 2, 3), not 100
    assert np.array_equal(prep.transform(train)[:, 0], np.array([1, 2, 3, 2], np.float32))


def test_preprocessor_refuses_to_transform_before_fit() -> None:
    with pytest.raises(RuntimeError):
        Preprocessor(drop=[]).transform(pd.DataFrame({"a": [1.0]}))


# ------------------------------------------------------------------ metrics
def test_metrics_match_scikit_learn_without_weights() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 4, 500))
    m = evaluate(y, pred, None, ["a", "b", "c", "d"])
    assert np.isclose(m["accuracy"], accuracy_score(y, pred))
    assert np.isclose(m["macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(m["mcc"], matthews_corrcoef(y, pred))


def test_metrics_use_weights_for_the_natural_mix() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 1])               # one class-1 row wrong
    w = np.array([9.0, 9.0, 1.0, 1.0])          # class 0 is nine times as common in reality
    m = evaluate(y, pred, w, ["a", "b"])
    assert np.isclose(m["accuracy"], 19 / 20)   # natural: 18 + 1 correct out of 20
    assert np.isclose(m["unweighted_accuracy"], 3 / 4)
    assert np.isclose(m["per_class"]["b"]["recall"], 0.5)
    assert np.isclose(m["per_class"]["a"]["precision"], 18 / 19)


def test_metrics_skip_classes_absent_from_test() -> None:
    m = evaluate(np.array([0, 0, 1]), np.array([0, 1, 1]), None, ["a", "b", "c"])
    assert m["classes_in_test"] == 2 and set(m["per_class"]) == {"a", "b"}


# ------------------------------------------------------------------ models
def test_contiguous_labels_maps_back_to_original_codes() -> None:
    class Echo:
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            self.y = y

        def predict(self, x: np.ndarray) -> np.ndarray:
            return self.y

    model = ContiguousLabels(Echo()).fit(np.zeros((3, 1)), np.array([5, 9, 5]))
    assert model.model.y.tolist() == [0, 1, 0]
    assert model.predict(np.zeros((3, 1))).tolist() == [5, 9, 5]


def test_sample_params_is_reproducible_and_inside_the_space() -> None:
    space = {"depth": [2, 4, 8], "leaf": [1, 5]}
    a = sample_params(space, 4, np.random.default_rng(7))
    assert a == sample_params(space, 4, np.random.default_rng(7))
    assert len({tuple(sorted(p.items())) for p in a}) == 4
    assert all(p["depth"] in space["depth"] and p["leaf"] in space["leaf"] for p in a)
    assert len(sample_params(space, 50, np.random.default_rng(0))) == 6   # whole grid


@pytest.mark.parametrize("name", ["decision_tree", "random_forest", "extra_trees", "xgboost",
                                  "lightgbm"])
@pytest.mark.parametrize("labels", [(0, 1), (2, 5, 7, 11)])
def test_one_pass_predictions_equal_predict(name: str, labels: tuple[int, ...]) -> None:
    """predict_with_proba (arg-max of predict_proba, decision D-027) gives the same labels as
    predict() for every model, also with labels that are not 0..k-1 and with overlapping
    classes, where the probabilities are not all near 0 or 1."""
    from iotml.experiment import predict_with_proba
    from iotml.models import make_model

    rng = np.random.default_rng(3)
    y = rng.choice(np.array(labels), size=1500)
    x = (rng.normal(size=(1500, 6)) + 0.6 * y[:, None]).astype(np.float32)
    params = {"n_estimators": 30} if name != "decision_tree" else {"max_depth": 6}
    if name == "lightgbm":
        params["verbose"] = -1
    model = make_model(name, params, seed=0, n_jobs=1).fit(x[:1000], y[:1000])
    pred, proba = predict_with_proba(model, x[1000:])
    assert proba is not None and proba.shape == (500, len(labels))
    assert np.array_equal(pred, model.predict(x[1000:]))


def test_family_mix_gives_the_share_of_each_label_per_predicted_family() -> None:
    import numpy as np

    from iotml.results import family_mix

    row_label = np.array([0, 0, 0, 0, 2, 2, 5], dtype=np.int8)
    y_pred = np.array([1, 1, 0, 2, 2, 2, 0], dtype=np.int8)
    mix = family_mix(row_label, y_pred, n_families=4)
    assert list(mix.index) == [0, 2, 5]
    assert list(mix.columns) == [0, 1, 2, 3]
    assert mix.loc[0].tolist() == [0.25, 0.5, 0.25, 0.0]
    assert mix.loc[2].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert mix.loc[5].tolist() == [1.0, 0.0, 0.0, 0.0]
