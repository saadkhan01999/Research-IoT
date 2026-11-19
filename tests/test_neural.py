"""Tests for the neural models on tiny synthetic data (skipped if PyTorch is missing)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from iotml.neural import NeuralClassifier, NeuralConfig  # noqa: E402


def _data(seed: int, windows: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, 600)
    shape = (600, 5, 4) if windows else (600, 4)
    x = rng.normal(0, 1, shape).astype(np.float32)
    x[..., 0] += y * 3 if not windows else (y * 3)[:, None]   # class signal in feature 0
    return x, y


@pytest.mark.parametrize(("arch", "loss", "windows"), [("mlp", "ce", False),
                                                      ("mlp", "focal", False),
                                                      ("cnn1d", "ce", True),
                                                      ("gru", "weighted_ce", True)])
def test_neural_models_learn_a_simple_signal(arch: str, loss: str, windows: bool) -> None:
    x, y = _data(0, windows)
    xv, yv = _data(1, windows)
    cfg = NeuralConfig(arch=arch, loss=loss, epochs=30, batch_size=32, lr=0.01, hidden=16,
                       patience=30,
                       threads=1)
    clf = NeuralClassifier(cfg, seed=0).fit(x, y, xv, yv, None, n_classes=3)
    assert (clf.predict(xv) == yv).mean() > 0.8
    proba = clf.predict_proba(xv)
    assert proba.shape == (600, 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert len(cfg.history) >= 1 and clf.n_parameters() > 0


def test_training_is_reproducible_for_a_seed() -> None:
    x, y = _data(0, False)
    cfg = dict(arch="mlp", loss="ce", epochs=2, batch_size=64, hidden=8, threads=1)
    a = NeuralClassifier(NeuralConfig(**cfg), seed=3).fit(x, y, x, y, None, 3).predict_proba(x)
    b = NeuralClassifier(NeuralConfig(**cfg), seed=3).fit(x, y, x, y, None, 3).predict_proba(x)
    assert np.allclose(a, b)
