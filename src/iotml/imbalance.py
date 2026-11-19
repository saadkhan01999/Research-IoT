"""Imbalance treatments, applied to the TRAINING sample only (after the split).

    none         train on the natural class mix
    weights      sample weights n / (k * n_class) ("balanced" class weights)
    under        random undersampling: every class capped at `target` rows
    smotetomek   SMOTE raises every class below `target` to `target`, then Tomek links are
                 removed (imbalanced-learn). SMOTE works in a log-scaled, standardised copy
                 of the features (fitted on the training sample) so that wide-range columns
                 such as Variance do not dominate the neighbour search; synthetic rows are
                 mapped back to the original scale.
"""

from __future__ import annotations

import numpy as np

TREATMENTS = ("none", "weights", "under", "smotetomek")


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Weight of each row = n / (k * rows of its class)."""
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return (len(y) / (len(classes) * counts))[inverse]


def undersample(y: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices with every class capped at `target` rows (sorted)."""
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(idx if idx.size <= target else rng.choice(idx, target, replace=False))
    return np.sort(np.concatenate(keep))


class LogScaler:
    """sign(x)*log1p(|x|), then standardised; invertible. Fitted on training rows only."""

    def fit(self, x: np.ndarray) -> LogScaler:
        z = np.sign(x) * np.log1p(np.abs(x))
        self.mean_ = z.mean(axis=0)
        self.scale_ = z.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.sign(x) * np.log1p(np.abs(x))
        return ((z - self.mean_) / self.scale_).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        u = z.astype(np.float64) * self.scale_ + self.mean_
        return (np.sign(u) * np.expm1(np.abs(u))).astype(np.float32)


def smote_tomek(x: np.ndarray, y: np.ndarray, target: int,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE up to `target` rows for the smaller classes, then Tomek-link cleaning."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks

    classes, counts = np.unique(y, return_counts=True)
    strategy = {int(c): int(target) for c, n in zip(classes, counts, strict=True)
                if 1 < n < target}
    scaler = LogScaler().fit(x)
    z = scaler.transform(x)
    if strategy:
        smallest = min(int(n) for c, n in zip(classes, counts, strict=True)
                       if int(c) in strategy)
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=min(5, smallest - 1),
                      random_state=seed)
        z, y = smote.fit_resample(z, y)
    z, y = TomekLinks(sampling_strategy="auto").fit_resample(z, y)
    return scaler.inverse(np.asarray(z)), np.asarray(y)
