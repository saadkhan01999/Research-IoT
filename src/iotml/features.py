"""Feature matrices for the models and the training-only preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """Drops copy columns and fills non-finite values with medians learned on training data.

    `fit` sees only the training matrix; `transform` applies the same columns and medians to
    any other matrix, so no statistic of validation or test data reaches the model.
    """

    drop: list[str]
    columns: list[str] = field(default_factory=list)
    medians: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> Preprocessor:
        self.columns = [c for c in frame.columns if c not in self.drop]
        x = self._raw(frame)
        self.medians = np.nanmedian(x, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("Preprocessor.transform called before fit")
        x = self._raw(frame)
        bad = np.isnan(x)
        if bad.any():
            x[bad] = np.take(self.medians, np.nonzero(bad)[1])
        return x.astype(np.float32)

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        # float32 throughout: the cached samples are float32 already, and a float64 copy of
        # 1.7 million window rows would not fit beside the models in 8 GB of RAM.
        x = frame[self.columns].to_numpy(dtype=np.float32, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x
