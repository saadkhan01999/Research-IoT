"""Neural models on CPU: an MLP for single rows, a 1D-CNN and a GRU for windows of
consecutive rows. Inputs are log-scaled and standardised with statistics from the training
rows only (iotml.imbalance.LogScaler). Training stops early on validation macro-F1.

Losses: "ce" (cross-entropy), "weighted_ce" (balanced class weights), "focal" (gamma=2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iotml.imbalance import LogScaler
from iotml.metrics import evaluate


@dataclass
class NeuralConfig:
    arch: str = "mlp"            # mlp | cnn1d | gru
    loss: str = "ce"             # ce | weighted_ce | focal
    epochs: int = 15
    batch_size: int = 1024
    lr: float = 1e-3
    patience: int = 3
    hidden: int = 128
    dropout: float = 0.2
    gamma: float = 2.0
    threads: int = 3
    history: list[dict[str, float]] = field(default_factory=list)


def _build(arch: str, d: int, k: int, hidden: int, dropout: float) -> Any:
    import torch.nn as nn

    if arch == "mlp":
        return nn.Sequential(nn.Linear(d, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden, k))
    if arch == "cnn1d":
        class CNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(d, hidden, 3, padding=1), nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(dropout),
                    nn.Linear(hidden, k))

            def forward(self, x: Any) -> Any:          # x: (batch, time, features)
                return self.body(x.transpose(1, 2))
        return CNN()
    if arch == "gru":
        class GRU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(d, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, k))

            def forward(self, x: Any) -> Any:
                _, h = self.gru(x)
                return self.head(h[-1])
        return GRU()
    raise ValueError(f"Unknown architecture '{arch}'")


def _loss_fn(cfg: NeuralConfig, y: np.ndarray, k: int) -> Any:
    import torch
    import torch.nn.functional as functional

    weight = None
    if cfg.loss in ("weighted_ce", "focal"):
        counts = np.bincount(y, minlength=k).astype(np.float64)
        w = np.where(counts > 0, len(y) / (k * np.maximum(counts, 1)), 0.0)
        weight = torch.tensor(w, dtype=torch.float32)
    if cfg.loss == "focal":
        def focal(logits: Any, target: Any) -> Any:
            logp = functional.log_softmax(logits, dim=1)
            logp_t = logp.gather(1, target[:, None]).squeeze(1)
            p_t = logp_t.exp()
            loss = -((1 - p_t) ** cfg.gamma) * logp_t
            if weight is not None:
                loss = loss * weight[target]
            return loss.mean()
        return focal
    return torch.nn.CrossEntropyLoss(weight=weight)


class NeuralClassifier:
    """sklearn-like wrapper: fit(x, y, x_val, y_val, w_val), predict, predict_proba.
    `x` is (rows, features) for the MLP or (windows, time, features) for CNN/GRU."""

    def __init__(self, cfg: NeuralConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.scaler = LogScaler()
        self.classes_: np.ndarray = np.array([])
        self.net: Any = None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        return self.scaler.transform(x.reshape(-1, shape[-1])).reshape(shape)

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            w_val: np.ndarray | None, n_classes: int) -> NeuralClassifier:
        """Train on (x, y); keep the epoch with the best validation macro-F1. `n_classes`
        is the size of the label scheme (class codes are 0..n_classes-1)."""
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.cfg.threads)
        rng = np.random.default_rng(self.seed)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k, d = self.classes_.size, x.shape[-1]
        names = [str(i) for i in range(n_classes)]
        self.scaler.fit(x.reshape(-1, d))
        xt = torch.from_numpy(self._scale(x))
        yt = torch.from_numpy(y_idx.astype(np.int64))
        self.net = _build(self.cfg.arch, d, k, self.cfg.hidden, self.cfg.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        loss_fn = _loss_fn(self.cfg, y_idx, k)
        best, best_state, waited = -1.0, None, 0
        for epoch in range(self.cfg.epochs):
            t0 = time.perf_counter()
            self.net.train()
            order = torch.from_numpy(rng.permutation(len(yt)))
            total = 0.0
            for i in range(0, len(order), self.cfg.batch_size):
                idx = order[i:i + self.cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            score = evaluate(y_val, self.predict(x_val), w_val, names)["macro_f1"]
            self.cfg.history.append({"epoch": epoch + 1, "train_loss": total / len(yt),
                                     "val_macro_f1": score,
                                     "seconds": time.perf_counter() - t0})
            if score > best:
                best, waited = score, 0
                best_state = {n: v.detach().clone() for n, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray, batch: int = 8192) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            xs = torch.from_numpy(self._scale(x))
            for i in range(0, len(xs), batch):
                out.append(torch.softmax(self.net(xs[i:i + batch]), dim=1).numpy())
        return np.concatenate(out)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
