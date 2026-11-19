"""Statistics: calibration error, grouped bootstrap intervals and rank-based model tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ calibration
def reliability(correct: np.ndarray, confidence: np.ndarray, weights: np.ndarray | None,
                bins: int = 15) -> tuple[float, pd.DataFrame]:
    """Expected calibration error (top-label, equal-width bins) and the reliability table."""
    w = np.ones(len(correct)) if weights is None else np.asarray(weights, dtype=float)
    conf = np.clip(np.asarray(confidence, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, ece, total = [], 0.0, w.sum()
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        acc = float(np.average(correct[m], weights=w[m]))
        avg_conf = float(np.average(conf[m], weights=w[m]))
        ece += wb / total * abs(acc - avg_conf)
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "weight_share": wb / total,
                     "accuracy": acc, "mean_confidence": avg_conf, "rows": int(m.sum())})
    return float(ece), pd.DataFrame(rows)


# ------------------------------------------------------------------ bootstrap
def confusion_by_group(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
                       groups: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k x k confusion matrix of every group (e.g. source file)."""
    names, g = np.unique(groups, return_inverse=True)
    flat = (g * k + y_true) * k + y_pred
    counts = np.bincount(flat, weights=weights, minlength=len(names) * k * k)
    return counts.reshape(len(names), k, k), names


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    true_n, pred_n = cm.sum(axis=1), cm.sum(axis=0)
    present = true_n > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred_n > 0, tp / pred_n, 0.0)
        r = np.where(true_n > 0, tp / true_n, 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(f1[present].mean())


def grouped_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
                      groups: np.ndarray, k: int, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> dict[str, float]:
    """Percentile interval of macro-F1 when whole groups (source files) are resampled with
    replacement within each class (D-040). The class of a group is the true class of its rows;
    in CICIoT2023 every file holds one label, so every group has one class. Resampling within
    classes keeps every class in every resample. A plain group resample drops a class whose
    only file is not drawn (about 37% of resamples for a one-file class); macro-F1 is then an
    average over fewer classes, and since the one-file classes are mostly hard ones, the
    interval moves upward and can miss the point estimate."""
    per_group, _ = confusion_by_group(y_true, y_pred, weights, groups, k)
    n_groups = per_group.shape[0]
    true_by_class = per_group.sum(axis=2)                 # (groups, classes): true rows
    stratum = true_by_class.argmax(axis=1)
    mixed = int(((true_by_class > 0).sum(axis=1) > 1).sum())
    strata = [np.flatnonzero(stratum == c) for c in np.unique(stratum)]
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        mult = np.zeros(n_groups, dtype=np.int64)
        for members in strata:
            np.add.at(mult, rng.choice(members, size=len(members), replace=True), 1)
        scores[b] = macro_f1_from_confusion(np.tensordot(mult, per_group, axes=1))
    point = macro_f1_from_confusion(per_group.sum(axis=0))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {"macro_f1": point, "ci_low": float(lo), "ci_high": float(hi),
            "groups": int(n_groups), "strata": len(strata),
            "single_group_strata": int(sum(len(m) == 1 for m in strata)),
            "mixed_groups": mixed, "n_boot": n_boot}


# ------------------------------------------------------------------ model comparison
def holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def compare_models(scores: pd.DataFrame) -> dict[str, Any]:
    """`scores`: one row per block (e.g. protocol x scheme), one column per model (higher is
    better). Friedman test, mean ranks, and Wilcoxon signed-rank tests of the best model
    against each other model with Holm correction."""
    scores = scores.dropna()
    ranks = scores.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()
    out: dict[str, Any] = {"blocks": int(len(scores)), "models": list(scores.columns),
                           "mean_ranks": mean_ranks.round(3).to_dict()}
    if len(scores) >= 3 and scores.shape[1] >= 3:
        stat, p = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
        out["friedman_chi2"], out["friedman_p"] = float(stat), float(p)
    best = mean_ranks.index[0]
    others = [c for c in scores.columns if c != best]
    raw = []
    for c in others:
        diff = scores[best] - scores[c]
        raw.append(1.0 if np.allclose(diff, 0) else
                   float(stats.wilcoxon(scores[best], scores[c], zero_method="zsplit").pvalue))
    out["best"] = best
    out["wilcoxon_vs_best"] = [{"model": c, "p_raw": pr, "p_holm": ph,
                                "mean_difference": float((scores[best] - scores[c]).mean())}
                               for c, pr, ph in zip(others, raw, holm(raw), strict=True)]
    return out
