"""Evaluation metrics with test-row weights (natural class mix, see iotml.sampling)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and false-positive rate of every class from a confusion matrix
    (rows = true class, columns = predicted class). Undefined values are 0."""
    tp = np.diag(cm)
    true_n = cm.sum(axis=1)
    pred_n = cm.sum(axis=0)
    total = cm.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_n > 0, tp / pred_n, 0.0)
        recall = np.where(true_n > 0, tp / true_n, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        fp = pred_n - tp
        negatives = total - true_n
        fpr = np.where(negatives > 0, fp / negatives, 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "support": true_n}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None,
             class_names: list[str]) -> dict[str, Any]:
    """All headline and per-class metrics. Macro averages run over classes present in the
    test sample. `weights` make the numbers estimate the natural class mix."""
    k = len(class_names)
    labels = np.arange(k)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    cm_w = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=w)
    cm_n = confusion_matrix(y_true, y_pred, labels=labels)
    pc = per_class(cm_w)
    present = cm_n.sum(axis=1) > 0
    accuracy = float(np.trace(cm_w) / cm_w.sum())
    result = {
        "accuracy": accuracy,
        "macro_precision": float(pc["precision"][present].mean()),
        "macro_recall": float(pc["recall"][present].mean()),
        "macro_f1": float(pc["f1"][present].mean()),
        "balanced_accuracy": float(pc["recall"][present].mean()),
        "weighted_f1": float(np.average(pc["f1"][present], weights=pc["support"][present])),
        "mcc": float(matthews_corrcoef(y_true, y_pred, sample_weight=w)),
        "macro_fpr": float(pc["fpr"][present].mean()),
        "unweighted_macro_f1": float(per_class(cm_n)["f1"][present].mean()),
        "unweighted_accuracy": float(np.trace(cm_n) / max(cm_n.sum(), 1)),
        "test_rows": int(len(y_true)),
        "classes_in_test": int(present.sum()),
        "per_class": {name: {m: float(pc[m][i]) for m in ("precision", "recall", "f1", "fpr")}
                      | {"test_rows": int(cm_n[i].sum())}
                      for i, name in enumerate(class_names) if present[i]},
        "confusion_weighted": cm_w.round(3).tolist(),
        "confusion_counts": cm_n.tolist(),
    }
    return result
