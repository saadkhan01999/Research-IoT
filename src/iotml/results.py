"""Load experiment results (one JSON per run) and saved predictions into tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iotml.config import project_path

SPEC_KEYS = ["protocol", "scheme", "model", "train_mode", "params_mode", "seed", "profile"]
HEADLINE = ["macro_f1", "accuracy", "balanced_accuracy", "mcc", "macro_precision",
            "macro_recall", "macro_fpr", "weighted_f1",
            # on the capped test sample itself, without the natural-mix weights
            "unweighted_macro_f1", "unweighted_accuracy"]


def load_runs(profile: str = "full", root: Path | None = None) -> pd.DataFrame:
    """One row per run: spec fields, headline metrics, timing, size, path."""
    root = root or project_path("results/experiments")
    rows = []
    for path in root.glob(f"*/*/*/*_{profile}.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        m = r["metrics"]
        row: dict[str, Any] = {k: r["spec"][k] for k in SPEC_KEYS}
        row.update({k: m.get(k) for k in HEADLINE})
        row.update({"fit_seconds": r.get("fit_seconds"), "model_bytes": r.get("model_bytes"),
                    "predict_us_per_row_batch": r.get("predict_us_per_row_batch"),
                    "train_rows": r.get("train_rows"), "path": str(path)})
        if "unseen_type" in m:
            row["held_out"] = m["unseen_type"]["held_out"]
            row["held_out_recall_own_class"] = m["unseen_type"]["recall_own_class"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_per_class(profile: str = "full", root: Path | None = None) -> pd.DataFrame:
    """One row per (run, class) with precision, recall, F1, FPR and test rows."""
    root = root or project_path("results/experiments")
    rows = []
    for path in root.glob(f"*/*/*/*_{profile}.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        base = {k: r["spec"][k] for k in SPEC_KEYS}
        for cls, v in r["metrics"]["per_class"].items():
            rows.append({**base, "class": cls, **v})
    return pd.DataFrame(rows)


def load_confusion(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Weighted confusion matrix of one run and the class names present in the scheme."""
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    cm = np.asarray(r["metrics"]["confusion_weighted"], dtype=float)
    names = list(r["metrics"]["per_class"])
    return cm, names


def predictions_path(path: str | Path) -> Path:
    """Predictions file of a run, from the path of its JSON result."""
    p = Path(path)
    rel = p.relative_to(project_path("results/experiments"))
    return project_path("data/interim/predictions") / rel.with_suffix(".npz")


def family_mix(row_label: np.ndarray, y_pred: np.ndarray, n_families: int) -> pd.DataFrame:
    """Where the test rows of each fine label go. Rows: the fine label codes among the test
    rows; columns: family codes 0..n_families-1; values: the share of the label's rows that the
    model predicted as that family (each row sums to 1)."""
    t = pd.crosstab(np.asarray(row_label), np.asarray(y_pred), normalize="index")
    t = t.reindex(columns=range(n_families), fill_value=0.0)
    t.index.name, t.columns.name = "label", "family"
    return t


def seen_family_mix(profile: str, model: str, seeds: list[int]) -> pd.DataFrame:
    """Reference for the unseen-type test: where the test rows of each of the 34 labels go
    when the label IS in training. Read from the saved predictions of the deduplicated split
    (8 families, capped sample, settings tuned on the random split), the runs the unseen-type
    runs start from. Mean over the given seeds; index = label name, columns = family name."""
    import pyarrow.parquet as pq

    from iotml.data import LABEL_CODE
    from iotml.labels import load_schemes

    cache = project_path("data/processed/ciciot2023.parquet")
    schemes = load_schemes(cache.parent, project_path("results/labels/merge_groups.json"))
    family_of = np.asarray(schemes["L8"].code_map)
    families, names = list(schemes["L8"].classes), list(schemes["L34"].classes)
    labels = pq.read_table(cache, columns=[LABEL_CODE]).column(0).to_numpy()
    total: pd.DataFrame | None = None
    for seed in seeds:
        f = (project_path("data/interim/predictions/random_dedup/L8") / model
             / f"capped_tuned_from_random_s{seed}_{profile}.npz")
        if not f.exists():
            raise FileNotFoundError(str(f))
        d = np.load(f)
        row_label = labels[d["row_id"]]
        if not np.array_equal(family_of[row_label], d["y_true"]):
            raise ValueError(f"{f}: row numbers do not match the saved true families")
        mix = family_mix(row_label, d["y_pred"], len(families))
        total = mix if total is None else total.add(mix, fill_value=0.0)
    if total is None:
        raise FileNotFoundError("no seeds given")
    mean = total / len(seeds)
    mean.index = [names[int(i)] for i in mean.index]
    mean.columns = families
    return mean


def summarise(runs: pd.DataFrame, keys: list[str], metrics: list[str] | None = None
              ) -> pd.DataFrame:
    """Mean, standard deviation and number of seeds of each metric per configuration."""
    metrics = metrics or ["macro_f1", "accuracy", "balanced_accuracy", "mcc"]
    agg = runs.groupby(keys)[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
