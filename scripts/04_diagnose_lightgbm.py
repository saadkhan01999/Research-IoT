"""Step 4: why did default LightGBM collapse on the 34-label task? (training data only)

Usage: python scripts/04_diagnose_lightgbm.py --profile quick [--seed 42] [--threads N]

Fits LightGBM on the cached natural-mix training sample of the random split with (a) the
library defaults and (b) learning rate 0.02, recording the multi-class log loss on the
TRAINING rows after every boosting round. No validation or test rows are used. Writes
results/baseline/lightgbm_diagnosis_<profile>_s<seed>.json.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, record_evaluation

from iotml.config import load_config, project_path, read_yaml
from iotml.data import LABEL_CODE, feature_columns
from iotml.features import Preprocessor
from iotml.labels import load_schemes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None,
                        help="threads for LightGBM (default: runtime.n_jobs of the profile)")
    args = parser.parse_args()
    cfg = load_config(args.profile)
    mcfg = read_yaml(project_path("config") / "models.yaml")
    cache = project_path(cfg["audit"]["cache"])
    sample = project_path(cfg["paths"]["interim"]) / "samples" / \
        f"random_s{args.seed}_{args.profile}_train_natural_n{cfg['experiment']['natural_train_rows']}.parquet"
    train = pd.read_parquet(sample)
    features = feature_columns(cache)
    scheme = load_schemes(cache.parent, project_path("results/labels/merge_groups.json"))["L34"]
    x = Preprocessor(drop=list(mcfg["features"]["drop"])).fit(train[features]) \
        .transform(train[features])
    classes, y = np.unique(scheme.encode(train[LABEL_CODE].to_numpy()), return_inverse=True)
    counts = np.bincount(y)

    threads = args.threads or int(cfg["runtime"]["n_jobs"])
    report: dict[str, Any] = {"seed": args.seed, "sample": sample.name, "threads": threads,
                              "training_rows": int(len(y)), "classes": int(classes.size),
                              "smallest_class_rows": int(counts.min()), "runs": {}}
    trials: list[tuple[str, dict[str, Any]]] = [("defaults", {}),
                                               ("learning_rate_0.02", {"learning_rate": 0.02})]
    for name, params in trials:
        history: dict = {}
        model = LGBMClassifier(random_state=args.seed, n_jobs=threads, verbose=-1, **params)
        model.fit(x, y, eval_set=[(x, y)], eval_names=["train"],
                  callbacks=[record_evaluation(history)])
        loss = np.asarray(history["train"]["multi_logloss"], dtype=float)
        pred = np.asarray(model.predict(x))
        report["runs"][name] = {
            "params": params,
            "train_logloss_first": float(loss[0]), "train_logloss_min": float(np.nanmin(loss)),
            "train_logloss_last": float(loss[-1]),
            "round_of_min": int(np.nanargmin(loss)) + 1,
            "rounds_loss_increased": int((np.diff(loss) > 0).sum()),
            "nan_or_inf_rounds": int((~np.isfinite(loss)).sum()),
            "train_accuracy": float((pred == y).mean()),
            "train_logloss_every_10": [round(float(v), 4) for v in loss[::10]],
        }
    out = project_path("results/baseline") / f"lightgbm_diagnosis_{args.profile}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1))
    for name, r in report["runs"].items():
        print(name, {k: v for k, v in r.items() if k != "train_logloss_every_10"})


if __name__ == "__main__":
    main()
