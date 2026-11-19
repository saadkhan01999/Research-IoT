"""Step 8: early detection along capture order (secondary "prediction" experiment).

Usage: python scripts/08_early_detection.py --profile quick|full [--models ...] [--seeds ...]

Every source file holds one capture of one class, so the label never changes inside a file:
forecasting the class of a later window from the true class of the current one is trivially
exact (the "repeat the last class" baseline scores 100%) and is not a meaningful test here.
We therefore measure early detection instead. Under the order_within_file protocol a model
trained on the earlier rows of every file labels the first n rows of each file's test block,
and the attack family (8-class scheme) is decided by majority vote of those n rows. The output
is block-level accuracy and macro family recall for n = 1 ... 50.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from iotml.audit import read_rows
from iotml.config import load_config, project_path
from iotml.data import LABEL_CODE
from iotml.experiment import Experiment, RunSpec
from iotml.features import Preprocessor
from iotml.logs import get_logger
from iotml.models import make_model
from iotml.splits import TEST
from iotml.system import keep_awake

PROTOCOL, SCHEME = "order_within_file", "L8"
HORIZONS = [1, 2, 3, 5, 10, 20, 50]


def first_test_rows(parts: np.ndarray, sources: pd.DataFrame, n: int) -> pd.DataFrame:
    """Global ids of the first n test rows of every file (fewer if the block is shorter)."""
    rows = []
    for fid, start, end in zip(sources["file_id"], sources["start"], sources["end"],
                               strict=True):
        test = np.flatnonzero(parts[start:end] == TEST)[:n] + start
        rows.append(pd.DataFrame({"file_id": fid, "row_id": test, "order": np.arange(test.size)}))
    return pd.concat(rows, ignore_index=True)


def vote_curve(blocks: pd.DataFrame, n_classes: int) -> list[dict[str, float]]:
    out = []
    for n in HORIZONS:
        first = blocks[blocks["order"] < n]
        decided = first.groupby("file_id").agg(
            truth=("truth", "first"),
            vote=("pred", lambda p: np.bincount(p, minlength=n_classes).argmax()),
            rows_used=("pred", "size"))
        correct = decided["vote"] == decided["truth"]
        per_family = correct.groupby(decided["truth"]).mean()
        out.append({"n_rows": n, "blocks": int(len(decided)),
                    "block_accuracy": float(correct.mean()),
                    "macro_family_recall": float(per_family.mean()),
                    "mean_rows_used": float(decided["rows_used"].mean())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--models", nargs="+", default=["xgboost", "random_forest"])
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = get_logger(f"08_early_{args.profile}", project_path("logs"))
    exp = Experiment(args.profile, log)
    scheme = exp.schemes[SCHEME]
    sources = pd.read_csv(exp.cache.parent / "source_files.csv")
    exp_cfg = cfg["experiment"]
    seeds = [int(s) for s in (args.seeds or exp_cfg.get("careful_seeds", exp_cfg["seeds"]))]
    rows = []
    t0 = time.time()
    with keep_awake(log):
        for seed in seeds:
            parts = exp.split(PROTOCOL, seed)
            blocks = first_test_rows(parts, sources, max(HORIZONS))
            frame = read_rows(exp.cache, np.sort(blocks["row_id"].to_numpy()),
                              exp.features + [LABEL_CODE], float32=True)
            frame = frame.set_index("row_id").loc[blocks["row_id"]].reset_index()
            train = exp.sample(PROTOCOL, seed, "train_capped")
            prep = Preprocessor(drop=list(exp.mcfg["features"]["drop"])).fit(
                train[exp.features])
            blocks["truth"] = scheme.encode(frame[LABEL_CODE].to_numpy())
            for name in args.models:
                spec = RunSpec(PROTOCOL, SCHEME, name, "capped", "tuned_from_random", seed,
                               args.profile)
                model = make_model(name, exp.params(spec, prep), seed, exp.n_jobs)
                model.fit(prep.transform(train[exp.features]),
                          scheme.encode(train[LABEL_CODE].to_numpy()))
                blocks["pred"] = model.predict(prep.transform(frame[exp.features])).astype(int)
                for r in vote_curve(blocks, scheme.n_classes):
                    rows.append({"seed": seed, "model": name, **r})
                log.info("early detection seed %d %s: n=1 %.3f, n=10 %.3f", seed, name,
                         rows[-len(HORIZONS)]["block_accuracy"], rows[-3]["block_accuracy"])
    out = project_path("results/early_detection")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"curve_{args.profile}.csv", index=False)
    table.groupby(["model", "n_rows"])[["block_accuracy", "macro_family_recall"]].agg(
        ["mean", "std"]).to_csv(out / f"curve_summary_{args.profile}.csv")
    log.info("all done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
