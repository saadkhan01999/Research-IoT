"""Step 5: label-conflict ceiling of the test part of every split protocol.

Usage: python scripts/05_split_ceilings.py --profile quick|full [--seeds 11 23 37]

The ceilings of Step 3 (results/labels/ceilings.json) are computed on all rows of the release.
The strict protocols test on other rows: the deduplicated split keeps one copy of each
(vector, label) pair, and the file-grouped and ordered splits drop test rows whose vector also
occurs in training. A model can only be compared with the ceiling of the rows it is tested on,
so this script computes, for the test part of each protocol and seed and for each label scheme:

    accuracy ceiling     share of test rows that carry the most common class of their vector
                         (among the test rows); no classifier that sees only the features can
                         be right on more of these rows
    recall bounds        per class, and their mean over the classes present

It reads the split saved by scripts/05_strict.py (data/interim/splits/<protocol>_s<seed>.npy)
and never builds a split itself; protocols whose split does not exist yet are listed and
skipped, so the script can be run again when more runs are done.

Writes results/splits/test_ceilings.csv (one row per protocol, seed and scheme) and
results/splits/test_ceilings_per_class.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from iotml.audit import ceilings_of_rows
from iotml.config import load_config, project_path
from iotml.data import LABEL_CODE
from iotml.labels import load_schemes
from iotml.logs import get_logger
from iotml.splits import TEST

MAIN_PROTOCOLS = ["random", "random_dedup", "file_grouped", "order_within_file"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = get_logger(f"05_split_ceilings_{args.profile}", project_path(cfg["paths"]["logs"]))
    seeds = args.seeds or list(cfg["experiment"].get("careful_seeds", cfg["experiment"]["seeds"]))
    cache = project_path(cfg["audit"]["cache"])
    interim = project_path(cfg["paths"]["interim"]) / "splits"
    out_dir = project_path(cfg["paths"]["results"]) / "splits"
    suffix = "" if args.profile == "full" else f"_{args.profile}"

    schemes = load_schemes(cache.parent, project_path("results/labels/merge_groups.json"))
    maps = {name: s.code_map for name, s in schemes.items()}
    row_hash = np.load(cache.parent / "row_hash.npy", mmap_mode="r")
    labels34 = pq.read_table(cache, columns=[LABEL_CODE]).column(0).to_numpy()

    summary, per_class, missing = [], [], []
    for protocol in MAIN_PROTOCOLS:
        for seed in seeds:
            path = interim / f"{protocol}_s{seed}.npy"
            if not path.exists():
                missing.append(f"{protocol} s{seed}")
                continue
            ids = np.flatnonzero(np.load(path) == TEST)
            res = ceilings_of_rows(np.asarray(row_hash[ids]), labels34[ids], maps)
            for name, r in res.items():
                summary.append({"protocol": protocol, "seed": seed, "scheme": name,
                                "test_rows": r["rows"],
                                "distinct_vectors": r["distinct_vectors"],
                                "accuracy_ceiling": r["accuracy_ceiling"],
                                "macro_recall_upper_bound": r["macro_recall_upper_bound"]})
                for i, cls in enumerate(schemes[name].classes):
                    per_class.append({"protocol": protocol, "seed": seed, "scheme": name,
                                      "class": cls, "test_rows": int(r["per_class_rows"][i]),
                                      "recall_upper_bound": float(r["recall_upper_bound"][i])})
            log.info("%s s%d: %d test rows, %d distinct vectors, ceiling L8 %.4f, L34 %.4f",
                     protocol, seed, res["L8"]["rows"], res["L8"]["distinct_vectors"],
                     res["L8"]["accuracy_ceiling"], res["L34"]["accuracy_ceiling"])
    if not summary:
        log.error("no saved split found in %s", interim)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_dir / f"test_ceilings{suffix}.csv", index=False)
    pd.DataFrame(per_class).to_csv(out_dir / f"test_ceilings_per_class{suffix}.csv",
                                   index=False)
    if missing:
        log.warning("split not built yet, skipped: %s", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
