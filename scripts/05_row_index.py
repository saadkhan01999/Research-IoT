"""Step 5: the per-row index the strict split protocols need.

Usage: python scripts/05_row_index.py

Writes, for all 46,776,697 rows of the cache:
    data/processed/row_hash.npy     uint64 hash of the 39 feature values (same hash as Step 2)
    data/processed/first_copy.npy   True for the first row of every (feature vector, label) pair
Both are memory-mapped files, so this runs within a few hundred MB of RAM.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
from numpy.lib.format import open_memmap

from iotml.audit import REC, write_buckets
from iotml.config import load_config, project_path
from iotml.data import feature_columns
from iotml.logs import get_logger


def main() -> None:
    config = load_config()
    log = get_logger("05_row_index", project_path(config["paths"]["logs"]))
    cache = project_path(config["audit"]["cache"])
    work = project_path(config["audit"]["work_dir"])
    if work.exists():
        shutil.rmtree(work)
    n = write_buckets(cache, feature_columns(cache), work, log)
    out = cache.parent
    row_hash = open_memmap(out / "row_hash.npy", mode="w+", dtype=np.uint64, shape=(n,))
    first = open_memmap(out / "first_copy.npy", mode="w+", dtype=np.bool_, shape=(n,))
    first[:] = False
    filled = 0
    for path in sorted(work.glob("b*.bin")):
        r = np.fromfile(path, dtype=REC)
        if r.size == 0:
            continue
        row_hash[r["row"]] = r["h"]
        r = r[np.lexsort((r["row"], r["lab"], r["h"]))]
        change = np.ones(r.size, dtype=bool)
        change[1:] = (r["h"][1:] != r["h"][:-1]) | (r["lab"][1:] != r["lab"][:-1])
        first[r["row"][change]] = True
        filled += r.size
    row_hash.flush()
    first.flush()
    shutil.rmtree(work)
    if filled != n:
        raise SystemExit(f"Filled {filled} rows, expected {n}")
    summary = {"rows": int(n), "first_copies": int(first.sum()),
               "duplicate_copies": int(n - first.sum())}
    (out / "row_index_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Done: %s", summary)


if __name__ == "__main__":
    main()
