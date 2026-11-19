"""Data audit of the CICIoT2023 cache: column profile, duplicates and label conflicts,
row-order structure, and the one-column leakage test.

All passes stream the cache, so the audit runs in well under 2 GB of RAM. Duplicate analysis
hashes each feature vector to 64 bits and processes one hash bucket at a time.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from iotml.data import CATEGORY_CODE, FILE_COL, LABEL_CODE, POS_COL, iter_batches

N_BUCKETS = 256
REC = np.dtype([("h", "<u8"), ("lab", "i1"), ("cat", "i1"), ("fid", "<i2"), ("row", "<u4")])


# ------------------------------------------------------------------ column profile
def column_profile(parquet: Path, features: list[str]) -> pd.DataFrame:
    """Exact per-column facts, one column in memory at a time."""
    rows = []
    for name in features:
        col = pq.read_table(parquet, columns=[name]).column(0)
        nulls = col.null_count
        a = col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        nan = np.isnan(a)
        pos_inf, neg_inf = np.isposinf(a), np.isneginf(a)
        finite = a[~(nan | pos_inf | neg_inf)]
        finite.sort()
        distinct = int(finite.size and 1 + np.count_nonzero(finite[1:] != finite[:-1]))
        rows.append({
            "feature": name, "rows": a.size, "null_or_nan": int(nulls or nan.sum()),
            "pos_inf": int(pos_inf.sum()), "neg_inf": int(neg_inf.sum()),
            "min": float(finite[0]) if finite.size else np.nan,
            "max": float(finite[-1]) if finite.size else np.nan,
            "mean": float(finite.mean()) if finite.size else np.nan,
            "std": float(finite.std()) if finite.size else np.nan,
            "distinct_values": distinct,
            "share_zero": float((finite == 0).mean()) if finite.size else np.nan,
            "constant": distinct <= 1,
        })
        del col, a, finite
    return pd.DataFrame(rows)


def column_redundancy(parquet: Path, features: list[str]) -> dict[str, Any]:
    """Exactly identical column pairs and Pearson correlations, streamed over the cache."""
    k = len(features)
    identical = np.ones((k, k), dtype=bool)
    n = 0
    s = np.zeros(k)
    ss = np.zeros((k, k))
    negatives = dict.fromkeys(features, 0)
    for frame in iter_batches(parquet, features, batch_rows=500_000):
        x = frame.to_numpy(dtype=np.float64)
        both_nan = np.isnan(x)[:, :, None] & np.isnan(x)[:, None, :]
        same = (x[:, :, None] == x[:, None, :]) | both_nan
        identical &= same.all(axis=0)
        del both_nan, same
        finite = np.where(np.isfinite(x), x, 0.0)  # non-finite cells count as 0 in the sums
        n += len(x)
        s += finite.sum(axis=0)
        ss += finite.T @ finite
        for j, name in enumerate(features):
            negatives[name] += int((x[:, j] < 0).sum())
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    pairs = [(features[i], features[j]) for i in range(k) for j in range(i + 1, k)]
    return {
        "identical_pairs": [list(p) for p, (i, j) in zip(pairs, _upper(k), strict=True)
                            if identical[i, j]],
        "correlated_pairs_abs_r_ge_0_99": [
            {"a": features[i], "b": features[j], "r": round(float(corr[i, j]), 5)}
            for i, j in _upper(k) if abs(corr[i, j]) >= 0.99 and not identical[i, j]],
        "negative_value_counts": {f: v for f, v in negatives.items() if v},
        "correlation": pd.DataFrame(corr, index=features, columns=features),
    }


def _upper(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


# ------------------------------------------------------------------ duplicates
def hash_rows(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """64-bit hash of each feature vector (identical values give identical hashes)."""
    return pd.util.hash_pandas_object(frame[features], index=False).to_numpy(dtype=np.uint64)


def write_buckets(parquet: Path, features: list[str], work: Path, log) -> int:
    """Pass 1: hash every row and append (hash, label, category, file, row) to its bucket."""
    work.mkdir(parents=True, exist_ok=True)
    handles = [open(work / f"b{b:03d}.bin", "wb") for b in range(N_BUCKETS)]  # noqa: SIM115
    total = 0
    try:
        cols = features + [LABEL_CODE, CATEGORY_CODE, FILE_COL]
        for frame in iter_batches(parquet, cols, batch_rows=500_000):
            rec = np.empty(len(frame), dtype=REC)
            rec["h"] = hash_rows(frame, features)
            rec["lab"] = frame[LABEL_CODE].to_numpy()
            rec["cat"] = frame[CATEGORY_CODE].to_numpy()
            rec["fid"] = frame[FILE_COL].to_numpy()
            rec["row"] = np.arange(total, total + len(frame), dtype=np.uint32)
            bucket = (rec["h"] & (N_BUCKETS - 1)).astype(np.int64)
            order = np.argsort(bucket, kind="stable")
            rec, bucket = rec[order], bucket[order]
            cuts = np.searchsorted(bucket, np.arange(N_BUCKETS + 1))
            for b in range(N_BUCKETS):
                if cuts[b + 1] > cuts[b]:
                    rec[cuts[b]:cuts[b + 1]].tofile(handles[b])
            total += len(frame)
            if total % 5_000_000 < 500_000:
                log.info("  hashed %d rows", total)
    finally:
        for fh in handles:
            fh.close()
    return total


def _runs(keys: Iterable[np.ndarray]) -> np.ndarray:
    """Start index of each run of equal consecutive key tuples (input sorted on the keys)."""
    keys = list(keys)
    change = np.zeros(len(keys[0]), dtype=bool)
    change[0] = True
    for k in keys:
        change[1:] |= k[1:] != k[:-1]
    return np.flatnonzero(change)


def _distinct_per_group(values: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of distinct `values` inside each group id."""
    order = np.lexsort((values, gid))
    v, gg = values[order], gid[order]
    start = _runs([gg, v])
    return np.bincount(gg[start], minlength=n_groups)


def analyse_buckets(work: Path, n_labels: int, benign_cat: int) -> dict[str, Any]:
    """Pass 2: duplicates, label conflicts, file spread and the accuracy ceilings."""
    tot = dict(rows=0, unique_vectors=0, rows_in_dup_groups=0, vectors_multi_label=0,
               rows_multi_label=0, vectors_multi_file=0, rows_multi_file=0,
               vectors_multi_category=0, rows_multi_category=0, vectors_multi_binary=0,
               rows_multi_binary=0, best34=0, best8=0, best2=0)
    per_label = {k: np.zeros(n_labels, dtype=np.int64) for k in (
        "rows", "dup_same_label", "conflict_rows", "other_file_same_label",
        "recall_upper_rows")}
    pair = np.zeros((n_labels, n_labels), dtype=np.int64)
    for path in sorted(work.glob("b*.bin")):
        r = np.fromfile(path, dtype=REC)
        if r.size == 0:
            continue
        r = r[np.lexsort((r["fid"], r["lab"], r["h"]))]
        h, lab, cat, fid = r["h"], r["lab"].astype(np.int64), r["cat"].astype(np.int64), r["fid"]
        binary = (cat != benign_cat).astype(np.int64)
        n = r.size
        tot["rows"] += n
        per_label["rows"] += np.bincount(lab, minlength=n_labels)

        g = _runs([h])                                  # vector groups
        gid = np.repeat(np.arange(g.size), np.diff(np.append(g, n)))
        size = np.diff(np.append(g, n))
        tot["unique_vectors"] += g.size
        tot["rows_in_dup_groups"] += int(size[size > 1].sum())

        n_lab = _distinct_per_group(lab, gid, g.size)
        n_cat = _distinct_per_group(cat, gid, g.size)
        n_bin = _distinct_per_group(binary, gid, g.size)
        n_fid = _distinct_per_group(fid.astype(np.int64), gid, g.size)
        for name, cnt in (("label", n_lab), ("file", n_fid), ("category", n_cat),
                          ("binary", n_bin)):
            multi = cnt > 1
            tot[f"vectors_multi_{name}"] += int(multi.sum())
            tot[f"rows_multi_{name}"] += int(size[multi].sum())

        # (vector, label) runs: rows of one label sharing one vector
        lr = _runs([h, lab])
        lr_size = np.diff(np.append(lr, n))
        lr_gid, lr_lab = gid[lr], lab[lr]
        per_label["dup_same_label"] += np.bincount(lr_lab, weights=lr_size - 1,
                                                   minlength=n_labels).astype(np.int64)
        conflict = n_lab[lr_gid] > 1
        per_label["conflict_rows"] += np.bincount(lr_lab[conflict], weights=lr_size[conflict],
                                                  minlength=n_labels).astype(np.int64)
        # same label, same vector, spread over more than one file
        lf = _runs([h, lab, fid.astype(np.int64)])
        lf_count = np.bincount(np.searchsorted(lr, lf, side="right") - 1, minlength=lr.size)
        spread = lf_count > 1
        per_label["other_file_same_label"] += np.bincount(
            lr_lab[spread], weights=lr_size[spread], minlength=n_labels).astype(np.int64)

        # ceilings: best possible accuracy = rows of the majority label in each vector group
        best_in_group = np.maximum.reduceat(lr_size, np.searchsorted(lr_gid, np.arange(g.size)))
        tot["best34"] += int(best_in_group.sum())
        is_best = lr_size == best_in_group[lr_gid]           # ties count for every tied label
        per_label["recall_upper_rows"] += np.bincount(lr_lab[is_best], weights=lr_size[is_best],
                                                      minlength=n_labels).astype(np.int64)
        for key, values in (("best8", cat), ("best2", binary)):
            order = np.lexsort((values, gid))          # group, then coarse label
            vv, gg = values[order], gid[order]
            start = _runs([gg, vv])
            run_size = np.diff(np.append(start, n))
            run_gid = gg[start]
            best = np.maximum.reduceat(run_size, np.searchsorted(run_gid, np.arange(g.size)))
            tot[key] += int(best.sum())

        # label pairs that share vectors (rows of label a whose vector also has label b)
        multi_groups = np.flatnonzero(n_lab > 1)
        if multi_groups.size:
            first = np.searchsorted(lr_gid, multi_groups)
            last = np.searchsorted(lr_gid, multi_groups, side="right")
            for s, e in zip(first, last, strict=True):
                labs, sizes = lr_lab[s:e], lr_size[s:e]
                for a, sa in zip(labs, sizes, strict=True):
                    pair[a, labs[labs != a]] += sa
    return {"totals": tot, "per_label": per_label, "pairs": pair}


def scheme_ceilings(work: Path, maps: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Label-conflict ceiling of each label scheme, from the hash buckets of write_buckets.

    For every distinct feature vector a classifier can output one class only, so the best
    possible accuracy is the share of rows that carry their vector's most common class. The
    per-class value counts a class as recoverable wherever it is (one of) the most common.
    """
    out: dict[str, dict[str, Any]] = {
        name: {"best": 0, "rows": 0,
               "per_class_rows": np.zeros(int(m.max()) + 1, dtype=np.int64),
               "per_class_best": np.zeros(int(m.max()) + 1, dtype=np.int64)}
        for name, m in maps.items()}
    for path in sorted(work.glob("b*.bin")):
        r = np.fromfile(path, dtype=REC)
        if r.size == 0:
            continue
        r = r[np.argsort(r["h"], kind="stable")]
        h, lab = r["h"], r["lab"].astype(np.int64)
        g = _runs([h])
        gid = np.repeat(np.arange(g.size), np.diff(np.append(g, r.size)))
        for name, m in maps.items():
            cls = m[lab]
            order = np.lexsort((cls, gid))
            cc, gg = cls[order], gid[order]
            start = _runs([gg, cc])
            size = np.diff(np.append(start, r.size))
            run_gid, run_cls = gg[start], cc[start]
            best = np.maximum.reduceat(size, np.searchsorted(run_gid, np.arange(g.size)))
            res = out[name]
            n_cls = res["per_class_rows"].size
            res["best"] += int(best.sum())
            res["rows"] += int(r.size)
            res["per_class_rows"] += np.bincount(cls, minlength=n_cls)
            top = size == best[run_gid]
            res["per_class_best"] += np.bincount(run_cls[top], weights=size[top],
                                                 minlength=n_cls).astype(np.int64)
    for res in out.values():
        res["accuracy_ceiling"] = res["best"] / res["rows"]
        res["recall_upper_bound"] = res["per_class_best"] / np.maximum(res["per_class_rows"], 1)
        res["macro_recall_upper_bound"] = float(res["recall_upper_bound"].mean())
    return out


def ceilings_of_rows(row_hash: np.ndarray, lab34: np.ndarray,
                     maps: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Label-conflict ceiling of each label scheme for one set of rows, in memory.

    Same definition as scheme_ceilings, for a subset such as the test part of a split: the
    accuracy of a classifier that gives every distinct vector of these rows its most common
    class among these rows. No classifier that sees only the features can do better on them.
    Classes without rows are left out of the mean recall bound.
    """
    order = np.argsort(row_hash, kind="stable")
    h = np.asarray(row_hash)[order]
    lab = np.asarray(lab34)[order].astype(np.int64)
    g = _runs([h])
    gid = np.repeat(np.arange(g.size, dtype=np.int32), np.diff(np.append(g, h.size)))
    out: dict[str, dict[str, Any]] = {}
    for name, m in maps.items():
        cls = m[lab].astype(np.int16)
        o = np.lexsort((cls, gid))
        cc, gg = cls[o], gid[o]
        start = _runs([gg, cc])
        size = np.diff(np.append(start, h.size))
        run_gid, run_cls = gg[start], cc[start]
        best = np.maximum.reduceat(size, np.searchsorted(run_gid, np.arange(g.size)))
        n_cls = int(m.max()) + 1
        rows = np.bincount(cls, minlength=n_cls)
        top = size == best[run_gid]
        per_best = np.bincount(run_cls[top], weights=size[top], minlength=n_cls)
        bound = per_best / np.maximum(rows, 1)
        out[name] = {"rows": int(h.size), "distinct_vectors": int(g.size),
                     "accuracy_ceiling": float(best.sum() / h.size),
                     "per_class_rows": rows, "recall_upper_bound": bound,
                     "macro_recall_upper_bound": float(bound[rows > 0].mean())}
    return out


# ------------------------------------------------------------------ row order
def order_structure(parquet: Path, features: list[str], center: np.ndarray, scale: np.ndarray,
                    seed: int) -> pd.DataFrame:
    """Per source file: are neighbouring rows more alike than random pairs of the same file?"""
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(parquet)
    cols = features + [FILE_COL, POS_COL]
    rows, carry = [], None
    for batch in pf.iter_batches(batch_size=400_000, columns=cols):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last_file = frame[FILE_COL].iloc[-1]
        done, carry = frame[frame[FILE_COL] != last_file], frame[frame[FILE_COL] == last_file]
        rows += [_order_one(f, features, center, scale, rng) for _, f in done.groupby(FILE_COL)]
    if carry is not None and len(carry):
        rows.append(_order_one(carry, features, center, scale, rng))
    return pd.DataFrame(rows)


def _order_one(frame: pd.DataFrame, features: list[str], center: np.ndarray, scale: np.ndarray,
               rng: np.random.Generator) -> dict[str, Any]:
    frame = frame.sort_values(POS_COL)
    x = frame[features].to_numpy(dtype=np.float64)
    x = np.nan_to_num((x - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(x)
    if n < 3:
        return {"file_id": int(frame[FILE_COL].iloc[0]), "rows": n}
    lag = np.abs(np.diff(x, axis=0)).mean()
    perm = x[rng.permutation(n)]
    rand = np.abs(np.diff(perm, axis=0)).mean()
    same_next = float(np.all(x[1:] == x[:-1], axis=1).mean())
    same_rand = float(np.all(perm[1:] == perm[:-1], axis=1).mean())
    return {"file_id": int(frame[FILE_COL].iloc[0]), "rows": n,
            "lag1_mean_abs_diff": float(lag), "random_pair_mean_abs_diff": float(rand),
            "ratio_lag_to_random": float(lag / rand) if rand > 0 else np.nan,
            "next_row_identical": same_next, "random_pair_identical": same_rand}


# ------------------------------------------------------------------ sampling for tests
def stratified_row_ids(parquet: Path, cap: int, seed: int) -> np.ndarray:
    """Global row ids: at most `cap` random rows from each label (all rows if fewer)."""
    labels = pq.read_table(parquet, columns=[LABEL_CODE]).column(0).to_numpy()
    rng = np.random.default_rng(seed)
    picked = []
    for code in np.unique(labels):
        ids = np.flatnonzero(labels == code)
        picked.append(ids if ids.size <= cap else rng.choice(ids, cap, replace=False))
    return np.sort(np.concatenate(picked))


def read_rows(parquet: Path, row_ids: np.ndarray, columns: list[str],
              float32: bool = False) -> pd.DataFrame:
    """Selected global rows, read by streaming (ids must be sorted). With `float32`, float
    columns are narrowed batch by batch to halve the memory."""
    parts, start = [], 0
    for frame in iter_batches(parquet, columns, batch_rows=250_000):
        stop = start + len(frame)
        lo, hi = np.searchsorted(row_ids, [start, stop])
        if hi > lo:
            piece = frame.iloc[row_ids[lo:hi] - start]
            if float32:
                floats = piece.select_dtypes("float64").columns
                piece = piece.astype(dict.fromkeys(floats, np.float32))
            parts.append(piece)
        start = stop
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "row_id", row_ids)
    return out
