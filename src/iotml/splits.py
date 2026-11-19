"""Split protocols: each returns one part code per row of the cache (0 train, 1 val, 2 test,
-1 unused).

    random_split        conventional row-level stratified split (Step 4 baseline)
    hash_split          identical feature vectors always fall in the same part
    file_grouped_split  whole source files per part; labels with too few files are cut into
                        contiguous blocks inside each file
    order_split         earlier rows of every file train, later rows test, with a gap
    purge               drop val/test rows whose exact vector also occurs in training
    unseen_type         one attack label removed from train and val (test keeps it)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRAIN, VAL, TEST, UNUSED = 0, 1, 2, -1
PART_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _bounds(shares: dict[str, float]) -> np.ndarray:
    bounds = np.cumsum([shares["train"], shares["val"], shares["test"]])
    if not np.isclose(bounds[-1], 1.0):
        raise ValueError(f"split shares must sum to 1, got {bounds[-1]}")
    return bounds


def random_split(strata: np.ndarray, shares: dict[str, float],
                 rng: np.random.Generator) -> np.ndarray:
    """Row-level split, stratified by `strata` (the L34 label): within every stratum a random
    `shares` fraction of rows goes to train, val and test. This is the conventional protocol
    of most published work."""
    strata = np.asarray(strata)
    parts = np.full(strata.size, UNUSED, dtype=np.int8)
    bounds = _bounds(shares)
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        idx = idx[rng.permutation(idx.size)]
        cut = np.round(bounds * idx.size).astype(int)
        parts[idx[: cut[0]]] = TRAIN
        parts[idx[cut[0]: cut[1]]] = VAL
        parts[idx[cut[1]: cut[2]]] = TEST
    return parts


def part_ids(parts: np.ndarray, part: int) -> np.ndarray:
    """Sorted global row ids of one part."""
    return np.flatnonzero(parts == part)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """A fixed bijective mixing of 64-bit integers (SplitMix64 finaliser)."""
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return (z ^ (z >> np.uint64(31))) & _MASK64


def hash_split(row_hash: np.ndarray, shares: dict[str, float], seed: int) -> np.ndarray:
    """Vector-grouped random split. A row's part depends only on its feature-vector hash and
    the seed, so every copy of a vector, whatever its label, lands in the same part."""
    bounds = _bounds(shares)
    salt = _splitmix64(np.array([seed], dtype=np.uint64))[0]
    u = _splitmix64(np.asarray(row_hash, dtype=np.uint64) ^ salt).astype(np.float64) / 2.0**64
    return np.searchsorted(bounds[:-1], u, side="right").astype(np.int8)


def file_gap(n: int, gap: int) -> int:
    """Gap for a file of n rows: `gap` rows, but never more than 2% of the file."""
    return min(gap, n // 50)


def _block_parts(n: int, bounds: np.ndarray, gap: int) -> np.ndarray:
    """Parts for the n rows of one file in order: train, then val, then test; the gap rows after
    each boundary are left unused so neighbouring windows do not straddle two parts."""
    gap = file_gap(n, gap)
    parts = np.full(n, UNUSED, dtype=np.int8)
    cuts = np.round(bounds * n).astype(int)
    starts = [0, cuts[0], cuts[1]]
    for part, (lo, hi) in enumerate(zip(starts, cuts, strict=True)):
        lo = lo + (gap if part > 0 else 0)
        if hi > lo:
            parts[lo:hi] = part
    return parts


def order_split(file_ids: np.ndarray, positions: np.ndarray, file_rows: np.ndarray,
                shares: dict[str, float], gap: int) -> np.ndarray:
    """Order-based split inside every source file (rows keep capture order, Step 2)."""
    bounds = _bounds(shares)
    parts = np.empty(file_ids.size, dtype=np.int8)
    for fid, n in enumerate(file_rows):
        rows = np.flatnonzero(file_ids == fid)
        parts[rows] = _block_parts(int(n), bounds, gap)[positions[rows]]
    return parts


def file_grouped_split(file_ids: np.ndarray, positions: np.ndarray, sources: pd.DataFrame,
                       shares: dict[str, float], rng: np.random.Generator, min_files: int,
                       gap: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Whole files per part for labels with at least `min_files` files; contiguous blocks
    inside each file for the other labels. `sources` has file_id, label, rows (one row per
    file, ids 0..n-1). Returns the parts and a table of how each file was used."""
    bounds = _bounds(shares)
    file_part = np.full(len(sources), UNUSED, dtype=np.int8)
    method = np.array(["file"] * len(sources), dtype=object)
    for _, group in sources.groupby("label"):
        if len(group) < min_files:
            method[group["file_id"].to_numpy()] = "block"
            continue
        order = group.iloc[rng.permutation(len(group))]
        total = order["rows"].sum()
        targets = {TEST: shares["test"] * total, VAL: shares["val"] * total}
        filled = {TEST: 0, VAL: 0}
        for i, row in enumerate(order.itertuples(index=False)):
            remaining = len(order) - i
            need = [p for p in (TEST, VAL) if filled[p] == 0 or filled[p] < targets[p]]
            if need and remaining > 1:      # always leave at least one file for training
                part = need[0]
                filled[part] += row.rows
            else:
                part = TRAIN
            file_part[row.file_id] = part
    parts = file_part[file_ids].astype(np.int8)
    for fid in np.flatnonzero(method == "block"):
        rows = np.flatnonzero(file_ids == fid)
        n = int(sources.loc[sources["file_id"] == fid, "rows"].iloc[0])
        parts[rows] = _block_parts(n, bounds, gap)[positions[rows]]
    used = sources[["file_id", "label", "rows"]].copy()
    used["method"] = method
    used["part"] = [PART_NAMES.get(int(p), "blocks") if m == "file" else "blocks"
                    for p, m in zip(file_part, method, strict=True)]
    return parts, used


def purge(parts: np.ndarray, row_hash: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Val/test rows whose exact feature vector also occurs in training become unused."""
    parts = parts.copy()
    train_hashes = np.unique(row_hash[parts == TRAIN])
    removed = {}
    for part in (VAL, TEST):
        rows = np.flatnonzero(parts == part)
        h = row_hash[rows]
        pos = np.searchsorted(train_hashes, h)
        pos[pos == train_hashes.size] = 0
        seen = train_hashes[pos] == h if train_hashes.size else np.zeros(h.size, bool)
        parts[rows[seen]] = UNUSED
        removed[PART_NAMES[part]] = int(seen.sum())
    return parts, removed


def unseen_type(parts: np.ndarray, labels34: np.ndarray, held_out: int) -> np.ndarray:
    """Remove one attack label from training and validation; its test rows stay in the test
    part so we can see which family the model gives a type it has never seen."""
    parts = parts.copy()
    parts[(labels34 == held_out) & ((parts == TRAIN) | (parts == VAL))] = UNUSED
    return parts
