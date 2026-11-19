"""Tests for the duplicate / label-conflict analysis on a tiny hand-made dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from iotml.audit import analyse_buckets, ceilings_of_rows, write_buckets


class _Log:
    def info(self, *args: object) -> None:  # silent logger for tests
        pass


def _cache(tmp: Path) -> Path:
    # vectors: A appears 3x (labels 0,0,1), B 2x (label 2, two files), C 1x (label 3)
    f1 = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    f2 = [5.0, 5.0, 5.0, 6.0, 6.0, 7.0]
    lab = [0, 0, 1, 2, 2, 3]
    cat = [0, 0, 0, 1, 1, 1]          # category 1 = attack, 0 = benign
    fid = [0, 0, 1, 2, 3, 4]
    table = pa.table({"f1": f1, "f2": f2, "label": pa.array(lab, pa.int8()),
                      "category": pa.array(cat, pa.int8()), "file_id": pa.array(fid, pa.int16()),
                      "row_in_file": pa.array([0] * 6, pa.int32())})
    path = tmp / "cache.parquet"
    pq.write_table(table, path)
    return path


def test_duplicates_conflicts_and_ceilings(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    n = write_buckets(cache, ["f1", "f2"], tmp_path / "w", _Log())
    assert n == 6
    res = analyse_buckets(tmp_path / "w", n_labels=4, benign_cat=0)
    tot, per, pairs = res["totals"], res["per_label"], res["pairs"]
    assert tot["rows"] == 6 and tot["unique_vectors"] == 3
    assert tot["vectors_multi_label"] == 1 and tot["rows_multi_label"] == 3
    # majority label per vector: A->0 (2 rows), B->2 (2), C->3 (1): 5 of 6 rows recoverable
    assert tot["best34"] == 5
    assert tot["best8"] == 6 and tot["best2"] == 6       # coarse labels do not conflict
    assert per["dup_same_label"].tolist() == [1, 0, 1, 0]
    assert per["conflict_rows"].tolist() == [2, 1, 0, 0]
    assert per["other_file_same_label"].tolist() == [0, 0, 2, 0]
    assert per["recall_upper_rows"].tolist() == [2, 0, 2, 1]
    assert pairs[0, 1] == 2 and pairs[1, 0] == 1 and pairs.sum() == 3


def test_hash_is_stable_for_equal_rows() -> None:
    from iotml.audit import hash_rows

    frame = pd.DataFrame({"a": [1.0, 1.0, np.nan, np.nan], "b": [2.0, 2.0, 3.0, 3.0]})
    h = hash_rows(frame, ["a", "b"])
    assert h[0] == h[1] and h[2] == h[3] and h[0] != h[2]


def test_ceilings_of_rows_for_a_subset() -> None:
    # vectors: A 3x (labels 0,0,1), B 2x (label 2), C 1x (label 3), D 2x (labels 0 and 1, a tie)
    h = np.array([10, 10, 10, 20, 20, 30, 40, 40], dtype=np.uint64)
    lab = np.array([0, 0, 1, 2, 2, 3, 0, 1], dtype=np.int8)
    fine = np.arange(4)
    merged = np.array([0, 0, 1, 2])              # labels 0 and 1 become one class
    res = ceilings_of_rows(h, lab, {"fine": fine, "merged": merged})
    # fine: A->0 (2 right), B (2), C (1), D tie (1): 6 of 8 rows
    assert res["fine"]["rows"] == 8 and res["fine"]["distinct_vectors"] == 4
    assert np.isclose(res["fine"]["accuracy_ceiling"], 6 / 8)
    # label 0: 2 rows on A plus 1 on D (tie counts for both); label 1: only its D row of 2
    assert np.allclose(res["fine"]["recall_upper_bound"], [3 / 3, 1 / 2, 1.0, 1.0])
    assert np.isclose(res["merged"]["accuracy_ceiling"], 1.0)
    # the order of the rows does not matter, and a subset has its own ceiling
    perm = np.array([7, 2, 5, 0, 3, 6, 1, 4])
    again = ceilings_of_rows(h[perm], lab[perm], {"fine": fine})
    assert np.isclose(again["fine"]["accuracy_ceiling"], 6 / 8)
    sub = ceilings_of_rows(h[[0, 1, 3, 5]], lab[[0, 1, 3, 5]], {"fine": fine})
    assert np.isclose(sub["fine"]["accuracy_ceiling"], 1.0)
    assert np.isclose(sub["fine"]["macro_recall_upper_bound"], 1.0)   # label 1 has no rows
