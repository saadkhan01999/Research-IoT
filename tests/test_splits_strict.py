"""Tests for the strict split protocols of Step 5: each checks the property the protocol
exists to guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iotml.splits import (
    TEST,
    TRAIN,
    UNUSED,
    VAL,
    file_grouped_split,
    hash_split,
    order_split,
    purge,
    unseen_type,
)

SHARES = {"train": 0.7, "val": 0.1, "test": 0.2}


def test_hash_split_keeps_identical_vectors_together() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.integers(0, 2**63, 5000, dtype=np.uint64)
    row_hash = np.repeat(vectors, 3)                      # every vector three times
    parts = hash_split(row_hash, SHARES, seed=11)
    assert np.all(parts.reshape(-1, 3) == parts.reshape(-1, 3)[:, :1])
    share = np.bincount(parts, minlength=3) / parts.size
    assert np.allclose(share, [0.7, 0.1, 0.2], atol=0.02)
    assert not np.array_equal(parts, hash_split(row_hash, SHARES, seed=12))


def test_order_split_trains_on_earlier_rows_with_a_gap() -> None:
    file_ids = np.repeat([0, 1], [100, 50])
    positions = np.concatenate([np.arange(100), np.arange(50)])
    parts = order_split(file_ids, positions, np.array([100, 50]), SHARES, gap=2)
    f0 = parts[:100]
    assert f0[:70].tolist() == [TRAIN] * 70
    assert f0[70:72].tolist() == [UNUSED, UNUSED]         # gap after the train block
    assert f0[72:80].tolist() == [VAL] * 8
    assert f0[80:82].tolist() == [UNUSED, UNUSED]
    assert f0[82:].tolist() == [TEST] * 18
    used = parts[parts != UNUSED]
    trains = np.flatnonzero(parts == TRAIN)
    tests = np.flatnonzero(parts == TEST)
    assert used.size > 0 and positions[trains].max() < 100
    assert all(positions[t] > 69 for t in tests if file_ids[t] == 0)


def test_file_grouped_split_never_cuts_a_file_with_enough_files() -> None:
    sources = pd.DataFrame({"file_id": range(7),
                            "label": ["A"] * 5 + ["B"] * 2,
                            "rows": [100, 120, 80, 90, 110, 60, 40]})
    file_ids = np.repeat(sources["file_id"], sources["rows"]).to_numpy()
    positions = np.concatenate([np.arange(n) for n in sources["rows"]])
    parts, used = file_grouped_split(file_ids, positions, sources, SHARES,
                                     np.random.default_rng(1), min_files=3, gap=1)
    for fid in range(5):                                   # label A: whole files
        assert len(set(parts[file_ids == fid].tolist())) == 1
    a_parts = {int(parts[file_ids == f][0]) for f in range(5)}
    assert a_parts == {TRAIN, VAL, TEST}                   # every part gets a file
    assert (used.loc[used["label"] == "B", "method"] == "block").all()
    b0 = parts[file_ids == 5]                              # label B: blocks inside the file
    assert b0[0] == TRAIN and b0[-1] == TEST


def test_gap_never_exceeds_two_percent_of_a_file() -> None:
    from iotml.splits import file_gap

    assert file_gap(1_000_000, 100) == 100
    assert file_gap(1252, 100) == 25
    assert file_gap(40, 100) == 0


def test_purge_removes_test_rows_seen_in_training() -> None:
    row_hash = np.array([1, 2, 3, 1, 4, 2, 5], dtype=np.uint64)
    parts = np.array([TRAIN, TRAIN, TRAIN, TEST, TEST, VAL, VAL], dtype=np.int8)
    out, removed = purge(parts, row_hash)
    assert out.tolist() == [TRAIN, TRAIN, TRAIN, UNUSED, TEST, UNUSED, VAL]
    assert removed == {"val": 1, "test": 1}


def test_unseen_type_removes_the_label_only_from_train_and_val() -> None:
    labels = np.array([0, 1, 1, 1, 2])
    parts = np.array([TRAIN, TRAIN, VAL, TEST, TEST], dtype=np.int8)
    out = unseen_type(parts, labels, held_out=1)
    assert out.tolist() == [TRAIN, UNUSED, UNUSED, TEST, TEST]
