"""Tests for windows of consecutive rows: they never cross a file or a part boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iotml.splits import TEST, TRAIN, UNUSED
from iotml.windows import gather, valid_window_ends, window_rows


def test_windows_stay_inside_one_file_and_one_part() -> None:
    sources = pd.DataFrame({"start": [0, 8], "end": [8, 12]})
    #          file 0: T T T T U X X X     file 1: T T T T
    parts = np.array([TRAIN] * 4 + [UNUSED] + [TEST] * 3 + [TRAIN] * 4, dtype=np.int8)
    ends = valid_window_ends(parts, sources, k=3)
    assert ends.tolist() == [2, 3, 7, 10, 11]
    rows = window_rows(ends, 3)
    assert rows.tolist()[0] == [0, 1, 2] and rows.tolist()[2] == [5, 6, 7]
    file_of = np.repeat([0, 1], [8, 4])
    for w in rows:                                   # one file and one part per window
        assert len(set(file_of[w])) == 1 and len(set(parts[w])) == 1


def test_gather_builds_the_window_tensor() -> None:
    frame = pd.DataFrame({"row_id": [3, 4, 5, 9], "a": [1.0, 2.0, 3.0, 9.0],
                          "b": [10.0, 20.0, 30.0, 90.0]})
    x = gather(frame, np.array([[3, 4, 5]]), ["a", "b"])
    assert x.shape == (1, 3, 2)
    assert x[0, :, 0].tolist() == [1.0, 2.0, 3.0]
