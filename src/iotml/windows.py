"""Windows of k consecutive rows from one capture file, all inside the same part of a split.

The cache keeps rows in file order and, inside a file, in capture order (Step 2), so a
window is a block of global row ids e-k+1..e. A window is valid only when every row in it
belongs to the same file and the same (used) part, so no window straddles train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iotml.splits import UNUSED


def valid_window_ends(parts: np.ndarray, sources: pd.DataFrame, k: int) -> np.ndarray:
    """Sorted global row ids that end a valid window of length k."""
    ends = []
    for start, end in zip(sources["start"].to_numpy(), sources["end"].to_numpy(), strict=True):
        p = parts[start:end]
        if p.size < k:
            continue
        brk = np.ones(p.size, dtype=bool)                 # True where a new run starts
        brk[1:] = p[1:] != p[:-1]
        brk |= p == UNUSED
        run_start = np.maximum.accumulate(np.where(brk, np.arange(p.size), 0))
        run_len = np.arange(p.size) - run_start + 1
        ok = (run_len >= k) & (p != UNUSED)
        ends.append(np.flatnonzero(ok) + start)
    return np.concatenate(ends) if ends else np.array([], dtype=np.int64)


def window_rows(ends: np.ndarray, k: int) -> np.ndarray:
    """(windows, k) matrix of the global row ids of every window, oldest row first."""
    return ends[:, None] - np.arange(k - 1, -1, -1)[None, :]


def gather(frame: pd.DataFrame, rows: np.ndarray, columns: list[str]) -> np.ndarray:
    """Feature tensor (windows, k, features) from a frame that has a sorted `row_id` column
    covering every row id in `rows`."""
    ids = frame["row_id"].to_numpy()
    pos = np.searchsorted(ids, rows.ravel())
    if not np.array_equal(ids[pos], rows.ravel()):
        raise ValueError("frame does not contain every row of the windows")
    values = frame[columns].to_numpy(dtype=np.float32)
    return values[pos].reshape(rows.shape[0], rows.shape[1], len(columns))
