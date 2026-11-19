"""Tests for link strengths and the threshold range of the merge rule."""

from __future__ import annotations

import pandas as pd

from iotml.labels import link_strengths, threshold_range

ROWS = {"A": 100, "B": 50, "C": 1000}
PAIRS = pd.DataFrame({"label": ["A", "B", "C", "B"],
                      "shares_vectors_with": ["B", "A", "B", "C"],
                      "rows": [30, 20, 60, 2]})


def test_link_strength_takes_the_larger_direction() -> None:
    links = link_strengths(PAIRS, ROWS).set_index(["a", "b"])["share"]
    assert links[("A", "B")] == 0.4          # max(30/100, 20/50)
    assert links[("B", "C")] == 0.06         # max(60/1000, 2/50)


def test_threshold_range_is_bounded_by_links() -> None:
    lo, hi = threshold_range(PAIRS, ROWS, chosen=0.2, step=0.01)
    assert lo == 0.07 and hi == 0.4          # above B-C (0.06), up to A-B (0.4)
