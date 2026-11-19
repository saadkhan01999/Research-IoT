"""Tests for label schemes and the merge rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from iotml.audit import scheme_ceilings, write_buckets
from iotml.labels import build_schemes, merge_groups, merged_name
from tests.test_audit import _cache, _Log


def test_merge_rule_links_only_heavy_sharing() -> None:
    rows = {"DDoS-TCP_Flood": 100, "DoS-TCP_Flood": 50, "MITM": 1000, "Benign": 5000}
    pairs = pd.DataFrame({
        "label": ["DDoS-TCP_Flood", "DoS-TCP_Flood", "MITM", "Benign"],
        "shares_vectors_with": ["DoS-TCP_Flood", "DDoS-TCP_Flood", "Benign", "MITM"],
        "rows": [60, 40, 50, 30]})            # MITM 5%, Benign 0.6%: below 10%
    assert merge_groups(pairs, rows, 0.10) == [["DDoS-TCP_Flood", "DoS-TCP_Flood"]]


def test_merged_name_drops_prefixes() -> None:
    assert merged_name(["DDoS-TCP_Flood", "DoS-TCP_Flood"]) == "TCP_Flood (merged)"
    assert merged_name(["DDoS-SYN_Flood", "DDoS-SynonymousIP_Flood", "DoS-SYN_Flood"]) == \
        "SYN_Flood/SynonymousIP_Flood (merged)"


def test_schemes_map_every_label() -> None:
    labels = ["BenignTraffic", "DDoS-TCP_Flood", "DoS-TCP_Flood", "XSS"]
    cat = {"BenignTraffic": "Benign", "DDoS-TCP_Flood": "DDoS", "DoS-TCP_Flood": "DoS",
           "XSS": "Web"}
    s = build_schemes(labels, cat, ["Benign", "DDoS", "DoS", "Web"],
                      [["DDoS-TCP_Flood", "DoS-TCP_Flood"]])
    codes = np.array([0, 1, 2, 3])
    assert s["L34"].encode(codes).tolist() == [0, 1, 2, 3]
    assert s["L8"].encode(codes).tolist() == [0, 1, 2, 3]
    assert s["L2"].encode(codes).tolist() == [0, 1, 1, 1]
    assert s["LM"].classes == ["BenignTraffic", "TCP_Flood (merged)", "XSS"]
    assert s["LM"].encode(codes).tolist() == [0, 1, 1, 2]


def test_scheme_ceiling_rises_when_conflicting_labels_merge(tmp_path: Path) -> None:
    cache = _cache(tmp_path)            # labels 0 and 1 share one vector (see test_audit)
    write_buckets(cache, ["f1", "f2"], tmp_path / "w", _Log())
    maps = {"orig": np.array([0, 1, 2, 3]), "merged": np.array([0, 0, 1, 2])}
    res = scheme_ceilings(tmp_path / "w", maps)
    assert res["orig"]["best"] == 5 and res["merged"]["best"] == 6
    assert res["orig"]["recall_upper_bound"].tolist() == [1.0, 0.0, 1.0, 1.0]
