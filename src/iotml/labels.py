"""Label schemes: the 34 original labels (L34), 8 categories (L8), binary (L2), and L34 with
the flood labels merged that the features cannot tell apart (LM).

Every scheme is a lookup table from the L34 code stored in the cache to the scheme's class
index, so any scheme can be applied to any sample without reading the data again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelScheme:
    """Classes of one scheme and the map from L34 codes to them."""

    name: str
    classes: list[str]
    code_map: np.ndarray  # code_map[l34_code] = class index in this scheme

    def encode(self, l34_codes: np.ndarray) -> np.ndarray:
        return self.code_map[np.asarray(l34_codes, dtype=np.int64)]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def merge_groups(pairs: pd.DataFrame, label_rows: dict[str, int],
                 min_share: float) -> list[list[str]]:
    """Connected groups of labels linked by heavy vector sharing.

    `pairs` has columns label, shares_vectors_with, rows: the rows of `label` whose feature
    vector also occurs under `shares_vectors_with`. Two labels are linked when that count is
    at least `min_share` of the rows of `label` (either direction).
    """
    parent = {name: name for name in label_rows}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in pairs.itertuples(index=False):
        if row.rows >= min_share * label_rows[row.label]:
            parent[root(row.label)] = root(row.shares_vectors_with)
    groups: dict[str, list[str]] = {}
    for name in label_rows:
        groups.setdefault(root(name), []).append(name)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def link_strengths(pairs: pd.DataFrame, label_rows: dict[str, int]) -> pd.DataFrame:
    """Strength of each label pair = the larger of the two shared-row shares."""
    df = pairs.assign(share=pairs["rows"] / pairs["label"].map(label_rows))
    df["a"] = df[["label", "shares_vectors_with"]].min(axis=1)
    df["b"] = df[["label", "shares_vectors_with"]].max(axis=1)
    return (df.groupby(["a", "b"], as_index=False)["share"].max()
            .sort_values("share", ascending=False).reset_index(drop=True))


def threshold_range(pairs: pd.DataFrame, label_rows: dict[str, int], chosen: float,
                    step: float = 0.001) -> tuple[float, float]:
    """Smallest and largest threshold (on a grid) that give the same groups as `chosen`."""
    target = merge_groups(pairs, label_rows, chosen)
    grid = np.round(np.arange(step, 1.0, step), 6)
    same = [t for t in grid if merge_groups(pairs, label_rows, float(t)) == target]
    # keep the contiguous run that contains `chosen`
    lo = hi = chosen
    for t in sorted(same):
        if t <= chosen and all(x in same for x in grid[(grid >= t) & (grid <= chosen)]):
            lo = min(lo, float(t))
        if t >= chosen and all(x in same for x in grid[(grid >= chosen) & (grid <= t)]):
            hi = max(hi, float(t))
    return lo, hi


def merged_name(group: list[str]) -> str:
    """Readable name: the attack types without the DDoS-/DoS- prefix, e.g. 'TCP_Flood (merged)'."""
    types = sorted({g.split("-", 1)[1] if g.startswith(("DDoS-", "DoS-")) else g for g in group})
    return "/".join(types) + " (merged)"


def build_schemes(labels: list[str], category_of: dict[str, str], categories: list[str],
                  merges: list[list[str]]) -> dict[str, LabelScheme]:
    """All label schemes, given the L34 label order used in the cache."""
    l8 = np.array([categories.index(category_of[name]) for name in labels])
    l2 = np.array([0 if category_of[name] == "Benign" else 1 for name in labels])
    merged_of = {name: merged_name(g) for g in merges for name in g}
    lm_names = sorted({merged_of.get(name, name) for name in labels})
    lm = np.array([lm_names.index(merged_of.get(name, name)) for name in labels])
    return {
        "L34": LabelScheme("L34", list(labels), np.arange(len(labels))),
        "L8": LabelScheme("L8", list(categories), l8),
        "L2": LabelScheme("L2", ["Benign", "Attack"], l2),
        "LM": LabelScheme("LM", lm_names, lm),
    }


def load_schemes(processed_dir: Path, merges_path: Path) -> dict[str, LabelScheme]:
    """Schemes for the cache in `processed_dir`, with the merge groups saved by Step 3."""
    codes = json.loads((processed_dir / "label_codes.json").read_text(encoding="utf-8"))
    merges = json.loads(merges_path.read_text(encoding="utf-8"))["groups"]
    return build_schemes(codes["labels"], codes["category_of"], codes["categories"], merges)
