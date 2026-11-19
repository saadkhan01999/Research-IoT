"""Step 3: how sensitive is the merged label scheme (LM) to the merge threshold?

Usage: python scripts/03_merge_sensitivity.py

Writes results/labels/merge_sensitivity.json: the range of thresholds that give the same
groups as the chosen one, the weakest link needed inside the groups, and the strongest link
between labels that stay separate.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from iotml.config import CONFIG_DIR, load_config, project_path
from iotml.labels import link_strengths, merge_groups, threshold_range


def main() -> None:
    config = load_config()
    audit_dir = project_path(config["audit"]["out_dir"])
    rows = pd.read_csv(audit_dir / "class_counts_34.csv").set_index("label")["rows"].to_dict()
    pairs = pd.read_csv(audit_dir / "conflict_pairs.csv")
    chosen = float(yaml.safe_load((CONFIG_DIR / "labels.yaml").read_text(encoding="utf-8"))
                   ["merge_rule"]["min_shared_share"])
    groups = merge_groups(pairs, rows, chosen)
    member = {name: i for i, g in enumerate(groups) for name in g}
    links = link_strengths(pairs, rows)
    inside = links[[member.get(a, -1) == member.get(b, -2) for a, b in zip(
        links["a"], links["b"], strict=True)]]
    outside = links.drop(inside.index)
    lo, hi = threshold_range(pairs, rows, chosen)
    result = {
        "chosen_threshold": chosen,
        "groups": groups,
        "same_groups_for_thresholds": [lo, hi],
        "links_inside_groups": inside.round(4).to_dict("records"),
        "strongest_links_between_separate_labels": outside.head(5).round(4).to_dict("records"),
    }
    out = project_path(config["paths"]["results"]) / "labels" / "merge_sensitivity.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
