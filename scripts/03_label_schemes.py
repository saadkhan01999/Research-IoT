"""Step 3: fix the label schemes and compute the label-conflict ceiling of each one.

Usage: python scripts/03_label_schemes.py

Inputs: the Step 2 audit (results/audit/conflict_pairs.csv, class_counts_34.csv,
audit_summary.json) and the cache. Outputs:
    results/labels/merge_groups.json   flood labels merged by the rule in config/labels.yaml
    results/labels/ceilings.json       ceiling per scheme (L34, L8, L2, LM, and L7 analysis-only)
    tables/label_scheme.csv            one row per original label: category, merged class,
                                       rows, files, per-class recall upper bounds
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd
import yaml

from iotml.audit import scheme_ceilings, write_buckets
from iotml.config import CONFIG_DIR, load_config, project_path
from iotml.data import feature_columns
from iotml.labels import build_schemes, merge_groups
from iotml.logs import get_logger


def main() -> None:
    config = load_config()
    log = get_logger("03_label_schemes", project_path(config["paths"]["logs"]))
    audit_dir = project_path(config["audit"]["out_dir"])
    cache = project_path(config["audit"]["cache"])
    out_dir = project_path(config["paths"]["results"]) / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = json.loads((cache.parent / "label_codes.json").read_text(encoding="utf-8"))
    rule = yaml.safe_load((CONFIG_DIR / "labels.yaml").read_text(encoding="utf-8"))["merge_rule"]

    counts = pd.read_csv(audit_dir / "class_counts_34.csv").set_index("label")
    pairs = pd.read_csv(audit_dir / "conflict_pairs.csv")
    groups = merge_groups(pairs, counts["rows"].to_dict(), float(rule["min_shared_share"]))
    (out_dir / "merge_groups.json").write_text(json.dumps({
        "groups": groups, "rule": rule,
        "computed_from": "results/audit/conflict_pairs.csv (Step 2, all 46,776,697 rows)",
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    log.info("Merge groups: %s", groups)

    schemes = build_schemes(codes["labels"], codes["category_of"], codes["categories"], groups)
    l7_classes = ["DDoS+DoS" if c in ("DDoS", "DoS") else c for c in codes["categories"]]
    l7_names = sorted(set(l7_classes))
    maps = {name: s.code_map for name, s in schemes.items()}
    maps["L7 (analysis only: DDoS and DoS as one family)"] = np.array(
        [l7_names.index(l7_classes[i]) for i in schemes["L8"].code_map])

    work = project_path(config["audit"]["work_dir"])
    if work.exists():
        shutil.rmtree(work)
    write_buckets(cache, feature_columns(cache), work, log)
    ceilings = scheme_ceilings(work, maps)
    shutil.rmtree(work)

    audit_l34 = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    expected = audit_l34["duplicates"]["best34"]
    if ceilings["L34"]["best"] != expected:
        raise SystemExit(f"L34 ceiling {ceilings['L34']['best']} differs from the audit "
                         f"({expected}); stop and investigate")

    summary = {}
    for name, res in ceilings.items():
        classes = schemes[name].classes if name in schemes else l7_names
        summary[name] = {
            "classes": len(classes), "rows": res["rows"],
            "accuracy_ceiling": res["accuracy_ceiling"],
            "macro_recall_upper_bound": res["macro_recall_upper_bound"],
            "recall_upper_bound": dict(zip(classes, map(float, res["recall_upper_bound"]),
                                           strict=True))}
    summary["check"] = "L34 ceiling equals the Step 2 audit value"
    (out_dir / "ceilings.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lm = schemes["LM"]
    table = pd.DataFrame({
        "label": codes["labels"],
        "category": [codes["category_of"][x] for x in codes["labels"]],
        "merged_class": [lm.classes[i] for i in lm.code_map],
    }).set_index("label")
    table["rows"] = counts["rows"]
    table["files"] = counts["files"]
    table["recall_upper_bound_L34"] = [summary["L34"]["recall_upper_bound"][x]
                                       for x in table.index]
    table["recall_upper_bound_LM"] = [summary["LM"]["recall_upper_bound"][lm.classes[i]]
                                      for i in lm.code_map]
    tables_dir = project_path(config["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.reset_index().sort_values(["category", "label"]).to_csv(
        tables_dir / "label_scheme.csv", index=False)
    for name, s in summary.items():
        if isinstance(s, dict):
            log.info("%s: %d classes, accuracy ceiling %.4f, macro recall bound %.4f", name,
                     s["classes"], s["accuracy_ceiling"], s["macro_recall_upper_bound"])


if __name__ == "__main__":
    main()
