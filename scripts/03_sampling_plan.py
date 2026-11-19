"""Step 3: the sampling plan table for the quick and full profiles.

Usage: python scripts/03_sampling_plan.py

For every label it shows the rows available (as published and after removing within-label
duplicates), the planned train/val/test sample sizes per profile (assuming a 70/10/20 split),
and whether the cap or the label size limits them. Writes tables/sampling_plan.csv and
results/labels/sampling_plan_summary.json (totals and a memory estimate).
"""

from __future__ import annotations

import json

import pandas as pd

from iotml.config import load_config, project_path

SHARES = {"train": 0.7, "val": 0.1, "test": 0.2}
BYTES_PER_VALUE = 4      # float32 in the models
MODEL_FEATURES = 37      # 39 columns minus the two exact copies (D-015)


def main() -> None:
    base = load_config()
    audit_dir = project_path(base["audit"]["out_dir"])
    dup = pd.read_csv(audit_dir / "duplicates_by_label.csv").set_index("label")
    counts = pd.read_csv(audit_dir / "class_counts_34.csv").set_index("label")
    table = counts[["category", "rows", "files"]].copy()
    table["rows_dedup"] = table["rows"] - dup.loc[table.index, "dup_same_label"]

    summary = {}
    for profile in ("quick", "full"):
        exp = load_config(profile)["experiment"]
        for part, share in SHARES.items():
            cap = exp["sampling"][part]["cap_per_label"]
            for basis in ("rows", "rows_dedup"):
                available = (table[basis] * share).round().astype(int)
                col = f"{profile}_{part}{'_dedup' if basis == 'rows_dedup' else ''}"
                table[col] = available.clip(upper=cap)
            table[f"{profile}_{part}_limited_by"] = [
                "cap" if a > cap else "label size"
                for a in (table["rows"] * share).round().astype(int)]
        train_rows = int(table[f"{profile}_train"].sum())
        summary[profile] = {
            "seeds": exp["seeds"],
            "train_rows": train_rows,
            "val_rows": int(table[f"{profile}_val"].sum()),
            "test_rows": int(table[f"{profile}_test"].sum()),
            "train_rows_dedup": int(table[f"{profile}_train_dedup"].sum()),
            "labels_below_train_cap": int((table[f"{profile}_train_limited_by"]
                                           == "label size").sum()),
            "train_matrix_mb": round(train_rows * MODEL_FEATURES * BYTES_PER_VALUE / 1e6, 1),
            "natural_train_rows": exp["natural_train_rows"],
        }
    tables_dir = project_path(base["paths"]["tables"])
    table.sort_values(["category", "rows"], ascending=[True, False]).reset_index().to_csv(
        tables_dir / "sampling_plan.csv", index=False)
    out = project_path(base["paths"]["results"]) / "labels" / "sampling_plan_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
