"""Step 2: audit the CICIoT2023 cache.

Usage: python scripts/02_audit.py [--part schema|counts|profile|duplicates|order|leakage|all]

Every part writes CSV/JSON files to results/audit/ and adds its key numbers to
results/audit/audit_summary.json, which the paper's number trace will point to.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from iotml.audit import (
    analyse_buckets,
    column_profile,
    column_redundancy,
    order_structure,
    read_rows,
    stratified_row_ids,
    write_buckets,
)
from iotml.config import CONFIG_DIR, load_config, project_path
from iotml.data import CATEGORY_CODE, FILE_COL, LABEL_CODE, feature_columns
from iotml.logs import get_logger


class Audit:
    def __init__(self, config: dict, log) -> None:
        self.cfg = config["audit"]
        self.seed = int(config["project"]["seed"])
        self.n_jobs = int(config["runtime"]["n_jobs"])
        self.log = log
        self.cache = project_path(self.cfg["cache"])
        self.out = project_path(self.cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        processed = self.cache.parent
        self.codes = json.loads((processed / "label_codes.json").read_text(encoding="utf-8"))
        self.sources = pd.read_csv(processed / "source_files.csv")
        self.features = feature_columns(self.cache)
        self.summary_path = self.out / "audit_summary.json"
        self.summary = (json.loads(self.summary_path.read_text(encoding="utf-8"))
                        if self.summary_path.exists() else {})

    def save(self, part: str, values: dict) -> None:
        self.summary[part] = {**values, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.summary_path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")

    # -------------------------------------------------------------- schema
    def schema(self) -> None:
        spec = yaml.safe_load((CONFIG_DIR / "paper_features.yaml").read_text(encoding="utf-8"))
        table = pd.DataFrame(spec["paper_table4"])
        ours = set(self.features)
        mapped = set(table["ours"].dropna())
        extra = sorted(ours - mapped)
        if extra != sorted(spec["only_in_ours"]):
            raise SystemExit(f"Columns not described in paper_features.yaml: {extra}")
        table["present_in_ours"] = table["ours"].notna()
        table.to_csv(self.out / "schema_vs_paper.csv", index=False)
        self.save("schema", {
            "our_features": len(self.features), "paper_features_excl_timestamp": len(table) - 1,
            "shared": int(table["present_in_ours"].sum()),
            "paper_only": [r for r in table.loc[~table["present_in_ours"], "paper"] if r != "ts"],
            "ours_only": extra, "timestamp_column_present": False,
            "renamed": {"Duration (paper, = TTL)": "Time_To_Live"}})

    # -------------------------------------------------------------- counts
    def counts(self) -> None:
        t = pq.read_table(self.cache, columns=[LABEL_CODE, CATEGORY_CODE])
        lab = t.column(0).to_numpy()
        n = lab.size
        labels, cats = self.codes["labels"], self.codes["categories"]
        lab_counts = np.bincount(lab, minlength=len(labels))
        files = self.sources.groupby("label")["file_id"].count()
        by_label = pd.DataFrame({"label": labels, "category": [self.codes["category_of"][x]
                                                               for x in labels],
                                 "rows": lab_counts,
                                 "files": [int(files.get(x, 0)) for x in labels]})
        by_label["share"] = by_label["rows"] / n
        by_label.sort_values("rows", ascending=False).to_csv(self.out / "class_counts_34.csv",
                                                             index=False)
        cat_counts = by_label.groupby("category")["rows"].sum().reindex(cats)
        pd.DataFrame({"category": cats, "rows": cat_counts.to_numpy(),
                      "share": cat_counts.to_numpy() / n}).to_csv(
            self.out / "class_counts_8.csv", index=False)
        benign = int(by_label.loc[by_label["category"] == "Benign", "rows"].sum())
        self.save("counts", {
            "rows": int(n), "labels": len(labels), "categories": len(cats),
            "source_files": len(self.sources), "benign_rows": benign,
            "attack_rows": int(n - benign),
            "largest_label": by_label.loc[by_label["rows"].idxmax(), "label"],
            "largest_label_rows": int(by_label["rows"].max()),
            "smallest_label": by_label.loc[by_label["rows"].idxmin(), "label"],
            "smallest_label_rows": int(by_label["rows"].min()),
            "imbalance_ratio": float(by_label["rows"].max() / by_label["rows"].min()),
            "labels_with_one_file": int((by_label["files"] == 1).sum())})

    # -------------------------------------------------------------- profile
    def profile(self) -> None:
        prof = column_profile(self.cache, self.features)
        prof.to_csv(self.out / "column_profile.csv", index=False)
        self.save("profile", {
            "columns_with_missing": prof.loc[prof["null_or_nan"] > 0, "feature"].tolist(),
            "missing_cells": int(prof["null_or_nan"].sum()),
            "columns_with_inf": prof.loc[(prof["pos_inf"] + prof["neg_inf"]) > 0,
                                         "feature"].tolist(),
            "inf_cells": int((prof["pos_inf"] + prof["neg_inf"]).sum()),
            "constant_columns": prof.loc[prof["constant"], "feature"].tolist(),
            "columns_with_at_most_10_values": prof.loc[prof["distinct_values"] <= 10,
                                                       "feature"].tolist()})

    # -------------------------------------------------------------- redundancy
    def redundancy(self) -> None:
        res = column_redundancy(self.cache, self.features)
        res.pop("correlation").to_csv(self.out / "feature_correlation.csv")
        self.save("redundancy", res)

    # -------------------------------------------------------------- duplicates
    def duplicates(self) -> None:
        work = project_path(self.cfg["work_dir"])
        if work.exists():
            shutil.rmtree(work)
        self.log.info("Hashing rows into buckets ...")
        n = write_buckets(self.cache, self.features, work, self.log)
        self.log.info("Analysing buckets ...")
        benign = self.codes["categories"].index("Benign")
        res = analyse_buckets(work, len(self.codes["labels"]), benign)
        shutil.rmtree(work)
        tot, per, pairs = res["totals"], res["per_label"], res["pairs"]
        labels = self.codes["labels"]
        by_label = pd.DataFrame({"label": labels, **{k: v for k, v in per.items()}})
        by_label["dup_same_label_share"] = by_label["dup_same_label"] / by_label["rows"]
        by_label["conflict_share"] = by_label["conflict_rows"] / by_label["rows"]
        by_label["recall_upper_bound"] = by_label["recall_upper_rows"] / by_label["rows"]
        by_label.sort_values("conflict_share", ascending=False).to_csv(
            self.out / "duplicates_by_label.csv", index=False)
        idx = np.argwhere(pairs > 0)
        pair_df = pd.DataFrame({"label": [labels[i] for i, _ in idx],
                                "shares_vectors_with": [labels[j] for _, j in idx],
                                "rows": [int(pairs[i, j]) for i, j in idx]})
        pair_df.sort_values("rows", ascending=False).to_csv(self.out / "conflict_pairs.csv",
                                                            index=False)
        self.save("duplicates", {
            **{k: int(v) for k, v in tot.items()},
            "hashed_rows": n,
            "duplicate_rows": tot["rows"] - tot["unique_vectors"],
            "duplicate_share": (tot["rows"] - tot["unique_vectors"]) / tot["rows"],
            "accuracy_ceiling_34": tot["best34"] / tot["rows"],
            "accuracy_ceiling_8": tot["best8"] / tot["rows"],
            "accuracy_ceiling_2": tot["best2"] / tot["rows"],
            "macro_recall_upper_bound_34": float(by_label["recall_upper_bound"].mean()),
            "hash": "pandas hash_pandas_object over the 39 float64 features (64-bit)"})

    # -------------------------------------------------------------- order
    def order(self) -> None:
        prof = pd.read_csv(self.out / "column_profile.csv").set_index("feature")
        center = prof.loc[self.features, "mean"].to_numpy()
        scale = prof.loc[self.features, "std"].replace(0, 1).fillna(1).to_numpy()
        res = order_structure(self.cache, self.features, center, scale, self.seed)
        res = res.merge(self.sources[["file_id", "path", "label"]], on="file_id", how="left")
        res.to_csv(self.out / "order_by_file.csv", index=False)
        valid = res.dropna(subset=["ratio_lag_to_random"])
        self.save("order", {
            "files": len(res),
            "median_ratio_lag_to_random": float(valid["ratio_lag_to_random"].median()),
            "files_ratio_below_0_9": int((valid["ratio_lag_to_random"] < 0.9).sum()),
            "files_ratio_below_0_5": int((valid["ratio_lag_to_random"] < 0.5).sum()),
            "median_next_row_identical": float(valid["next_row_identical"].median()),
            "median_random_pair_identical": float(valid["random_pair_identical"].median())})

    # -------------------------------------------------------------- leakage
    def leakage(self) -> None:
        ids = stratified_row_ids(self.cache, int(self.cfg["leak_cap_per_label"]), self.seed)
        data = read_rows(self.cache, ids, self.features + [LABEL_CODE, CATEGORY_CODE, FILE_COL])
        # Trees work in float32; put missing/infinite cells just outside the observed range
        # (the largest finite value in the data is about 1.4e8).
        x_all = np.nan_to_num(data[self.features].to_numpy(np.float64), nan=-1.0,
                              posinf=1e30, neginf=-1e30)
        benign = self.codes["categories"].index("Benign")
        tasks = {"34-class": data[LABEL_CODE].to_numpy(), "8-class": data[CATEGORY_CODE].to_numpy(),
                 "binary": (data[CATEGORY_CODE].to_numpy() != benign).astype(int)}
        folds = StratifiedKFold(int(self.cfg["leak_folds"]), shuffle=True, random_state=self.seed)

        def score(task: str, cols: list[int]) -> dict:
            y = tasks[task]
            acc, f1 = [], []
            for tr, te in folds.split(x_all, y):
                tree = DecisionTreeClassifier(max_depth=int(self.cfg["leak_tree_max_depth"]),
                                              min_samples_leaf=int(self.cfg["leak_tree_min_leaf"]),
                                              random_state=self.seed)
                tree.fit(x_all[tr][:, cols], y[tr])
                pred = tree.predict(x_all[te][:, cols])
                acc.append(accuracy_score(y[te], pred))
                f1.append(f1_score(y[te], pred, average="macro"))
            name = "ALL FEATURES" if len(cols) > 1 else self.features[cols[0]]
            return {"task": task, "feature": name, "accuracy": float(np.mean(acc)),
                    "macro_f1": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1))}

        jobs = [(t, [j]) for t in tasks for j in range(len(self.features))]
        jobs += [(t, list(range(len(self.features)))) for t in tasks]
        results = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(score)(t, c) for t, c in jobs)
        table = pd.DataFrame(results)
        for task, y in tasks.items():
            majority = np.bincount(y).max() / y.size
            table.loc[table["task"] == task, "majority_baseline_accuracy"] = majority
        table.sort_values(["task", "macro_f1"], ascending=[True, False]).to_csv(
            self.out / "one_column_leakage.csv", index=False)
        top = {t: table[(table["task"] == t) & (table["feature"] != "ALL FEATURES")]
               .nlargest(3, "macro_f1")[["feature", "macro_f1"]].to_dict("records")
               for t in tasks}
        allf = {t: float(table[(table["task"] == t) & (table["feature"] == "ALL FEATURES")]
                         ["macro_f1"].iloc[0]) for t in tasks}
        self.save("leakage", {"sample_rows": len(data), "cap_per_label":
                              int(self.cfg["leak_cap_per_label"]), "top_single_features": top,
                              "all_features_tree_macro_f1": allf,
                              "note": "decision tree on one feature, stratified 5-fold, "
                                      "random split within the capped sample"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parts = ["schema", "counts", "profile", "redundancy", "duplicates", "order", "leakage"]
    parser.add_argument("--part", default="all", choices=[*parts, "all"])
    args = parser.parse_args()
    config = load_config()
    log = get_logger("02_audit", project_path(config["paths"]["logs"]))
    audit = Audit(config, log)
    (audit.out / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for part in parts if args.part == "all" else [args.part]:
        t0 = time.time()
        log.info("== %s", part)
        getattr(audit, part)()
        log.info("== %s done in %.1f min", part, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
