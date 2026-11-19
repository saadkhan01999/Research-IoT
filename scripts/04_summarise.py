"""Step 4: summarise the random-split runs (mean and standard deviation over seeds).

Usage: python scripts/04_summarise.py --profile quick|full

Writes results/baseline/summary_<profile>.csv and per_class_<profile>.csv, each with the
label-conflict ceiling of the scheme next to the results.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from iotml.config import project_path

HEADLINE = ["macro_f1", "accuracy", "balanced_accuracy", "mcc", "macro_precision",
            "macro_fpr", "unweighted_macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--protocol", default="random")
    args = parser.parse_args()

    root = project_path("results/experiments") / args.protocol
    ceilings = json.loads(project_path("results/labels/ceilings.json").read_text(encoding="utf-8"))
    rows, per_class = [], []
    for path in sorted(root.glob(f"*/*/*_{args.profile}.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        spec, m = r["spec"], r["metrics"]
        rows.append({**{k: spec[k] for k in ("scheme", "model", "train_mode", "params_mode",
                                              "seed")},
                     **{k: m[k] for k in HEADLINE},
                     "fit_seconds": r["fit_seconds"], "model_mb": r["model_bytes"] / 1e6})
        for cls, v in m["per_class"].items():
            per_class.append({**{k: spec[k] for k in ("scheme", "model", "train_mode",
                                                       "params_mode", "seed")},
                              "class": cls, **v})
    if not rows:
        raise SystemExit(f"No results for profile {args.profile} in {root}")
    df = pd.DataFrame(rows)
    keys = ["scheme", "model", "train_mode", "params_mode"]
    summary = df.groupby(keys).agg(
        seeds=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in HEADLINE},
        **{f"{m}_sd": (m, "std") for m in ("macro_f1", "accuracy")},
        fit_seconds_mean=("fit_seconds", "mean"), model_mb_mean=("model_mb", "mean"),
    ).reset_index()
    summary["accuracy_ceiling"] = summary["scheme"].map(
        lambda s: ceilings[s]["accuracy_ceiling"])
    summary["accuracy_above_ceiling"] = summary["accuracy_mean"] > summary["accuracy_ceiling"]
    out_dir = project_path("results/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"summary_{args.profile}.csv", index=False)
    pc = pd.DataFrame(per_class).groupby(keys + ["class"]).agg(
        recall=("recall", "mean"), precision=("precision", "mean"), f1=("f1", "mean"),
        fpr=("fpr", "mean"), test_rows=("test_rows", "mean")).reset_index()
    pc.to_csv(out_dir / f"per_class_{args.profile}.csv", index=False)
    show = ["scheme", "model", "train_mode", "params_mode", "seeds", "macro_f1_mean",
            "accuracy_mean", "accuracy_ceiling", "fit_seconds_mean"]
    print(summary[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
