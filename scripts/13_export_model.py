"""Step 13: export one trained model for the dashboard, in files that need no research code.

Usage: python scripts/13_export_model.py --profile full [--model auto|xgboost|...]
       [--scheme L8] [--protocol random_dedup] [--seed 11] [--sample-rows 5000]

Retrains the model exactly as in the experiments (capped training sample, settings tuned on the
random split) and writes models/dashboard/<scheme>_<model>/:
  model.json | model.txt | model.joblib   native XGBoost / LightGBM file, or scikit-learn joblib
  preprocess.json   input feature order, dropped copy columns, kept columns, training medians
  classes.json      model output index -> class name; 34-label name -> class (to score labelled
                    uploads); column aliases for the 46-feature release
  model_card.json   what the model is, how it was trained, and its measured results on every
                    split protocol (from results/experiments), with the label-conflict ceiling
  samples/*.csv     test rows for trying the dashboard (natural class mix, balanced, and an
                    ordered stream of capture blocks), with the true label
  samples/expected_*.csv   predictions of the research code for the same rows (the dashboard's
                    consistency test must reproduce them)
  manifest.json     SHA-256 of every file
With --model auto the model with the best mean macro-F1 under the protocol is chosen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from iotml.audit import hash_rows, read_rows
from iotml.config import load_config, project_path
from iotml.data import LABEL_CODE
from iotml.envinfo import library_versions
from iotml.experiment import Experiment, RunSpec
from iotml.features import Preprocessor
from iotml.io import sha256_file
from iotml.logs import get_logger
from iotml.models import TREE_MODELS, make_model
from iotml.results import load_runs
from iotml.sampling import seed_rng
from iotml.splits import TEST
from iotml.system import awake_seconds

STREAM_BLOCK_ROWS = 200   # consecutive rows per capture in the capture_stream sample
STREAM_BLOCKS = 25        # captures in the capture_stream sample

# The 46-feature release names the time-to-live column "Duration" (see results/audit).
ALIASES_46 = {"Duration": "Time_To_Live"}
DISPLAY = {"decision_tree": "Decision tree", "random_forest": "Random forest",
           "extra_trees": "Extra trees", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def choose_model(profile: str, scheme: str, protocol: str) -> str:
    runs = load_runs(profile)
    sub = runs[(runs["scheme"] == scheme) & (runs["protocol"] == protocol)
               & (runs["train_mode"] == "capped") & runs["model"].isin(TREE_MODELS)]
    if sub.empty:
        return "xgboost"
    return str(sub.groupby("model")["macro_f1"].mean().idxmax())


def measured_results(profile: str, scheme: str, model: str) -> list[dict[str, Any]]:
    runs = load_runs(profile)
    sub = runs[(runs["scheme"] == scheme) & (runs["model"] == model)
               & (runs["train_mode"] == "capped")
               & runs["params_mode"].isin(["tuned", "tuned_from_random"])
               & ~runs["protocol"].str.startswith("unseen_type__")]
    # Ceiling of the rows each protocol tests on (scripts/05_split_ceilings.py, D-032); the
    # release ceiling in the card does not apply to deduplicated or purged test parts.
    path = project_path("results/splits/test_ceilings"
                        + ("" if profile == "full" else f"_{profile}") + ".csv")
    ceilings: dict[str, float] = {}
    if path.exists():
        c = pd.read_csv(path)
        ceilings = c[c["scheme"] == scheme].groupby("protocol")["accuracy_ceiling"].mean() \
            .to_dict()
    out = []
    for protocol, g in sub.groupby("protocol"):
        out.append({"protocol": protocol, "seeds": int(len(g)),
                    "macro_f1_mean": float(g["macro_f1"].mean()),
                    "macro_f1_sd": float(g["macro_f1"].std()) if len(g) > 1 else None,
                    "accuracy_mean": float(g["accuracy"].mean()),
                    "accuracy_ceiling": ceilings.get(str(protocol)),
                    "balanced_accuracy_mean": float(g["balanced_accuracy"].mean())})
    return out


def frame_for_upload(rows: pd.DataFrame, features: list[str], label_names: list[str]
                     ) -> pd.DataFrame:
    out = rows[features].copy()
    out["label"] = [label_names[int(c)] for c in rows[LABEL_CODE]]
    return out


def ordered_stream(exp: Experiment, seed: int, rows_per_block: int, blocks: int,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Consecutive test rows of several captures (ordered protocol), one block per capture,
    to replay as a stream in which the class changes from block to block."""
    parts = exp.split("order_within_file", seed)
    sources = pd.read_csv(exp.cache.parent / "source_files.csv")
    picked = sources.iloc[rng.permutation(len(sources))]
    ids, meta = [], []
    for row in picked.itertuples(index=False):
        test = np.flatnonzero(parts[row.start:row.end] == TEST)[:rows_per_block]
        if test.size < rows_per_block:
            continue
        ids.append(test + row.start)
        meta += [(Path(row.path).name, int(i)) for i in test]
        if len(ids) == blocks:
            break
    flat = np.concatenate(ids)
    order = np.argsort(flat)
    frame = read_rows(exp.cache, flat[order], exp.features + [LABEL_CODE], float32=True)
    frame = frame.set_index("row_id").loc[flat].reset_index()
    frame["capture"] = [m[0] for m in meta]
    frame["row_in_capture"] = [m[1] for m in meta]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--scheme", default="L8")
    parser.add_argument("--protocol", default="random_dedup")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-rows", type=int, default=5000)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = get_logger("13_export_model", project_path("logs"))
    exp = Experiment(args.profile, log)
    exp_cfg = cfg["experiment"]
    seed = args.seed or int(exp_cfg.get("careful_seeds", exp_cfg["seeds"])[0])
    name = choose_model(args.profile, args.scheme, args.protocol) if args.model == "auto" \
        else args.model
    scheme = exp.schemes[args.scheme]
    l34 = exp.schemes["L34"].classes
    out = project_path(f"models/dashboard/{args.scheme}_{name}")
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log.info("exporting %s for %s (%s, seed %d)", name, args.scheme, args.protocol, seed)

    train = exp.sample(args.protocol, seed, "train_capped")
    test = exp.sample(args.protocol, seed, "test")
    drop = list(exp.mcfg["features"]["drop"])
    prep = Preprocessor(drop=drop).fit(train[exp.features])
    spec = RunSpec(args.protocol, args.scheme, name, "capped", "tuned_from_random", seed,
                   args.profile)
    params = exp.params(spec, prep)
    model = make_model(name, params, seed, exp.n_jobs)
    t0 = awake_seconds()  # stops while the laptop sleeps (decision D-026)
    model.fit(prep.transform(train[exp.features]), scheme.encode(train[LABEL_CODE].to_numpy()))
    fit_s = awake_seconds() - t0

    inner = model.model
    if name == "xgboost":
        model_file = "model.json"
        inner.save_model(out / model_file)
    elif name == "lightgbm":
        model_file = "model.txt"
        inner.booster_.save_model(str(out / model_file))
    else:
        model_file = "model.joblib"
        joblib.dump(inner, out / model_file, compress=3)

    (out / "preprocess.json").write_text(json.dumps({
        "features_in": exp.features, "drop": drop, "columns": prep.columns,
        "medians": [float(v) for v in prep.medians],
        "nonfinite": "missing and +/-inf values are replaced by the training median"},
        indent=1), encoding="utf-8")
    index_to_class = [scheme.classes[int(c)] for c in model.classes_]
    label_to_class = {lab: scheme.classes[int(scheme.code_map[i])] for i, lab in enumerate(l34)}
    label_to_class["Benign_Final"] = label_to_class["BenignTraffic"]
    (out / "classes.json").write_text(json.dumps({
        "scheme": args.scheme, "index_to_class": index_to_class,
        "classes": scheme.classes, "label_to_class": label_to_class,
        "column_aliases": ALIASES_46}, indent=1), encoding="utf-8")

    # ---- samples with the true label and the research code's predictions
    rng = seed_rng(seed, "dashboard", "samples")
    w = test["_weight"].to_numpy()
    natural = test.iloc[np.sort(rng.choice(len(test), args.sample_rows, replace=False,
                                           p=w / w.sum()))]
    y_test = scheme.encode(test[LABEL_CODE].to_numpy())
    per = max(1, args.sample_rows // scheme.n_classes)
    balanced_idx = np.concatenate([rng.permutation(np.flatnonzero(y_test == c))[:per]
                                   for c in range(scheme.n_classes)])
    balanced = test.iloc[rng.permutation(balanced_idx)]
    stream = ordered_stream(exp, seed, rows_per_block=STREAM_BLOCK_ROWS,
                            blocks=STREAM_BLOCKS, rng=rng)
    samples = {"natural_mix": natural, "balanced": balanced, "capture_stream": stream}
    sources = {
        "natural_mix": f"test part of the {args.protocol} split, natural class mix",
        "balanced": f"test part of the {args.protocol} split, equal rows per class",
        "capture_stream": "consecutive test rows of single captures under the ordered split "
                          "(one block per capture)",
    }
    # Share of each sample's rows whose exact feature vector is also in the training rows:
    # replays of these files are demonstrations, and this says how far they are from unseen.
    train_hash = np.unique(hash_rows(train[exp.features].astype(np.float32), exp.features))
    sample_info: dict[str, dict[str, Any]] = {}
    for sname, rows in samples.items():
        upload = frame_for_upload(rows, exp.features, l34)
        if "capture" in rows:
            upload["capture"] = rows["capture"].to_numpy()
            upload["row_in_capture"] = rows["row_in_capture"].to_numpy()
        upload.to_csv(out / "samples" / f"{sname}.csv", index=False)
        x = prep.transform(rows[exp.features])
        proba = model.predict_proba(x)
        pred = [index_to_class[i] for i in proba.argmax(axis=1)]
        pd.DataFrame({"predicted": pred, "confidence": proba.max(axis=1)}).to_csv(
            out / "samples" / f"expected_{sname}.csv", index=False)
        seen = np.isin(hash_rows(rows[exp.features].astype(np.float32), exp.features),
                       train_hash)
        sample_info[sname] = {"rows": int(len(upload)), "source": sources[sname],
                              "seen_in_training_share": round(float(seen.mean()), 4)}
        if sname == "capture_stream":
            sample_info[sname].update({"blocks": int(rows["capture"].nunique()),
                                       "rows_per_block": STREAM_BLOCK_ROWS})
        log.info("sample %s: %d rows, %.1f%% of vectors also in training", sname,
                 len(upload), 100 * seen.mean())

    ceilings = json.loads(project_path("results/labels/ceilings.json").read_text(
        encoding="utf-8"))
    card = {
        "name": f"{DISPLAY.get(name, name)} ({args.scheme}: {scheme.n_classes} classes)",
        "model": name,
        "scheme": args.scheme, "classes": scheme.classes, "model_file": model_file,
        "trained": {"dataset": "CICIoT2023, per-capture CSV release (39 features)",
                    "protocol": args.protocol, "seed": seed, "profile": args.profile,
                    "training_rows": int(len(train)),
                    "cap_per_label": int(exp_cfg["sampling"]["train"]["cap_per_label"]),
                    "params": {k: v for k, v in params.items()},
                    "fit_seconds": round(fit_s, 1),
                    "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "measured": measured_results(args.profile, args.scheme, name),
        "samples": sample_info,
        "ceiling": {"accuracy": ceilings[args.scheme]["accuracy_ceiling"],
                    "macro_recall": ceilings[args.scheme]["macro_recall_upper_bound"],
                    "note": "Identical feature vectors carry different labels in this release; "
                            "no model can exceed these values on the full release."},
        "limits": [
            "Trained on laboratory traffic from one testbed (105 devices). It has not been "
            "tested on other networks and should not be used as the only protection.",
            "DDoS and DoS floods of the same protocol often produce identical feature rows "
            "and are frequently confused.",
            "Expects the CICIoT2023 feature columns; rows from other feature extractors are "
            "not comparable.",
        ],
        "versions": library_versions(),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=1, default=str),
                                         encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    (out / "manifest.json").write_text(json.dumps(
        {p.relative_to(out).as_posix(): sha256_file(p) for p in files}, indent=1),
        encoding="utf-8")
    log.info("exported to %s in %.0f s", out, fit_s)


if __name__ == "__main__":
    main()
