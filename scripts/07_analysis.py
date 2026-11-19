"""Step 7: statistics, calibration, error analysis, explanations and cost.

Usage: python scripts/07_analysis.py --part stats|calibration|errors|explain|cost|all

Reads the run results of Steps 4-6 (results/experiments) and their saved predictions
(data/interim/predictions); writes CSV/JSON files to results/analysis/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from iotml.config import project_path
from iotml.data import LABEL_CODE
from iotml.envinfo import cpu_name
from iotml.experiment import Experiment, RunSpec
from iotml.features import Preprocessor
from iotml.logs import get_logger
from iotml.models import TREE_MODELS, make_model
from iotml.results import load_runs, predictions_path, summarise
from iotml.sampling import cap_per_class, seed_rng
from iotml.stats import compare_models, grouped_bootstrap, reliability
from iotml.system import awake_seconds

COST_BATCH_ROWS = 10_000   # rows in the batch used for the per-row batch time
LATENCY_REPEATS = 300      # single-row predictions timed per model and thread count
BOOTSTRAP_RESAMPLES = 1000  # resamples of whole capture files for the macro-F1 interval
SHAP_ROWS_PER_CLASS = 250   # test rows per class explained with TreeSHAP

PROFILE = "full"   # overridden by --profile
MAIN_PROTOCOLS = ["random", "random_dedup", "file_grouped", "order_within_file"]
MAIN_SCHEMES = ["L8", "L34", "LM"]
OUT = project_path("results/analysis")


def main_runs() -> pd.DataFrame:
    """Capped training with tuned settings (Step 4 careful baseline and Step 5)."""
    runs = load_runs(PROFILE)
    keep = (runs["train_mode"] == "capped") & runs["params_mode"].isin(
        ["tuned", "tuned_from_random"]) & runs["protocol"].isin(MAIN_PROTOCOLS) \
        & runs["model"].isin(TREE_MODELS)
    return runs[keep]


def file_of(row_ids: np.ndarray, sources: pd.DataFrame) -> np.ndarray:
    return np.searchsorted(sources["end"].to_numpy(), row_ids, side="right")


# ------------------------------------------------------------------ stats
def part_stats(log: Any) -> None:
    runs = main_runs()
    table = summarise(runs, ["protocol", "scheme", "model"])
    table.to_csv(OUT / "main_results.csv", index=False)
    comparisons = {}
    for scheme in MAIN_SCHEMES + ["all"]:
        sub = runs if scheme == "all" else runs[runs["scheme"] == scheme]
        blocks = sub.groupby(["protocol", "scheme", "model"])["macro_f1"].mean().unstack("model")
        if len(blocks) >= 2:
            comparisons[scheme] = compare_models(blocks)
    (OUT / "model_comparison.json").write_text(json.dumps(comparisons, indent=1),
                                               encoding="utf-8")
    sources = pd.read_csv(project_path("data/processed/source_files.csv"))
    rows = []
    for _, r in runs[runs["seed"] == runs["seed"].min()].iterrows():
        p = predictions_path(r["path"])
        if not p.exists():
            continue
        z = np.load(p)
        k = int(max(z["y_true"].max(), z["y_pred"].max())) + 1
        res = grouped_bootstrap(z["y_true"].astype(int), z["y_pred"].astype(int), z["weight"],
                                file_of(z["row_id"].astype(np.int64), sources), k,
                                n_boot=BOOTSTRAP_RESAMPLES, seed=int(r["seed"]))
        rows.append({"protocol": r["protocol"], "scheme": r["scheme"], "model": r["model"],
                     "seed": r["seed"], **res})
    pd.DataFrame(rows).to_csv(OUT / "bootstrap_ci.csv", index=False)
    log.info("stats: %d configurations, %d bootstrap intervals", len(table), len(rows))


# ------------------------------------------------------------------ calibration
def part_calibration(log: Any) -> None:
    runs = main_runs()
    rows = []
    reliability_dir = OUT / "reliability"
    reliability_dir.mkdir(parents=True, exist_ok=True)
    for (protocol, scheme, model), group in runs.groupby(["protocol", "scheme", "model"]):
        parts = [np.load(predictions_path(p)) for p in group["path"]
                 if predictions_path(p).exists()]
        if not parts:
            continue
        correct = np.concatenate([(z["y_true"] == z["y_pred"]).astype(float) for z in parts])
        conf = np.concatenate([z["confidence"].astype(float) for z in parts])
        w = np.concatenate([z["weight"] for z in parts])
        ok = np.isfinite(conf)
        if not ok.any():
            continue
        ece, table = reliability(correct[ok], conf[ok], w[ok])
        table.to_csv(reliability_dir / f"{protocol}_{scheme}_{model}.csv", index=False)
        rows.append({"protocol": protocol, "scheme": scheme, "model": model, "ece": ece,
                     "seeds": len(parts),
                     "mean_confidence": float(np.average(conf[ok], weights=w[ok])),
                     "accuracy": float(np.average(correct[ok], weights=w[ok]))})
    pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
    log.info("calibration: %d configurations", len(rows))


# ------------------------------------------------------------------ errors
def part_errors(log: Any) -> None:
    runs = main_runs()
    pairs, matrices = [], {}
    for (protocol, scheme, model), group in runs.groupby(["protocol", "scheme", "model"]):
        cms = [np.asarray(json.loads(project_path(p).read_text(encoding="utf-8"))
                          ["metrics"]["confusion_weighted"], float) for p in group["path"]]
        cm = np.mean(cms, axis=0)                      # average over seeds
        classes = _scheme_classes(scheme)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)   # rows = true class
        matrices[(protocol, scheme, model)] = (norm, classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and norm[i, j] > 0:
                    pairs.append({"protocol": protocol, "scheme": scheme, "model": model,
                                  "true": classes[i], "predicted": classes[j],
                                  "share_of_true_class": float(norm[i, j])})
    pdf = pd.DataFrame(pairs).sort_values("share_of_true_class", ascending=False)
    pdf.to_csv(OUT / "confusion_pairs.csv", index=False)
    cm_dir = OUT / "confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for (protocol, scheme, model), (norm, classes) in matrices.items():
        pd.DataFrame(norm, index=classes, columns=classes).to_csv(
            cm_dir / f"{protocol}_{scheme}_{model}.csv")
    log.info("errors: %d confusion pairs", len(pdf))


def _scheme_classes(scheme: str) -> list[str]:
    from iotml.labels import load_schemes

    schemes = load_schemes(project_path("data/processed"),
                           project_path("results/labels/merge_groups.json"))
    return schemes[scheme].classes


# ------------------------------------------------------------------ explain
def part_explain(log: Any, exp: Experiment) -> None:
    import shap

    protocol, scheme_name, seed = "random_dedup", "L8", int(exp.exp["seeds"][0])
    scheme = exp.schemes[scheme_name]
    train = exp.sample(protocol, seed, "train_capped")
    test = exp.sample(protocol, seed, "test")
    prep = Preprocessor(drop=list(exp.mcfg["features"]["drop"])).fit(train[exp.features])
    params = exp.params(RunSpec(protocol, scheme_name, "xgboost", "capped",
                                "tuned_from_random", seed, PROFILE), prep)
    model = make_model("xgboost", params, seed, exp.n_jobs)
    model.fit(prep.transform(train[exp.features]),
              scheme.encode(train[LABEL_CODE].to_numpy()))
    y_test = scheme.encode(test[LABEL_CODE].to_numpy())
    pick = cap_per_class(np.arange(len(test)), y_test, SHAP_ROWS_PER_CLASS,
                         seed_rng(seed, "shap")).row_ids
    x = prep.transform(test.iloc[pick][exp.features])
    t0 = time.perf_counter()
    values = np.asarray(shap.TreeExplainer(model.model).shap_values(x))
    if values.ndim == 3 and values.shape[0] == len(x):        # (rows, features, classes)
        values = values.transpose(2, 0, 1)
    classes = [scheme.classes[c] for c in model.classes_]
    rows = [{"class": cls, "feature": f, "mean_abs_shap": float(np.abs(values[ci][:, fi]).mean())}
            for ci, cls in enumerate(classes) for fi, f in enumerate(prep.columns)]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "shap_importance.csv", index=False)
    overall = table.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
    overall.to_csv(OUT / "shap_overall.csv")
    log.info("explain: SHAP on %d rows in %.0fs; top features %s", len(x),
             time.perf_counter() - t0, list(overall.index[:5]))


# ------------------------------------------------------------------ cost
def _latency(model: Any, x: np.ndarray, repeats: int) -> dict[str, float]:
    single = []
    for i in range(repeats):
        row = x[i % len(x): i % len(x) + 1]
        t0 = time.perf_counter()
        model.predict(row)
        single.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    model.predict(x)
    batch = time.perf_counter() - t0
    return {"single_row_median_ms": 1e3 * float(np.median(single)),
            "single_row_p95_ms": 1e3 * float(np.quantile(single, 0.95)),
            "batch_us_per_row": 1e6 * batch / len(x)}


def _set_threads(model: Any, n: int) -> None:
    inner = model.model
    if hasattr(inner, "n_jobs"):
        inner.set_params(n_jobs=n)


def part_cost(log: Any, exp: Experiment) -> None:
    protocol, scheme_name, seed = "random_dedup", "L8", int(exp.exp["seeds"][0])
    scheme = exp.schemes[scheme_name]
    train = exp.sample(protocol, seed, "train_capped")
    test = exp.sample(protocol, seed, "test")
    prep = Preprocessor(drop=list(exp.mcfg["features"]["drop"])).fit(train[exp.features])
    x_train = prep.transform(train[exp.features])
    y_train = scheme.encode(train[LABEL_CODE].to_numpy())
    x_test = prep.transform(test[exp.features])[:COST_BATCH_ROWS]
    models_dir = project_path("models")
    models_dir.mkdir(exist_ok=True)
    rows = []
    for name in TREE_MODELS:
        spec = RunSpec(protocol, scheme_name, name, "capped", "tuned_from_random", seed, PROFILE)
        model = make_model(name, exp.params(spec, prep), seed, exp.n_jobs)
        # Fit time on the clock that stops during sleep; the latency loop keeps perf_counter
        # (finer resolution) and records how long the laptop slept while it ran (should be 0).
        t0 = awake_seconds()
        model.fit(x_train, y_train)
        fit_s = awake_seconds() - t0
        path = models_dir / f"{name}_{protocol}_{scheme_name}_s{seed}.joblib"
        joblib.dump({"model": model, "preprocessor": prep}, path, compress=3)
        result = {"model": name, "file_mb": path.stat().st_size / 1e6, "fit_seconds": fit_s,
                  "batch_rows": len(x_test), "latency_repeats": LATENCY_REPEATS,
                  "train_rows": len(x_train)}
        awake0, wall0 = awake_seconds(), time.perf_counter()
        for threads in (1, exp.n_jobs):
            _set_threads(model, threads)
            lat = _latency(model, x_test, repeats=LATENCY_REPEATS)
            result.update({f"{k}_{threads}thr": v for k, v in lat.items()})
        slept = (time.perf_counter() - wall0) - (awake_seconds() - awake0)
        result["slept_during_latency_s"] = max(0.0, round(slept, 1))
        if slept > 1:
            log.warning("cost %s: the laptop slept %.0f s during the latency test", name, slept)
        rows.append(result)
        log.info("cost %s: %s", name, {k: round(v, 3) for k, v in result.items()
                                        if isinstance(v, float)})
    table = pd.DataFrame(rows)
    table["machine"] = f"{cpu_name()}, {platform.system()} {platform.release()}"
    table["threads_note"] = f"1 thread and {exp.n_jobs} threads; batch of {len(x_test):,} rows"
    table.to_csv(OUT / "cost.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="full", choices=["quick", "full"])
    parser.add_argument("--part", default="all",
                        choices=["stats", "calibration", "errors", "explain", "cost", "all"])
    args = parser.parse_args()
    global PROFILE, OUT
    PROFILE = args.profile
    OUT = project_path("results/analysis" if PROFILE == "full" else "results/analysis_quick")
    OUT.mkdir(parents=True, exist_ok=True)
    log = get_logger("07_analysis", project_path("logs"))
    exp = Experiment(PROFILE, log)
    parts = ["stats", "calibration", "errors", "explain", "cost"]
    for part in parts if args.part == "all" else [args.part]:
        t0 = time.time()
        log.info("== %s", part)
        if part in ("explain", "cost"):
            {"explain": part_explain, "cost": part_cost}[part](log, exp)
        else:
            {"stats": part_stats, "calibration": part_calibration, "errors": part_errors}[
                part](log)
        log.info("== %s done in %.1f min", part, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
