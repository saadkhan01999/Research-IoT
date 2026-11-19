"""Step 6b: neural models.

Usage: python scripts/06_neural.py --profile quick|full [--part rows|windows|all] [--seeds ...]

Part "rows": an MLP with cross-entropy and with focal loss on single rows, under the
random_dedup, file_grouped and order_within_file protocols (8-class task).
Part "windows": a 1D-CNN and a GRU on windows of k consecutive rows of one capture, under the
order_within_file protocol (the only one where windows cannot straddle train and test), next
to XGBoost on the window's last row and on window statistics, all scored on the same windows.
Results use the same JSON layout as the tree runs; finished runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from iotml.audit import read_rows
from iotml.config import load_config, project_path
from iotml.data import LABEL_CODE
from iotml.envinfo import library_versions
from iotml.experiment import (
    ROW_COL,
    WEIGHT_COL,
    Experiment,
    RunSpec,
    _json_default,
    predict_with_proba,
)
from iotml.features import Preprocessor
from iotml.logs import get_logger
from iotml.metrics import evaluate
from iotml.models import make_model
from iotml.neural import NeuralClassifier, NeuralConfig
from iotml.sampling import cap_per_class, seed_rng
from iotml.splits import TEST, TRAIN, VAL
from iotml.system import awake_seconds, keep_awake
from iotml.windows import gather, valid_window_ends, window_rows

ROW_PROTOCOLS = ["random_dedup", "file_grouped", "order_within_file"]
SCHEME = "L8"


def write(exp: Experiment, spec: RunSpec, result: dict[str, Any]) -> None:
    out = spec.path(exp.out_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.update({"spec": asdict(spec), "libraries": library_versions(),
                   "config_hash": exp.config_hash(),
                   "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    out.write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")


def neural_cfg(cfg: dict[str, Any], arch: str, loss: str) -> NeuralConfig:
    n = cfg["experiment"]["neural"]
    return NeuralConfig(arch=arch, loss=loss, epochs=int(n["epochs"]),
                        batch_size=int(n["batch_size"]), lr=float(n["lr"]),
                        patience=int(n["patience"]), hidden=int(n["hidden"]),
                        threads=int(cfg["runtime"]["n_jobs"]))


def fit_and_score(exp: Experiment, spec: RunSpec, model: Any, fit_args: tuple[Any, ...],
                  x_test: np.ndarray, y_test: np.ndarray, test_frame: pd.DataFrame,
                  extra: dict[str, Any]) -> None:
    """Fit `model` with `fit_args`, score it on the test windows/rows, save JSON and
    predictions in the same layout as the tree runs."""
    exp.check_memory()
    scheme = exp.schemes[spec.scheme]
    # awake_seconds stops while the laptop sleeps, so a closed lid does not inflate the times.
    t0, wall0 = awake_seconds(), time.perf_counter()
    model.fit(*fit_args)
    fit_s = awake_seconds() - t0
    fit_wall_s = time.perf_counter() - wall0
    t0 = awake_seconds()
    pred, proba = predict_with_proba(model, x_test)  # one pass (decision D-027)
    predict_s = awake_seconds() - t0
    metrics = evaluate(y_test, pred, test_frame[WEIGHT_COL].to_numpy(), scheme.classes)
    exp.save_predictions(spec, test_frame, y_test, pred, model, x_test, proba=proba)
    write(exp, spec, {**extra, "metrics": metrics, "fit_seconds": fit_s,
                      "fit_wall_seconds": fit_wall_s, "predict_seconds": predict_s,
                      "predict_us_per_row_batch": 1e6 * predict_s / max(len(y_test), 1)})
    exp.log.info("  %s %s %s s%d: macro-F1 %.4f, accuracy %.4f, fit %.0fs", spec.protocol,
                 spec.scheme, spec.model, spec.seed, metrics["macro_f1"], metrics["accuracy"],
                 fit_s)


# ---------------------------------------------------------------- part A: rows
def run_rows(exp: Experiment, cfg: dict[str, Any], seeds: list[int], profile: str) -> None:
    scheme = exp.schemes[SCHEME]
    drop = list(exp.mcfg["features"]["drop"])
    for protocol in ROW_PROTOCOLS:
        for seed in seeds:
            specs = {loss: RunSpec(protocol, SCHEME, f"mlp_{loss}", "capped", "default", seed,
                                   profile) for loss in ("ce", "focal")}
            if all(s.path(exp.out_root).exists() for s in specs.values()):
                continue
            train = exp.sample(protocol, seed, "train_capped")
            val = exp.sample(protocol, seed, "val")
            test = exp.sample(protocol, seed, "test")
            prep = Preprocessor(drop=drop).fit(train[exp.features])
            x_tr, x_va, x_te = (prep.transform(f[exp.features]) for f in (train, val, test))
            y_tr, y_va, y_te = (scheme.encode(f[LABEL_CODE].to_numpy())
                                for f in (train, val, test))
            for loss, spec in specs.items():
                if spec.path(exp.out_root).exists():
                    continue
                clf = NeuralClassifier(neural_cfg(cfg, "mlp", loss), seed)
                fit_and_score(
                    exp, spec, clf,
                    (x_tr, y_tr, x_va, y_va, val[WEIGHT_COL].to_numpy(), scheme.n_classes),
                    x_te, y_te, test,
                    {"params": asdict(clf.cfg) | {"history": None}, "train_rows": len(y_tr),
                     "test_rows": len(y_te)})
                result = json.loads(spec.path(exp.out_root).read_text(encoding="utf-8"))
                result["history"] = clf.cfg.history
                result["n_parameters"] = clf.n_parameters()
                spec.path(exp.out_root).write_text(json.dumps(result, indent=1,
                                                              default=_json_default),
                                                   encoding="utf-8")


# ---------------------------------------------------------------- part B: windows
def window_sample(exp: Experiment, cfg: dict[str, Any], seed: int, profile: str,
                  part: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(raw window tensor, L34 labels of the windows, frame with row_id and weight of the
    window's last row), cached per seed and part."""
    wcfg = cfg["experiment"]["windows"]
    k = int(wcfg["k"])
    cache = exp.interim / "windows" / f"order_k{k}_s{seed}_{profile}_{part}.npz"
    if cache.exists():
        z = np.load(cache)
        frame = pd.DataFrame({ROW_COL: z["ends"], WEIGHT_COL: z["weights"]})
        return z["x"], z["y34"], frame
    parts = exp.split("order_within_file", seed)
    sources = pd.read_csv(exp.cache.parent / "source_files.csv")
    code = {"train": TRAIN, "val": VAL, "test": TEST}[part]
    ends = valid_window_ends(np.where(parts == code, parts, -1).astype(np.int8), sources, k)
    s = cap_per_class(ends, exp.labels34[ends], int(wcfg[f"{part}_cap"]),
                      seed_rng(seed, "windows", part))
    rows = window_rows(s.row_ids, k)
    needed = np.unique(rows.ravel())
    exp.log.info("  reading %d rows for %d %s windows (seed %d)", needed.size, len(s), part,
                 seed)
    frame = read_rows(exp.cache, needed, exp.features, float32=True)
    x = gather(frame, rows, exp.features)
    y34 = exp.labels34[s.row_ids]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y34=y34, ends=s.row_ids, weights=s.weights)
    return x, y34, pd.DataFrame({ROW_COL: s.row_ids, WEIGHT_COL: s.weights})


def window_features(x: np.ndarray) -> np.ndarray:
    """Last row, mean and standard deviation over the window, side by side."""
    return np.concatenate([x[:, -1, :], x.mean(axis=1), x.std(axis=1)], axis=1)


def last_row(x: np.ndarray) -> np.ndarray:
    return x[:, -1, :]


def prepare_windows(x: np.ndarray, prep: Preprocessor, features: list[str]) -> np.ndarray:
    """Apply the training-fitted preprocessing to every row of every window."""
    flat = pd.DataFrame(x.reshape(-1, x.shape[-1]), columns=features)
    return prep.transform(flat).reshape(x.shape[0], x.shape[1], -1)


def run_windows(exp: Experiment, cfg: dict[str, Any], seeds: list[int], profile: str) -> None:
    scheme = exp.schemes[SCHEME]
    drop = list(exp.mcfg["features"]["drop"])
    protocol = "order_windows"
    tuned = json.loads((exp.out_root / "random" / "tuning" /
                        f"xgboost_capped_{profile}.json").read_text(encoding="utf-8"))
    xgb_params = {"tree_method": "hist", **tuned["best_params"]}
    for seed in seeds:
        models = ["cnn1d_ce", "gru_ce", "gru_focal", "xgboost_last_row", "xgboost_window_stats"]
        specs = {m: RunSpec(protocol, SCHEME, m, "windows", "default", seed, profile)
                 for m in models}
        if all(s.path(exp.out_root).exists() for s in specs.values()):
            continue
        data = {p: window_sample(exp, cfg, seed, profile, p) for p in ("train", "val", "test")}
        prep = Preprocessor(drop=drop).fit(
            pd.DataFrame(data["train"][0].reshape(-1, len(exp.features)), columns=exp.features))
        x = {p: prepare_windows(data[p][0], prep, exp.features) for p in data}
        y = {p: scheme.encode(data[p][1]) for p in data}
        test_frame = data["test"][2]
        w_val = data["val"][2][WEIGHT_COL].to_numpy()
        for name, spec in specs.items():
            if spec.path(exp.out_root).exists():
                continue
            extra = {"train_rows": len(y["train"]), "test_rows": len(y["test"]),
                     "window_length": int(cfg["experiment"]["windows"]["k"])}
            if name.startswith("xgboost"):
                feats = last_row if name == "xgboost_last_row" else window_features
                model = make_model("xgboost", xgb_params, seed, exp.n_jobs)
                fit_and_score(exp, spec, model, (feats(x["train"]), y["train"]),
                              feats(x["test"]), y["test"], test_frame,
                              {**extra, "params": xgb_params})
            else:
                arch, loss = name.split("_", 1)
                clf = NeuralClassifier(neural_cfg(cfg, arch, loss), seed)
                fit_and_score(exp, spec, clf,
                              (x["train"], y["train"], x["val"], y["val"], w_val,
                               scheme.n_classes),
                              x["test"], y["test"], test_frame,
                              {**extra, "params": asdict(clf.cfg) | {"history": None}})
        exp.log.info("windows seed %s done", seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--part", choices=["rows", "windows", "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = get_logger(f"06_neural_{args.profile}", project_path("logs"))
    exp = Experiment(args.profile, log)
    exp_cfg = cfg["experiment"]
    seeds = [int(s) for s in (args.seeds or exp_cfg.get("careful_seeds", exp_cfg["seeds"]))]
    t0 = time.time()
    with keep_awake(log):
        if args.part in ("rows", "all"):
            run_rows(exp, cfg, seeds, args.profile)
        if args.part in ("windows", "all"):
            run_windows(exp, cfg, seeds, args.profile)
    log.info("all done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
