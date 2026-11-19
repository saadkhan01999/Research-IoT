"""Experiment runner: split, sample, preprocess, (tune), fit, evaluate, save, resume.

One run = (protocol, label scheme, model, training mode, parameter mode, seed, profile). Each
run writes one JSON file; an existing file means the run is done and is skipped. Validation and
test samples depend only on (protocol, seed, profile), so every model and training mode of a
seed is scored on the same rows.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from iotml.audit import read_rows
from iotml.config import load_config, project_path, read_yaml
from iotml.data import FILE_COL, LABEL_CODE, POS_COL, feature_columns
from iotml.envinfo import library_versions
from iotml.features import Preprocessor
from iotml.imbalance import balanced_weights, smote_tomek, undersample
from iotml.labels import LabelScheme, load_schemes
from iotml.metrics import evaluate
from iotml.models import make_model, sample_params
from iotml.sampling import Sample, cap_per_class, seed_rng, uniform
from iotml.splits import (
    TEST,
    TRAIN,
    UNUSED,
    VAL,
    file_grouped_split,
    hash_split,
    order_split,
    part_ids,
    purge,
    random_split,
    unseen_type,
)
from iotml.system import awake_seconds

WEIGHT_COL = "_weight"
ROW_COL = "row_id"


def predict_with_proba(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Predicted labels and class probabilities from one pass over `x`.

    The tree models (scikit-learn, XGBoost, LightGBM) predict the class with the highest
    probability, so taking the arg-max of predict_proba gives the same labels as predict()
    (tests/test_pipeline.py checks this for all five). Scoring the test rows once instead of
    twice halves the prediction time, which is large for LightGBM with many classes. Models
    without predict_proba fall back to predict()."""
    classes = getattr(model, "classes_", None)
    try:
        proba = np.asarray(model.predict_proba(x))
    except (AttributeError, NotImplementedError):
        return np.asarray(model.predict(x)), None
    if classes is None or len(classes) != proba.shape[1]:
        return np.asarray(model.predict(x)), proba
    return np.asarray(classes)[np.argmax(proba, axis=1)], proba


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    scheme: str
    model: str
    train_mode: str      # natural | capped
    params_mode: str     # default | tuned
    seed: int
    profile: str

    def path(self, root: Path) -> Path:
        return (root / self.protocol / self.scheme / self.model /
                f"{self.train_mode}_{self.params_mode}_s{self.seed}_{self.profile}.json")


class Experiment:
    def __init__(self, profile: str, log) -> None:
        self.profile = profile
        self.cfg = load_config(profile)
        self.exp = self.cfg["experiment"]
        self.mcfg = read_yaml(project_path("config") / "models.yaml")
        self.log = log
        self.n_jobs = int(self.cfg["runtime"]["n_jobs"])
        self.cache = project_path(self.cfg["audit"]["cache"])
        self.features = feature_columns(self.cache)
        self.schemes = load_schemes(self.cache.parent,
                                    project_path("results/labels/merge_groups.json"))
        self.out_root = project_path(self.cfg["paths"]["results"]) / "experiments"
        self.interim = project_path(self.cfg["paths"]["interim"])
        self._labels34: np.ndarray | None = None
        # last SMOTE-Tomek result: (key, x, y, seconds). The treated rows do not depend on the
        # model, so the next model with the same sample, scheme and seed reuses them (D-038).
        self._smote_cache: tuple[tuple[Any, ...], np.ndarray, np.ndarray, float] | None = None
        self._treatment_seconds: float | None = None

    # ------------------------------------------------------------ data
    @property
    def labels34(self) -> np.ndarray:
        if self._labels34 is None:
            self._labels34 = pq.read_table(self.cache, columns=[LABEL_CODE]).column(0).to_numpy()
        return self._labels34

    def split(self, protocol: str, seed: int) -> np.ndarray:
        """Part code of every row for one protocol and seed (cached), plus a JSON record of
        how the split was made (results/splits/)."""
        path = self.interim / "splits" / f"{protocol}_s{seed}.npy"
        if path.exists():
            return np.load(path)
        parts, info = self.make_split(protocol, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, parts)
        counts = {name: int((parts == code).sum()) for code, name in
                  ((TRAIN, "train"), (VAL, "val"), (TEST, "test"), (UNUSED, "unused"))}
        record = self.out_root.parent / "splits" / f"{protocol}_s{seed}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"protocol": protocol, "seed": seed, "rows": counts,
                                     **info}, indent=1, default=_json_default), encoding="utf-8")
        return parts

    def make_split(self, protocol: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        shares = self.mcfg["split"]["shares"]
        if protocol == "random":
            return random_split(self.labels34, shares, seed_rng(seed, "split", protocol)), {}
        if protocol.startswith("unseen_type__"):
            label = protocol.split("__", 1)[1]
            code = self.schemes["L34"].classes.index(label)
            base = self.split("random_dedup", seed)
            return unseen_type(base, self.labels34, code), {"held_out": label,
                                                           "base_protocol": "random_dedup"}
        index = self.row_index()
        gap = int(self.mcfg["split"]["gap_rows"])
        if protocol == "random_dedup":
            parts = hash_split(index["row_hash"], shares, seed)
            parts[~index["first_copy"]] = UNUSED
            return parts, {"duplicate_copies_dropped": int((~index["first_copy"]).sum()),
                           "grouping": "identical feature vectors share a part"}
        sources = pd.read_csv(self.cache.parent / "source_files.csv")
        if protocol == "file_grouped":
            parts, used = file_grouped_split(
                index["file_ids"], index["positions"], sources, shares,
                seed_rng(seed, "split", protocol),
                int(self.mcfg["split"]["min_files_for_file_split"]), gap)
            info: dict[str, Any] = {"files": used.to_dict("records")}
        elif protocol == "order_within_file":
            parts = order_split(index["file_ids"], index["positions"],
                                sources["rows"].to_numpy(), shares, gap)
            info = {}
        else:
            raise ValueError(f"Unknown protocol '{protocol}'")
        parts, removed = purge(parts, index["row_hash"])
        return parts, {**info, "gap_rows": gap, "purged_rows_seen_in_train": removed}

    def row_index(self) -> dict[str, np.ndarray]:
        """Per-row hash, first-copy flag, source file and position (built by 05_row_index)."""
        processed = self.cache.parent
        if not (processed / "row_hash.npy").exists():
            raise FileNotFoundError("Run scripts/05_row_index.py first")
        table = pq.read_table(self.cache, columns=[FILE_COL, POS_COL])
        return {"row_hash": np.load(processed / "row_hash.npy", mmap_mode="r"),
                "first_copy": np.load(processed / "first_copy.npy"),
                "file_ids": table.column(0).to_numpy(), "positions": table.column(1).to_numpy()}

    def sample(self, protocol: str, seed: int, part: str) -> pd.DataFrame:
        """Rows of one sample (train_natural, train_capped, val, test) with weights, cached.
        The cache name carries the sample size, so changing a cap never reuses an old sample."""
        path = (self.interim / "samples" /
                f"{protocol}_s{seed}_{self.profile}_{part}_n{self.sample_size(part)}.parquet")
        if path.exists():
            return pd.read_parquet(path)
        parts = self.split(protocol, seed)
        rng = seed_rng(seed, "sample", protocol, part)
        sampling = self.exp["sampling"]
        if part == "train_natural":
            ids = part_ids(parts, TRAIN)
            s: Sample = uniform(ids, int(self.exp["natural_train_rows"]), rng)
        elif part == "train_imbalance":
            ids = part_ids(parts, TRAIN)
            s = uniform(ids, int(self.exp["imbalance"]["train_rows"]), rng)
        else:
            key = {"train_capped": "train", "val": "val", "test": "test"}[part]
            code = {"train": TRAIN, "val": VAL, "test": TEST}[key]
            ids = part_ids(parts, code)
            s = cap_per_class(ids, self.labels34[ids], sampling[key]["cap_per_label"], rng)
        self.log.info("  reading %d rows for %s/s%d/%s", len(s), protocol, seed, part)
        frame = read_rows(self.cache, s.row_ids, self.features + [LABEL_CODE], float32=True)
        frame[WEIGHT_COL] = s.weights
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    def sample_size(self, part: str) -> int:
        """Rows (natural samples) or rows per label (capped samples) of a sample part."""
        if part == "train_natural":
            return int(self.exp["natural_train_rows"])
        if part == "train_imbalance":
            return int(self.exp["imbalance"]["train_rows"])
        key = {"train_capped": "train", "val": "val", "test": "test"}[part]
        return int(self.exp["sampling"][key]["cap_per_label"])

    # ------------------------------------------------------------ one run
    def check_memory(self, max_wait_s: int | None = None, poll_s: int = 30) -> float:
        """Wait before a run while the machine is short of RAM (with too little free memory
        Windows swaps to disk and a 20-second fit can take over an hour). A short spike from
        another program only delays the run; after `max_wait_s` the run stops."""
        need = float(self.cfg["runtime"]["min_free_ram_gb"])
        if max_wait_s is None:
            max_wait_s = int(60 * float(self.cfg["runtime"].get("max_wait_min", 20)))
        waited = 0
        while True:
            free_gb = psutil.virtual_memory().available / 1e9
            if free_gb >= need:
                return free_gb
            if waited >= max_wait_s:
                raise MemoryError(
                    f"Only {free_gb:.2f} GB of RAM is free (need {need} GB) after waiting "
                    f"{waited // 60} min. Close other programs (browsers, editors) and start "
                    "the script again; finished runs are kept.")
            if waited == 0:
                self.log.warning("only %.2f GB of RAM free (need %.1f); waiting up to %d min",
                                 free_gb, need, max_wait_s // 60)
            time.sleep(poll_s)
            waited += poll_s

    def run(self, spec: RunSpec) -> dict[str, Any] | None:
        out = spec.path(self.out_root)
        if out.exists():
            return None
        free_before = self.check_memory()
        scheme = self.schemes[spec.scheme]
        imbalance = spec.train_mode.startswith("imb_")
        train_part = "train_imbalance" if imbalance else f"train_{spec.train_mode}"
        train = self.sample(spec.protocol, spec.seed, train_part)
        test = self.sample(spec.protocol, spec.seed, "test")
        prep = Preprocessor(drop=list(self.mcfg["features"]["drop"])).fit(train[self.features])
        x_train = prep.transform(train[self.features])
        y_train = scheme.encode(train[LABEL_CODE].to_numpy())
        weights = None
        # Times use awake_seconds, which stops while the laptop sleeps (closed lid).
        t0 = awake_seconds()
        self._treatment_seconds = None
        if imbalance:
            x_train, y_train, weights = self.treat(spec, x_train, y_train)
        treat_s = awake_seconds() - t0
        if self._treatment_seconds is not None:
            # SMOTE-Tomek: the time the treatment took, also when its rows were reused (D-038)
            treat_s = self._treatment_seconds
        params = self.params(spec, prep)

        model = make_model(spec.model, params, spec.seed, self.n_jobs)
        t0, wall0 = awake_seconds(), time.perf_counter()
        model.fit(x_train, y_train, sample_weight=weights)
        fit_s = awake_seconds() - t0
        fit_wall_s = time.perf_counter() - wall0
        x_test = prep.transform(test[self.features])
        # One pass over the test rows gives both the class and the confidence (decision D-027).
        t0 = awake_seconds()
        pred, proba = predict_with_proba(model, x_test)
        predict_s = awake_seconds() - t0
        y_test = scheme.encode(test[LABEL_CODE].to_numpy())
        metrics = evaluate(y_test, pred, test[WEIGHT_COL].to_numpy(), scheme.classes)
        if spec.protocol.startswith("unseen_type__"):
            metrics["unseen_type"] = self.unseen_type_result(spec, test, pred, scheme)
        self.save_predictions(spec, test, y_test, pred, model, x_test, proba=proba)

        result = {
            "spec": asdict(spec), "params": params, "features_used": prep.columns,
            "train_rows": int(len(y_train)), "train_rows_before_treatment": int(len(train)),
            "treatment_seconds": treat_s, "test_rows": int(len(test)),
            "train_class_counts": {int(c): int(n) for c, n in
                                   zip(*np.unique(y_train, return_counts=True), strict=True)},
            "fit_seconds": fit_s, "fit_wall_seconds": fit_wall_s, "predict_seconds": predict_s,
            "predict_us_per_row_batch": 1e6 * predict_s / max(len(test), 1),
            "model_bytes": self.model_size(model.model),
            "free_ram_gb_before": round(free_before, 2),
            "free_ram_gb_after": round(psutil.virtual_memory().available / 1e9, 2),
            "metrics": metrics, "libraries": library_versions(),
            "config_hash": self.config_hash(), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
        self.log.info("  %s %s %s %s s%d: macro-F1 %.4f, accuracy %.4f, fit %.0fs", spec.scheme,
                      spec.model, spec.train_mode, spec.params_mode, spec.seed,
                      metrics["macro_f1"], metrics["accuracy"], fit_s)
        return result

    # ------------------------------------------------------------ hyperparameters
    def params(self, spec: RunSpec, prep: Preprocessor) -> dict[str, Any]:
        base = dict(self.mcfg["models"][spec.model]["defaults"])
        if spec.model == "xgboost":
            base.setdefault("tree_method", "hist")
        if spec.params_mode == "default":
            return base
        # "tuned_from_random": the settings tuned on the random split, reused unchanged under a
        # strict protocol, so only the split differs between protocols (Step 5, D-020).
        source = "random" if spec.params_mode == "tuned_from_random" else spec.protocol
        # imbalance runs (imb_*) reuse the settings tuned for capped training
        tuned_mode = "capped" if spec.train_mode.startswith("imb_") else spec.train_mode
        path = (self.out_root / source / "tuning" /
                f"{spec.model}_{tuned_mode}_{self.profile}.json")
        if not path.exists():
            if source != spec.protocol:
                raise FileNotFoundError(f"No tuned settings for {spec.model} on the random "
                                        f"split ({path}); run Step 4 first")
            self.tune(spec, prep, base, path)
        return {**base, **json.loads(path.read_text(encoding="utf-8"))["best_params"]}

    def tune(self, spec: RunSpec, prep: Preprocessor, base: dict[str, Any], path: Path) -> None:
        """Random search with the same budget for every model, scored on the validation sample
        (macro-F1, natural-mix weights). Done once per model, protocol, training mode and
        profile, on the first seed and the tuning scheme; the result is reused."""
        tcfg = self.mcfg["tuning"]
        seed = int(self.exp["seeds"][0])
        scheme: LabelScheme = self.schemes[tcfg["tuned_on_scheme"]]
        train = self.sample(spec.protocol, seed, f"train_{spec.train_mode}")
        val = self.sample(spec.protocol, seed, "val")
        rng = seed_rng(seed, "tune", spec.model, spec.protocol)
        cap = int(tcfg["max_train_rows"])
        if len(train) > cap:
            train = train.iloc[np.sort(rng.choice(len(train), cap, replace=False))]
        tprep = Preprocessor(drop=prep.drop).fit(train[self.features])
        x_tr, y_tr = tprep.transform(train[self.features]), scheme.encode(train[LABEL_CODE])
        x_va, y_va = tprep.transform(val[self.features]), scheme.encode(val[LABEL_CODE])
        space = self.mcfg["models"][spec.model]["search"]
        trials = []
        for params in sample_params(space, int(self.exp["tuning_trials"]), rng):
            model = make_model(spec.model, {**base, **params}, seed, self.n_jobs)
            t0 = time.perf_counter()
            model.fit(x_tr, y_tr)
            score = evaluate(y_va, model.predict(x_va), val[WEIGHT_COL].to_numpy(),
                             scheme.classes)["macro_f1"]
            trials.append({"params": params, "val_macro_f1": score,
                           "seconds": time.perf_counter() - t0})
            self.log.info("    tune %s %s: %.4f", spec.model, params, score)
        best = max(trials, key=lambda t: t["val_macro_f1"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"best_params": best["params"], "trials": trials,
                                    "seed": seed, "scheme": scheme.name, "train_rows": len(train),
                                    "metric": tcfg["metric"]}, indent=1, default=_json_default),
                        encoding="utf-8")

    def treat(self, spec: RunSpec, x: np.ndarray,
              y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply the imbalance treatment named in the training mode (training rows only)."""
        treatment = spec.train_mode.removeprefix("imb_")
        target = int(self.exp["imbalance"]["target_per_class"])
        rng = seed_rng(spec.seed, "imbalance", treatment, spec.scheme)
        if treatment == "none":
            return x, y, None
        if treatment == "weights":
            return x, y, balanced_weights(y)
        if treatment == "under":
            keep = undersample(y, target, rng)
            return x[keep], y[keep], None
        if treatment == "smotetomek":
            # The result depends on the rows, the labels, the target and the seed only. The key
            # carries a fingerprint of the rows and labels, so other data is never reused.
            key = (spec.protocol, spec.scheme, spec.seed, spec.profile, target, x.shape,
                   float(np.nansum(x)), int(np.asarray(y, dtype=np.int64).sum()))
            if self._smote_cache is None or self._smote_cache[0] != key:
                self._smote_cache = None          # free the old rows before making new ones
                t0 = awake_seconds()
                xr, yr = smote_tomek(x, y, target, spec.seed)
                self._smote_cache = (key, xr, yr, awake_seconds() - t0)
            else:
                self.log.info("  SMOTE-Tomek rows of this sample reused (made in %.0f s)",
                              self._smote_cache[3])
            self._treatment_seconds = self._smote_cache[3]
            return self._smote_cache[1], self._smote_cache[2].copy(), None
        raise ValueError(f"Unknown imbalance treatment '{treatment}'")

    def unseen_type_result(self, spec: RunSpec, test: pd.DataFrame, pred: np.ndarray,
                           scheme: LabelScheme) -> dict[str, Any]:
        """For the held-out attack label: how its test rows were classified. With the L8 scheme
        'family_recall' is the share assigned to the correct family."""
        label = spec.protocol.split("__", 1)[1]
        code = self.schemes["L34"].classes.index(label)
        mask = test[LABEL_CODE].to_numpy() == code
        own_class = int(scheme.code_map[code])
        predicted = pd.Series(pred[mask]).map(lambda c: scheme.classes[int(c)])
        return {"held_out": label, "test_rows": int(mask.sum()),
                "own_class": scheme.classes[own_class],
                "recall_own_class": float((pred[mask] == own_class).mean()) if mask.any() else None,
                "predicted_as": predicted.value_counts(normalize=True).round(4).to_dict()}

    def save_predictions(self, spec: RunSpec, test: pd.DataFrame, y_test: np.ndarray,
                         pred: np.ndarray, model: Any, x_test: np.ndarray,
                         proba: np.ndarray | None = None) -> None:
        """Test-row predictions for bootstrap intervals, statistics and calibration (Step 7).
        Stored outside git (data/interim/predictions) because of their size. `proba`, when
        given, saves a second pass over the test rows."""
        if proba is None:
            try:
                proba = np.asarray(model.predict_proba(x_test))
            except (AttributeError, NotImplementedError):
                proba = None
        if proba is not None:
            confidence = proba.max(axis=1).astype(np.float16)
        else:
            confidence = np.full(len(pred), np.nan, dtype=np.float16)
        path = (self.interim / "predictions" / spec.protocol / spec.scheme / spec.model /
                spec.path(Path(".")).name.replace(".json", ".npz"))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, row_id=test[ROW_COL].to_numpy().astype(np.uint32),
                            y_true=y_test.astype(np.int8), y_pred=np.asarray(pred, np.int8),
                            weight=test[WEIGHT_COL].to_numpy().astype(np.float32),
                            confidence=confidence)

    def model_size(self, model: Any) -> int:
        """Size of the pickled model, written to a temporary file (not copied in memory)."""
        path = self.interim / "tmp_model_size.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size = path.stat().st_size
        path.unlink()
        return size

    def config_hash(self) -> str:
        text = json.dumps([self.cfg, self.mcfg], sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")
