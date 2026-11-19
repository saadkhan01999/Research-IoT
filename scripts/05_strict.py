"""Step 5: the strict split protocols (capped training; by default the settings tuned on the
random split are reused, so only the split changes between protocols).

Usage: python scripts/05_strict.py --profile quick|full
       [--protocols random_dedup file_grouped order_within_file]
       [--unseen all | --unseen <label> ...] [--schemes ...] [--models ...] [--seeds ...]

By default every protocol reuses the settings tuned on the random split (D-020), so only the
split changes; `--params-mode tuned` re-tunes on each protocol's own validation part instead
(same trial budget for every model). `--unseen all` adds one protocol per attack label whose
family has more than one label (the label is removed from training; its test rows show which
family the model gives a type it has never seen). Finished runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import time

from iotml.config import load_config, project_path
from iotml.experiment import Experiment, RunSpec
from iotml.logs import get_logger
from iotml.models import TREE_MODELS
from iotml.system import keep_awake

BASE_PROTOCOLS = ["random_dedup", "file_grouped", "order_within_file"]


def unseen_labels(cache_dir) -> list[str]:
    """Attack labels in families with at least two labels (BruteForce and Benign excluded)."""
    codes = json.loads((cache_dir / "label_codes.json").read_text(encoding="utf-8"))
    family_size: dict[str, int] = {}
    for label in codes["labels"]:
        fam = codes["category_of"][label]
        family_size[fam] = family_size.get(fam, 0) + 1
    return [label for label in codes["labels"]
            if codes["category_of"][label] != "Benign"
            and family_size[codes["category_of"][label]] > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--protocols", nargs="*", default=BASE_PROTOCOLS)
    parser.add_argument("--unseen", nargs="*", default=[])
    parser.add_argument("--schemes", nargs="+")
    parser.add_argument("--models", nargs="+", default=list(TREE_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--params-mode", default="tuned_from_random",
                        choices=["tuned_from_random", "tuned", "default"])
    args = parser.parse_args()

    cfg = load_config(args.profile)
    log = get_logger(f"05_strict_{args.profile}", project_path("logs"))
    exp = Experiment(args.profile, log)
    protocols = list(args.protocols)
    if args.unseen:
        labels = unseen_labels(exp.cache.parent) if args.unseen == ["all"] else args.unseen
        protocols += [f"unseen_type__{label}" for label in labels]
    seeds = args.seeds or cfg["experiment"].get("careful_seeds", cfg["experiment"]["seeds"])
    schemes = args.schemes or ["L8", "L34", "LM"]
    t0 = time.time()
    with keep_awake(log):
        for protocol in protocols:
            proto_schemes = ["L8"] if protocol.startswith("unseen_type__") else schemes
            for seed in seeds:
                for scheme in proto_schemes:
                    for model in args.models:
                        exp.run(RunSpec(protocol, scheme, model, "capped", args.params_mode,
                                        int(seed), args.profile))
                log.info("%s seed %s done after %.1f min", protocol, seed,
                         (time.time() - t0) / 60)
    log.info("all done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
