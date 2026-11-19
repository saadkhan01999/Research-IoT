"""Step 4: the "as published" baseline on the conventional random split.

Usage: python scripts/04_baseline.py --profile quick|full [--models ...] [--schemes ...]
       [--train-modes natural capped] [--params-modes default tuned] [--seeds ...]

Runs every (seed, training mode, label scheme, model, parameter mode) of the profile on the
random protocol. Finished runs are skipped, so the script can be stopped and restarted.
"""

from __future__ import annotations

import argparse
import time

from iotml.config import load_config, project_path
from iotml.experiment import Experiment, RunSpec
from iotml.logs import get_logger
from iotml.models import TREE_MODELS
from iotml.system import keep_awake


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--models", nargs="+", default=list(TREE_MODELS))
    parser.add_argument("--schemes", nargs="+")
    parser.add_argument("--train-modes", nargs="+", default=["natural", "capped"])
    parser.add_argument("--params-modes", nargs="+", default=["default", "tuned"])
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    exp_cfg = load_config(args.profile)["experiment"]
    log = get_logger(f"04_baseline_{args.profile}", project_path("logs"))
    exp = Experiment(args.profile, log)
    seeds = args.seeds or exp_cfg["seeds"]
    careful = set(exp_cfg.get("careful_seeds", seeds))   # D-021: capped runs, first seeds
    schemes = args.schemes or exp_cfg["label_schemes"]
    t_start = time.time()
    with keep_awake(log):
        for seed in seeds:
            for train_mode in args.train_modes:
                if train_mode == "capped" and not args.seeds and seed not in careful:
                    continue
                for scheme in schemes:
                    for model in args.models:
                        for params_mode in args.params_modes:
                            exp.run(RunSpec("random", scheme, model, train_mode, params_mode,
                                            int(seed), args.profile))
            log.info("seed %s done after %.1f min", seed, (time.time() - t_start) / 60)
    log.info("all done in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
