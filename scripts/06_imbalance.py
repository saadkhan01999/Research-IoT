"""Step 6a: imbalance treatments on the deduplicated split (training rows only).

Usage: python scripts/06_imbalance.py --profile quick|full [--models ...] [--schemes ...]
       [--treatments none weights under smotetomek] [--seeds ...]

Every treatment starts from the same natural-mix training sample of the seed
(experiment.imbalance.train_rows rows) and is scored on the same test sample. Settings are
the ones tuned on the random split (capped training). Finished runs are skipped.
"""

from __future__ import annotations

import argparse
import time

from iotml.config import load_config, project_path
from iotml.experiment import Experiment, RunSpec
from iotml.imbalance import TREATMENTS
from iotml.logs import get_logger
from iotml.system import keep_awake


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=["quick", "full"], required=True)
    parser.add_argument("--protocol", default="random_dedup")
    parser.add_argument("--models", nargs="+", default=["xgboost", "random_forest"])
    parser.add_argument("--schemes", nargs="+", default=["L8", "L34"])
    parser.add_argument("--treatments", nargs="+", default=list(TREATMENTS))
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    log = get_logger(f"06_imbalance_{args.profile}", project_path("logs"))
    exp = Experiment(args.profile, log)
    seeds = args.seeds or cfg["experiment"].get("careful_seeds", cfg["experiment"]["seeds"])
    t0 = time.time()
    with keep_awake(log):
        for seed in seeds:
            for scheme in args.schemes:
                for model in args.models:
                    for treatment in args.treatments:
                        exp.run(RunSpec(args.protocol, scheme, model, f"imb_{treatment}",
                                        "tuned_from_random", int(seed), args.profile))
            log.info("imbalance seed %s done after %.1f min", seed, (time.time() - t0) / 60)
    log.info("all done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
