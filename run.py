"""Single entry point for every task, so the same commands work on Windows and Linux.

Usage: python run.py <task>      (python run.py --list shows all tasks)

The Makefile only forwards to this file, because `make` is not installed on Windows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

TASKS: dict[str, list[list[str]]] = {
    "env": [[PY, "scripts/00_env_check.py"]],
    "checksums": [[PY, "scripts/00_checksums.py"]],
    "verify-data": [[PY, "scripts/00_checksums.py", "--verify"]],
    # Step 2: data cache and audit
    "cache": [[PY, "scripts/02_build_parquet.py"]],
    "audit": [[PY, "scripts/02_audit.py", "--part", "all"]],
    # Step 3: label schemes, merge sensitivity and the sampling plan
    "labels": [[PY, "scripts/03_label_schemes.py"], [PY, "scripts/03_merge_sensitivity.py"],
               [PY, "scripts/03_sampling_plan.py"]],
    # Step 4: random-split baseline. "As published" = natural class mix with library defaults
    # (L8, L34); "careful baseline" = capped training with tuned settings (all four schemes).
    "baseline-quick": [[PY, "scripts/04_baseline.py", "--profile", "quick"],
                       [PY, "scripts/04_summarise.py", "--profile", "quick"]],
    "baseline-full": [
        [PY, "scripts/04_baseline.py", "--profile", "full", "--train-modes", "natural",
         "--params-modes", "default", "--schemes", "L8", "L34"],
        [PY, "scripts/04_baseline.py", "--profile", "full", "--train-modes", "capped",
         "--params-modes", "tuned"],
        [PY, "scripts/04_summarise.py", "--profile", "full"]],
    # Why default LightGBM failed on L34: training log loss per round.
    "diagnose-lightgbm": [[PY, "scripts/04_diagnose_lightgbm.py", "--profile", "full",
                           "--seed", "37", "--threads", "1"]],
    # Step 5: strict split protocols and unseen attack types (needs the row index first)
    "row-index": [[PY, "scripts/05_row_index.py"]],
    "strict": [[PY, "scripts/05_strict.py", "--profile", "full", "--schemes", "L8", "L34",
                "LM"],
               [PY, "scripts/05_strict.py", "--profile", "full", "--protocols", "--unseen",
                "all", "--schemes", "L8", "--models", "xgboost", "--seeds", "11"]],
    "split-ceilings": [[PY, "scripts/05_split_ceilings.py", "--profile", "full"]],
    # Steps 6-8
    "imbalance": [[PY, "scripts/06_imbalance.py", "--profile", "full"]],
    "neural": [[PY, "scripts/06_neural.py", "--profile", "full"]],
    "analysis": [[PY, "scripts/07_analysis.py", "--profile", "full", "--part", "all"]],
    "early": [[PY, "scripts/08_early_detection.py", "--profile", "full"]],
    # Figures, tables and the table of every reported number, from the saved results
    "figures": [[PY, "scripts/09_figures.py", "--profile", "full"]],
    "tables": [[PY, "scripts/09_tables.py", "--profile", "full"]],
    "numbers": [[PY, "scripts/10_numbers.py", "--profile", "full"]],
    "results": [[PY, "scripts/05_split_ceilings.py", "--profile", "full"],
                [PY, "scripts/09_figures.py", "--profile", "full"],
                [PY, "scripts/09_tables.py", "--profile", "full"],
                [PY, "scripts/10_numbers.py", "--profile", "full"]],
    # Model bundle for the dashboard (models/dashboard/)
    "export-model": [[PY, "scripts/13_export_model.py", "--profile", "full"]],
    "lint": [[PY, "-m", "ruff", "check", "."], [PY, "-m", "mypy"]],
    "test": [[PY, "-m", "pytest"]],
}

PLANNED: dict[str, str] = {}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "--list"}:
        print(__doc__)
        print("Available:", ", ".join(TASKS))
        print("Planned:  ", ", ".join(f"{k} ({v})" for k, v in PLANNED.items()))
        return 0
    task = argv[0]
    if task in PLANNED:
        print(f"Task '{task}' is not built yet; it arrives in {PLANNED[task]}.")
        return 1
    if task not in TASKS:
        print(f"Unknown task '{task}'. Run: python run.py --list")
        return 2
    for cmd in TASKS[task]:
        print(">>", " ".join(Path(c).name if c == PY else c for c in cmd), flush=True)
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
