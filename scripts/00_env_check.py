"""Step 0: record the machine and library versions in results/env/.

Usage: python scripts/00_env_check.py
"""

from __future__ import annotations

import json
import time

from iotml.config import PROJECT_ROOT, load_config, project_path
from iotml.envinfo import collect
from iotml.logs import get_logger

MIN_RAM_GB = 8
MIN_FREE_DISK_GB = 20


def main() -> None:
    config = load_config()
    log = get_logger("00_env_check", project_path(config["paths"]["logs"]))
    env = collect(PROJECT_ROOT)

    out_dir = project_path(config["paths"]["results"]) / "env"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"env_{time.strftime('%Y%m%d')}.json"
    out_file.write_text(json.dumps(env, indent=2), encoding="utf-8")

    log.info("CPU %s (%s cores / %s threads)", env["cpu"], env["cpu_physical_cores"],
             env["cpu_logical_cores"])
    log.info("RAM %.1f GB total, %.1f GB available", env["ram_total_gb"], env["ram_available_gb"])
    log.info("Free disk on project drive: %.1f GB", env["disk_free_gb_project_drive"])
    log.info("NVIDIA GPU driver found: %s", env["nvidia_gpu_driver"])
    missing = [name for name, ver in env["libraries"].items() if ver is None]
    if missing:
        log.info("Not installed yet (added in later steps when needed): %s", ", ".join(missing))
    if env["ram_total_gb"] < MIN_RAM_GB + 1:
        log.warning("Low RAM: all experiments must use per-class row caps and chunked reading.")
    if env["disk_free_gb_project_drive"] < MIN_FREE_DISK_GB:
        log.warning("Less than %d GB free on the project drive.", MIN_FREE_DISK_GB)
    log.info("Saved %s", out_file.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
