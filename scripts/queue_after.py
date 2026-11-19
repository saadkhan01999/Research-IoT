"""Run commands after another job has finished, keeping the computer awake throughout.

Usage: python scripts/queue_after.py [--wait-file LOG --marker TEXT] [--poll 30] [--retries N]
                                      --cmd "python ..." [--cmd "python ..."] [--keep-going]

Waits until a line of LOG starts with TEXT (the earlier job's last line), then runs each
command in order from the project root. Stops at the first failing command unless
--keep-going is given. Only line starts count: a waiting queue writes the marker of the job it
waits for into its own log, and a chained queue must not mistake that line for the end.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from iotml.config import PROJECT_ROOT
from iotml.system import keep_awake


def finished(path: Path, marker: str) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return any(line.startswith(marker) for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wait-file", type=Path)      # omit both to start at once
    parser.add_argument("--marker", default="")
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--cmd", action="append", required=True)
    parser.add_argument("--keep-going", action="store_true")
    # A command that fails (for example the memory guard after its 60-minute wait) is run
    # again; the experiment scripts skip runs that are already saved, so a retry only does
    # what is left.
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--retry-wait", type=int, default=600, help="seconds before a retry")
    args = parser.parse_args()
    with keep_awake():
        if args.wait_file is not None:
            print(f"waiting for '{args.marker}' in {args.wait_file}", flush=True)
            while not finished(args.wait_file, args.marker):
                time.sleep(args.poll)
        status = 0
        for cmd in args.cmd:
            parts = shlex.split(cmd)
            if parts and parts[0] == "python":
                parts[0] = sys.executable
            for attempt in range(args.retries + 1):
                print(f"== {cmd}" + (f"  (attempt {attempt + 1})" if attempt else ""),
                      flush=True)
                status = subprocess.run(parts, cwd=PROJECT_ROOT).returncode
                print(f"exit={status} for: {cmd}", flush=True)
                if status == 0 or attempt == args.retries:
                    break
                time.sleep(args.retry_wait)
            if status != 0 and not args.keep_going:
                break
    return status


if __name__ == "__main__":
    sys.exit(main())
