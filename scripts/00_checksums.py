"""Step 0: record (or verify) SHA-256 checksums of every raw input file.

Usage:
    python scripts/00_checksums.py            # write data/checksums/raw_sha256.csv
    python scripts/00_checksums.py --verify   # recompute and compare with the saved file

Besides the zip and report files, it hashes the decompressed CSV inside the zip, so the
check still holds if the archive is ever re-compressed. It also checks that the
source-file table in merge_report.txt adds up to the expected number of rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from iotml.config import PROJECT_ROOT, load_config, project_path
from iotml.io import list_zip_members, parse_source_files, sha256_file, sha256_zip_member
from iotml.logs import get_logger

FIELDS = ["item", "kind", "size_bytes", "crc32", "sha256", "recorded_at"]


def compute_rows(config: dict, log) -> list[dict[str, str]]:
    """Hash the raw zip, the report, and each member of the zip."""
    paths = config["paths"]
    chunk = int(config["runtime"]["hash_chunk_mb"])
    zip_path = project_path(paths["raw_zip"])
    report = project_path(paths["merge_report"])
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for path in (zip_path, report):
        if not path.is_file():
            raise FileNotFoundError(f"Raw input missing: {path}. See README, section Data.")
        log.info("Hashing %s ...", path.name)
        rows.append({"item": path.relative_to(PROJECT_ROOT).as_posix(), "kind": "file",
                     "size_bytes": str(path.stat().st_size), "crc32": "",
                     "sha256": sha256_file(path, chunk), "recorded_at": now})
    for member in list_zip_members(zip_path):
        log.info("Hashing zip member %s (%.2f GB decompressed) ...", member.name,
                 member.file_size / 1e9)
        rows.append({"item": f"{zip_path.name}::{member.name}", "kind": "zip_member",
                     "size_bytes": str(member.file_size), "crc32": member.crc32,
                     "sha256": sha256_zip_member(zip_path, member.name, chunk),
                     "recorded_at": now})
    return rows


def check_source_table(config: dict, log) -> None:
    """The report's per-file table must cover all source files and all rows."""
    files = parse_source_files(project_path(config["paths"]["merge_report"]))
    total = sum(f.rows for f in files)
    expected_files = int(config["data"]["n_source_files"])
    expected_rows = int(config["data"]["expected_rows"])
    log.info("Source-file table: %d files, %d rows", len(files), total)
    if len(files) != expected_files or total != expected_rows:
        raise ValueError(f"Source-file table mismatch: {len(files)} files / {total} rows, "
                         f"expected {expected_files} / {expected_rows}")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def verify(rows: list[dict[str, str]], saved: Path, log) -> bool:
    """Compare fresh hashes with the saved ones; log every difference."""
    with saved.open(encoding="utf-8") as fh:
        old = {r["item"]: r["sha256"] for r in csv.DictReader(fh)}
    ok = True
    for row in rows:
        if old.get(row["item"]) != row["sha256"]:
            ok = False
            log.error("CHANGED or NEW: %s", row["item"])
    for item in set(old) - {r["item"] for r in rows}:
        ok = False
        log.error("MISSING: %s", item)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="compare with saved checksums")
    args = parser.parse_args()

    config = load_config()
    log = get_logger("00_checksums", project_path(config["paths"]["logs"]))
    check_source_table(config, log)
    rows = compute_rows(config, log)
    saved = project_path(config["paths"]["checksums"]) / "raw_sha256.csv"

    if args.verify:
        if not saved.is_file():
            sys.exit(f"No saved checksums at {saved}; run without --verify first.")
        if not verify(rows, saved, log):
            sys.exit("Raw data does not match the saved checksums. Do not run experiments.")
        log.info("All raw inputs match %s", saved.relative_to(PROJECT_ROOT))
    else:
        write_csv(rows, saved)
        log.info("Saved %s", saved.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
