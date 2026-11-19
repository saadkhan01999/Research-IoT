"""Step 2: build the columnar cache data/processed/ciciot2023.parquet from the raw zip.

Usage: python scripts/02_build_parquet.py

Also writes data/processed/source_files.csv (one row per source pcap-CSV file),
data/processed/label_codes.json, and data/processed/cache_manifest.json (row counts and
checksums), and refuses to run if the raw inputs do not match their saved checksums.
"""

from __future__ import annotations

import csv
import json
import time

from iotml.config import load_config, load_labels, project_path
from iotml.data import LabelCodes, build_parquet, source_table
from iotml.io import sha256_file
from iotml.logs import get_logger


def raw_checksums_ok(config: dict, log) -> bool:
    saved = project_path(config["paths"]["checksums"]) / "raw_sha256.csv"
    with saved.open(encoding="utf-8") as fh:
        expected = {r["item"]: r["sha256"] for r in csv.DictReader(fh) if r["kind"] == "file"}
    for item, digest in expected.items():
        if sha256_file(project_path(item)) != digest:
            log.error("Checksum mismatch for %s", item)
            return False
    return True


def main() -> None:
    config = load_config()
    paths = config["paths"]
    log = get_logger("02_build_parquet", project_path(paths["logs"]))
    if not raw_checksums_ok(config, log):
        raise SystemExit("Raw data changed since Step 0; stop and investigate.")

    label_to_category = load_labels()
    codes = LabelCodes.from_mapping(label_to_category)
    folder_label = {"Benign_Final": "BenignTraffic"}
    sources = source_table(project_path(paths["merge_report"]), folder_label)
    unknown = set(sources["label"]) - set(codes.labels)
    if unknown:
        raise SystemExit(f"Folders without a label in labels.yaml: {sorted(unknown)}")
    sources["category"] = sources["label"].map(label_to_category)

    out_dir = project_path(paths["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sources.to_csv(out_dir / "source_files.csv", index=False)
    (out_dir / "label_codes.json").write_text(codes.to_json(), encoding="utf-8")

    t0 = time.time()
    parquet = out_dir / "ciciot2023.parquet"
    log.info("Building %s from %s ...", parquet.name, paths["raw_zip"])
    stats = build_parquet(project_path(paths["raw_zip"]), paths["raw_member"], sources, codes,
                          parquet, log)
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "minutes": round((time.time() - t0) / 60, 1),
        **stats,
        "source_files": len(sources),
        "parquet_bytes": parquet.stat().st_size,
        "parquet_sha256": sha256_file(parquet),
        "label_check": "every row's label equals the label of its source file",
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Done: %s", manifest)


if __name__ == "__main__":
    main()
