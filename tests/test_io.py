"""Tests for checksum, zip and source-table helpers."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from iotml.io import list_zip_members, parse_source_files, sha256_file, sha256_zip_member

REPORT_SAMPLE = """Some header
Rows per source file
------------------------------------------------------------------------
      3218  CSV/Backdoor_Malware/Backdoor_Malware.pcap.csv
    272026  CSV/Benign_Final/BenignTraffic.pcap.csv
        12  CSV/XSS/XSS.pcap.csv
"""


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    data = b"0123456789" * 100_000
    path = tmp_path / "a.bin"
    path.write_bytes(data)
    assert sha256_file(path, chunk_mb=1) == hashlib.sha256(data).hexdigest()


def test_zip_member_hash_and_listing(tmp_path: Path) -> None:
    data = b"a,b\n1,2\n"
    zpath = tmp_path / "t.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dir/", b"")
        zf.writestr("dir/x.csv", data)
    members = list_zip_members(zpath)
    assert [m.name for m in members] == ["dir/x.csv"]
    assert members[0].file_size == len(data)
    assert sha256_zip_member(zpath, "dir/x.csv") == hashlib.sha256(data).hexdigest()


def test_parse_source_files_keeps_order_and_counts(tmp_path: Path) -> None:
    report = tmp_path / "merge_report.txt"
    report.write_text(REPORT_SAMPLE, encoding="utf-8")
    files = parse_source_files(report)
    assert [f.rows for f in files] == [3218, 272026, 12]
    assert files[1].folder == "Benign_Final"


def test_parse_source_files_requires_section(tmp_path: Path) -> None:
    report = tmp_path / "merge_report.txt"
    report.write_text("no table here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_source_files(report)
