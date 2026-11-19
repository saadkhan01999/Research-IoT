"""File-level input helpers: checksums, zip members and the source-file table."""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

MB = 1024 * 1024
_SOURCE_ROW = re.compile(r"^\s*(\d+)\s{2}(CSV/\S.*)$")
_SOURCE_HEADER = "Rows per source file"


@dataclass(frozen=True)
class ZipMember:
    """Directory entry of one file inside a zip archive."""

    name: str
    file_size: int
    compress_size: int
    crc32: str


@dataclass(frozen=True)
class SourceFile:
    """One original per-pcap CSV and the number of its rows kept in the merged file."""

    path: str
    rows: int

    @property
    def folder(self) -> str:
        """Class folder of the file, e.g. ``DDoS-ICMP_Flood``."""
        return self.path.split("/")[1]


def _read_blocks(handle: IO[bytes], chunk_mb: int) -> Iterator[bytes]:
    while block := handle.read(chunk_mb * MB):
        yield block


def sha256_stream(handle: IO[bytes], chunk_mb: int = 8) -> str:
    """SHA-256 of everything left in an open binary stream."""
    digest = hashlib.sha256()
    for block in _read_blocks(handle, chunk_mb):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_mb: int = 8) -> str:
    """SHA-256 of a file on disk."""
    with path.open("rb") as fh:
        return sha256_stream(fh, chunk_mb)


def sha256_zip_member(zip_path: Path, member: str, chunk_mb: int = 8) -> str:
    """SHA-256 of the decompressed bytes of one zip member."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        return sha256_stream(fh, chunk_mb)


def list_zip_members(zip_path: Path) -> list[ZipMember]:
    """File entries of a zip archive (folders are skipped)."""
    with zipfile.ZipFile(zip_path) as zf:
        return [
            ZipMember(i.filename, i.file_size, i.compress_size, f"{i.CRC:08x}")
            for i in zf.infolist()
            if not i.is_dir()
        ]


def parse_source_files(report_path: Path) -> list[SourceFile]:
    """Read the 'Rows per source file' table of merge_report.txt, in merged-row order.

    The merged CSV holds the rows of these files one after another in exactly this order,
    so the table gives the source file of every row.
    """
    lines = report_path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(_SOURCE_HEADER)
    except ValueError as exc:
        raise ValueError(f"No '{_SOURCE_HEADER}' section in {report_path}") from exc
    files = []
    for line in lines[start + 1 :]:
        match = _SOURCE_ROW.match(line)
        if match:
            files.append(SourceFile(path=match.group(2).strip(), rows=int(match.group(1))))
    if not files:
        raise ValueError(f"The '{_SOURCE_HEADER}' section in {report_path} is empty")
    return files
