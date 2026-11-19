"""Raw data to a columnar cache, plus readers for the cache.

The merged CSV is streamed out of the zip, never fully loaded. Features are kept as float64,
so the audit sees the exact values of the raw file. Each row gets the id of its source
pcap-CSV file and its position in that file, rebuilt from merge_report.txt.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from iotml.io import SourceFile, parse_source_files

LABEL_COL, CATEGORY_COL = "Label", "Category"
FILE_COL, POS_COL = "file_id", "row_in_file"
LABEL_CODE, CATEGORY_CODE = "label", "category"


@dataclass(frozen=True)
class LabelCodes:
    """Integer codes for the 34 labels and 8 categories (sorted names, stable across runs)."""

    labels: list[str]
    categories: list[str]
    category_of: dict[str, str]

    @classmethod
    def from_mapping(cls, label_to_category: dict[str, str]) -> LabelCodes:
        return cls(sorted(label_to_category), sorted(set(label_to_category.values())),
                   dict(label_to_category))

    def label_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.labels)}

    def category_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.categories)}

    def to_json(self) -> str:
        return json.dumps({"labels": self.labels, "categories": self.categories,
                           "category_of": self.category_of}, indent=1)


def read_header(zip_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        return io.TextIOWrapper(fh, encoding="ascii").readline().rstrip("\r\n").split(",")


def source_table(report: Path, folder_label: dict[str, str]) -> pd.DataFrame:
    """One row per source file, in row order: id, path, label, rows, first and last row."""
    files: list[SourceFile] = parse_source_files(report)
    table = pd.DataFrame({"file_id": np.arange(len(files), dtype=np.int16),
                          "path": [f.path for f in files],
                          "folder": [f.folder for f in files],
                          "rows": np.array([f.rows for f in files], dtype=np.int64)})
    table["label"] = table["folder"].map(lambda f: folder_label.get(f, f))
    table["start"] = np.concatenate([[0], np.cumsum(table["rows"].to_numpy())[:-1]])
    table["end"] = table["start"] + table["rows"]
    return table


def stream_csv(zip_path: Path, member: str, features: list[str],
               block_mb: int = 32) -> Iterator[pa.RecordBatch]:
    """Record batches of the merged CSV in file order; features parsed as float64."""
    types = {name: pa.float64() for name in features}
    types.update({LABEL_COL: pa.string(), CATEGORY_COL: pa.string()})
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        reader = pacsv.open_csv(
            fh, read_options=pacsv.ReadOptions(block_size=block_mb << 20, use_threads=True),
            convert_options=pacsv.ConvertOptions(column_types=types))
        yield from reader


def build_parquet(zip_path: Path, member: str, sources: pd.DataFrame, codes: LabelCodes,
                  out_path: Path, log, block_mb: int = 32) -> dict[str, int]:
    """Write the cache and check that every row's label equals its source file's label."""
    features = [c for c in read_header(zip_path, member) if c not in (LABEL_COL, CATEGORY_COL)]
    label_idx, cat_idx = codes.label_index(), codes.category_index()
    ends = sources["end"].to_numpy()
    starts = sources["start"].to_numpy()
    file_label = sources["label"].map(label_idx).to_numpy()
    schema = pa.schema([(f, pa.float64()) for f in features] + [
        (LABEL_CODE, pa.int8()), (CATEGORY_CODE, pa.int8()),
        (FILE_COL, pa.int16()), (POS_COL, pa.int32())])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial")
    written = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3) as writer:
        for batch in stream_csv(zip_path, member, features, block_mb):
            n = batch.num_rows
            idx = np.arange(written, written + n, dtype=np.int64)
            fid = np.searchsorted(ends, idx, side="right")
            labels = pd.Series(batch.column(LABEL_COL).to_numpy(zero_copy_only=False))
            lab = labels.map(label_idx)
            cat = pd.Series(batch.column(CATEGORY_COL).to_numpy(zero_copy_only=False)).map(cat_idx)
            if lab.isna().any() or cat.isna().any():
                raise ValueError(f"Unknown label/category near row {written}")
            lab_np = lab.to_numpy(dtype=np.int8)
            expected_cat = labels.map(codes.category_of).map(cat_idx).to_numpy()
            if not np.array_equal(cat.to_numpy(), expected_cat):
                raise ValueError(f"Category does not match labels.yaml near row {written}")
            if not np.array_equal(lab_np, file_label[fid]):
                bad = int(idx[np.argmax(lab_np != file_label[fid])])
                raise ValueError(f"Row {bad}: label differs from its source file's label; "
                                 "the source-file table does not line up with the data")
            arrays = [batch.column(f) for f in features] + [
                pa.array(lab_np), pa.array(cat.to_numpy(dtype=np.int8)),
                pa.array(fid.astype(np.int16)), pa.array((idx - starts[fid]).astype(np.int32))]
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
            written += n
            if written // 5_000_000 != (written - n) // 5_000_000:
                log.info("  %d rows written", written)
    if written != int(sources["rows"].sum()):
        tmp.unlink()
        raise ValueError(f"Wrote {written} rows, source table says {int(sources['rows'].sum())}")
    tmp.replace(out_path)
    return {"rows": written, "features": len(features)}


def feature_columns(parquet: Path) -> list[str]:
    names = pq.read_schema(parquet).names
    return [n for n in names if n not in (LABEL_CODE, CATEGORY_CODE, FILE_COL, POS_COL)]


def iter_batches(parquet: Path, columns: list[str] | None = None,
                 batch_rows: int = 1_000_000) -> Iterator[pd.DataFrame]:
    """The cache in row order, as pandas frames of at most `batch_rows` rows."""
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()
