"""Stream Parquet shards into a frozen combined pool with an exact input manifest.

Port of build_fl2_combined.py with explicit paths, configurable batching, strict
schema errors, and no silently skipped shards. Also supports raw pool assembly.
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ._shared import resolve_files
from ..preprocessing._io import ensure_output, record_run


def combine(paths, output, batch_size=50_000, compression="zstd"):
    output = ensure_output(output)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not paths:
        raise ValueError("No input files to combine")
    schema = pq.ParquetFile(paths[0]).schema_arrow
    # Validate all input schemas before creating a combined file.
    for path in paths:
        current = pq.ParquetFile(path).schema_arrow
        if set(current.names) != set(schema.names):
            raise ValueError(f"Column set differs in {path}")
        ordered = pa.schema([current.field(name) for name in schema.names])
        if not ordered.equals(schema, check_metadata=False):
            raise ValueError(f"Column types differ in {path}; normalize schemas explicitly")
    rows = 0
    with pq.ParquetWriter(output, schema, compression=compression) as writer:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=schema.names):
                table = pa.Table.from_batches([batch]).cast(schema)
                writer.write_table(table)
                rows += table.num_rows
    record_run(output, paths, __file__, {
        "rows": rows, "batch_size": batch_size, "compression": compression,
        "schema_policy": "equal column sets/types; order normalized to first input; no skipped shards",
    })
    print(f"Combined {len(paths)} files and {rows:,} rows -> {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Files or quoted globs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--min-shard-bytes", type=int, default=0,
                        help="Set 1000001 to replay the historical >1 MB screen")
    args = parser.parse_args()
    if args.min_shard_bytes < 0:
        parser.error("--min-shard-bytes must be nonnegative")
    ensure_output(args.output)
    paths, excluded = resolve_files(args.input, args.min_shard_bytes)
    if excluded:
        print(f"Size filter excluded {len(excluded)} files")
    combine(paths, args.output, args.batch_size)


if __name__ == "__main__":
    main()
