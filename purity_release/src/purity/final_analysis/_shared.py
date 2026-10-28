"""Explicit inputs and strict schema validation for archived table recipes."""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pie", required=True, nargs="+", help="Files or quoted globs")
    parser.add_argument("--michel", required=True, nargs="+", help="Files or quoted globs")
    parser.add_argument("--output", required=True, type=Path, help="Result JSON path")
    parser.add_argument(
        "--min-shard-bytes", type=int, default=1_000_001,
        help="Minimum file size, inclusive; default preserves the source >1 MB filter. "
             "Use 0 for small examples.",
    )
    args = parser.parse_args()
    if args.min_shard_bytes < 0:
        parser.error("--min-shard-bytes must be nonnegative")
    manifest_path = args.output.with_name(args.output.stem + ".inputs.json")
    for path in (args.output, manifest_path):
        if path.exists():
            parser.error(f"Output already exists; choose a new output path: {path}")
    return args


def resolve_files(patterns, minimum_bytes):
    paths = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No input files match {pattern!r}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                raise ValueError(f"Input is not a file: {path}")
            paths.add(path)
    included = sorted(p for p in paths if p.stat().st_size >= minimum_bytes)
    excluded = sorted(paths - set(included))
    if not included:
        raise ValueError("No input files pass --min-shard-bytes")
    return included, excluded


def iter_arrays(paths, columns, purify_michel=False):
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        table = parquet.read(columns=columns)
        data = {key: table.column(key).to_numpy(zero_copy_only=False) for key in columns}
        if purify_michel:
            event_type = data["truth_event_type"].astype("int64")
            keep = ((event_type & 1) == 0) & ((event_type & 0x200) == 0)
            keep &= data["truth_positron_energy"] <= 55
            data = {key: value[keep] for key, value in data.items()}
        yield data


def _file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size}


def write_result(args, result, files, exclusions, recipe):
    manifest_path = args.output.with_name(args.output.stem + ".inputs.json")
    for path in (args.output, manifest_path):
        if path.exists():
            raise FileExistsError(f"Output already exists; choose a new output path: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    # This records the selected current files, not an invented historical shard list.
    manifest = {
        "recipe": recipe,
        "minimum_shard_bytes": args.min_shard_bytes,
        "inputs": {channel: [_file_record(p) for p in paths]
                   for channel, paths in files.items()},
        "excluded_by_size": {channel: [_file_record(p) for p in paths]
                             for channel, paths in exclusions.items()},
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name(f"{recipe}.py"))
        },
        "data_contents_hashed": False,
    }
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output} and {manifest_path}")
