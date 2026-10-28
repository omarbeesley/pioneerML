"""Assemble one source-pool shard from an ordered input inventory.

Generalizes the final michel_shard_concat.py rule: sorted files at positions
index * files_per_pool through (index + 1) * files_per_pool, exclusive.
"""

import argparse
from pathlib import Path

from ..final_analysis._shared import resolve_files
from ..final_analysis.combine_events import combine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="Unmixed files or quoted globs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int, help="Zero-based pool-shard index")
    parser.add_argument("--files-per-pool", type=int, default=14)
    args = parser.parse_args()
    if args.index < 0 or args.files_per_pool < 1:
        parser.error("--index must be nonnegative and --files-per-pool positive")
    paths, _ = resolve_files(args.input, 0)
    selected = paths[args.index * args.files_per_pool:(args.index + 1) * args.files_per_pool]
    if not selected:
        parser.error("Requested shard selects no source files")
    combine(selected, args.output, batch_size=50_000, compression="snappy")


if __name__ == "__main__":
    main()
