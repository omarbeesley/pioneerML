"""Run an explicitly configured, provenance-recorded inference job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Pinned JSON profile in the release configs/ directory")
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="Mixed/resliced event parquet files, in processing order")
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty output directory")
    parser.add_argument("--max-events", type=int, help="Limit for a quick check; not a full benchmark")
    parser.add_argument("--batch-size", type=int, help="Override recorded batch size; changes tail dropout replay")
    parser.add_argument("--seed", type=int, help="Override the profile's random seed")
    parser.add_argument("--dump-atar-hits", action="store_true",
                        help="Main model: also write per-hit tables for classification and geometry analysis")
    parser.add_argument("--tags", nargs="+",
                        help="Main model: one output filename prefix per input (otherwise generated automatically)")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema_version") != 1 or cfg.get("model") not in {"main", "tail"}:
        parser.error("Unsupported profile schema or model family")
    if set(cfg.get("model_kwargs", {})) != {"dropout"}:
        parser.error("This release supports the pinned architecture and a dropout setting only")
    expected_class = "PURITYModel" if cfg["model"] == "main" else "PURITYTailModel"
    if cfg.get("model_class") != expected_class:
        parser.error("Model class does not match the selected family")
    if cfg["model"] != "main" and (args.dump_atar_hits or args.tags is not None):
        parser.error("--dump-atar-hits and --tags apply only to main-model inference")
    if args.tags is not None:
        if len(args.tags) != len(args.data) or len(set(args.tags)) != len(args.tags):
            parser.error("--tags must give one unique prefix per input file")
        if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) is None for tag in args.tags):
            parser.error("Tags must start with a letter or digit and contain only letters, digits, _, . or -")
    checkpoint = (cfg_path.parent.parent / cfg["checkpoint"]).resolve()
    inputs = [p.resolve() for p in args.data]
    for path in [checkpoint, *inputs]:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    actual_hash = sha256(checkpoint)
    if actual_hash != cfg["checkpoint_sha256"]:
        parser.error(f"Checkpoint checksum does not match the pinned profile: {checkpoint}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output directory must be empty; use a new directory to preserve run provenance")
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")

    inference = dict(cfg["inference"])
    if args.batch_size is not None:
        inference["batch_size"] = args.batch_size
    if args.seed is not None:
        inference["seed"] = args.seed
    if inference["batch_size"] <= 0 or inference["max_hits"] <= 0:
        parser.error("Batch size and hit limit must be positive")
    if not 0 <= inference["seed"] < 2**32:
        parser.error("Seed must be an integer in [0, 2**32)")
    if inference["mc_passes"] != 1:
        parser.error("The release profiles support the historical single-pass inference only")
    if inference["mode"] != "train":
        parser.error("These historical release profiles require inference mode 'train'")

    # All physics switches are set before model imports; inherited shell settings
    # cannot silently select a different architecture or acceptance convention.
    defaults = {"PURITY_GATE_MODE": "3d", "PURITY_BEST_XY_BETA": "0.5",
                "PURITY_ROLE_FILM": "0", "PURITY_ENERGY_TIME_GATE_NS": "0"}
    environment = defaults | {k: str(v) for k, v in cfg.get("environment", {}).items()}
    environment["PURITY_MAX_HITS"] = str(inference["max_hits"])
    os.environ.update(environment)
    import numpy as np
    import torch
    random.seed(inference["seed"])
    np.random.seed(inference["seed"])
    torch.manual_seed(inference["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference["seed"])

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "profile": cfg, "profile_path": str(cfg_path), "checkpoint_sha256": actual_hash,
              "effective_environment": environment, "effective_inference": inference,
              "max_events": args.max_events, "dump_atar_hits": args.dump_atar_hits,
              "output_tags": args.tags, "python": sys.version,
              "device": "cuda" if torch.cuda.is_available() else "cpu",
              "dependencies": {name: metadata.version(name) for name in
                               ["torch", "torch-geometric", "numpy", "pandas", "pyarrow",
                                "scipy", "matplotlib", "tqdm"]},
              "source_sha256": {str(p.relative_to(Path(__file__).parent)): sha256(p)
                                for p in sorted(Path(__file__).parent.rglob("*.py"))},
              "inputs": [{"path": str(p), "bytes": p.stat().st_size,
                          "sha256": sha256(p)} for p in inputs]}
    record_path = output / "run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    dropout = str(cfg["model_kwargs"]["dropout"])
    common = ["--checkpoint", str(checkpoint), "--dropout", dropout,
              "--batch_size", str(inference["batch_size"])]
    if args.max_events is not None:
        common += ["--max_events", str(args.max_events)]
    if cfg["model"] == "main":
        module = "purity.benchmark"
        # Explicit, unique names also work when two inputs share a parent folder.
        tags = args.tags or [f"sample_{i:03d}_{p.stem}" for i, p in enumerate(inputs)]
        record["output_tags"] = tags
        cli = common + ["--mc_passes", str(inference["mc_passes"]),
                        "--eval_path", *map(str, inputs), "--tag", *tags,
                        "--output_dir", str(output)]
        if args.dump_atar_hits:
            cli += ["--dump_atar_hits"]
    else:
        module = "purity.eval_tail"
        cli = common + ["--train_mode", "--max_hits", str(inference["max_hits"]),
                        "--infer_only", "--data", *map(str, inputs), "--output_dir", str(output)]
    record["command"] = [sys.executable, "-m", module, *cli]
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    previous_argv = sys.argv
    try:
        sys.argv = [module, *cli]
        runpy.run_module(module, run_name="__main__")
        import pyarrow.parquet as pq
        expected = ([output / f"{tag}_events.parquet" for tag in tags]
                    if cfg["model"] == "main" else [output / "predictions.parquet"])
        rows = {p.name: pq.ParquetFile(p).metadata.num_rows for p in expected}
        if any(count <= 0 for count in rows.values()):
            raise ValueError("Inference produced no events for at least one input sample")
        record["output_event_rows"] = rows
        if args.dump_atar_hits:
            hit_rows = {f"{tag}_atar_hits.parquet": pq.ParquetFile(output / f"{tag}_atar_hits.parquet").metadata.num_rows for tag in tags}
            if any(count <= 0 for count in hit_rows.values()):
                raise ValueError("Requested ATAR hit dump is empty for at least one sample")
            record["output_hit_rows"] = hit_rows
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.argv = previous_argv
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
