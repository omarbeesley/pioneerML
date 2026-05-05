"""
Generate six unmixed PURITY parquet datasets, one per (channel × split):
  pie/train, pie/val, pie/eval
  michel/train, michel/val, michel/eval

Each is produced by invoking root_to_parquet.py on the corresponding
sub-directory of ROOT files at:
  /mnt/e/global_ai_recon/{pie,michel}/{train,val,eval}/*.root

Usage:
    python generate_unmixed.py                                # all six
    python generate_unmixed.py --only pie_train               # one of the six
    python generate_unmixed.py --max_train 100000 --max_val 20000 --max_eval 100000
"""
import argparse
import os
import subprocess
import sys

DATA_DIR = "/mnt/c/Users/obbee/research/notebooks/ML/data/purity"
ROOT_BASE = "/mnt/e/global_ai_recon"
ROOT_TO_PARQUET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "root_to_parquet.py")

JOBS = {
    "pie_train":    dict(channel="pie",    split="train", out_name="unmixed_pie_train.parquet"),
    "pie_val":      dict(channel="pie",    split="val",   out_name="unmixed_pie_val.parquet"),
    "pie_eval":     dict(channel="pie",    split="eval",  out_name="unmixed_pie_eval.parquet"),
    "michel_train": dict(channel="michel", split="train", out_name="unmixed_michel_train.parquet"),
    "michel_val":   dict(channel="michel", split="val",   out_name="unmixed_michel_val.parquet"),
    "michel_eval":  dict(channel="michel", split="eval",  out_name="unmixed_michel_eval.parquet"),
}


def run_job(name, spec, args):
    max_events = {
        "train": args.max_train,
        "val":   args.max_val,
        "eval":  args.max_eval,
    }[spec["split"]]

    input_glob = os.path.join(ROOT_BASE, spec["channel"], spec["split"], "*.root")
    output     = os.path.join(args.output_dir, spec["out_name"])

    cmd = [
        sys.executable, ROOT_TO_PARQUET,
        "--input",  input_glob,
        "--output", output,
        "--max_events", str(max_events),
    ]
    if args.shuffle_files:
        cmd.append("--shuffle_files")
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])

    print(f"\n=== [{name}] {spec['channel']}/{spec['split']}  "
          f"max_events={max_events}  ->  {output}", flush=True)
    print(f"    {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"    wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(JOBS.keys()), default=None,
                        help="Run a single job instead of all six.")
    parser.add_argument("--max_train", type=int, default=200000,
                        help="Max events for the train splits.")
    parser.add_argument("--max_val",   type=int, default=20000,
                        help="Max events for the val splits.")
    parser.add_argument("--max_eval",  type=int, default=100000,
                        help="Max events for the eval splits.")
    parser.add_argument("--output_dir", type=str, default=DATA_DIR,
                        help="Directory to write all 6 parquet files.")
    parser.add_argument("--shuffle_files", action="store_true", default=True,
                        help="Pass --shuffle_files to root_to_parquet.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed used with --shuffle_files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = [args.only] if args.only else list(JOBS.keys())
    for name in jobs:
        run_job(name, JOBS[name], args)


if __name__ == "__main__":
    main()
