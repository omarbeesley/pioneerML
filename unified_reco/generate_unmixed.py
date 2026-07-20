"""
Generate unmixed PURITY parquet datasets, one per (channel × split):
  pie/train, pie/val, pie/eval
  michel/train, michel/val, michel/eval
  mudif/train, mudif/val, mudif/eval   (muon decay-in-flight: pi-DAR -> mu-DIF -> e)
  pidif/train, pidif/val, pidif/eval   (pion decay-in-flight: pi-DIF -> mu-DAR -> e)

Each is produced by invoking root_to_parquet.py on the corresponding
sub-directory of ROOT files at:
  {ROOT_BASE}/{pie,michel,mudif,pidif}/{train,val,eval}/*.root

root_to_parquet keeps muDIF events automatically (its skim only drops
*pion*-DIF, kPidif); muDIF events carry kMudif and a muon_decay_ke > 0. The
piDIF channel is converted with --keep_pidif (added by run_job for that channel
only) so pion-DIF survives; those events carry kPidif and a pion_decay_ke > 0.

Usage:
    python generate_unmixed.py                                # all jobs
    python generate_unmixed.py --only mudif_train             # one job
    python generate_unmixed.py --max_train 100000 --max_val 20000 --max_eval 100000
"""
import argparse
import os
import subprocess
import sys

DATA_DIR = "/data/nvme0/prod_ml_data/unmixed_parquets/"
ROOT_BASE = "/data/nvme0/root_files/"
ROOT_TO_PARQUET = "/home/obeesley/pioneerML/unified_reco/root_to_parquet.py"

JOBS = {
    "pie_train":    dict(channel="pie",    split="train", out_name="unmixed_pie_train.parquet"),
    "pie_val":      dict(channel="pie",    split="val",   out_name="unmixed_pie_val.parquet"),
    "pie_eval":     dict(channel="pie",    split="eval",  out_name="unmixed_pie_eval.parquet"),
    "michel_train": dict(channel="michel", split="train", out_name="unmixed_michel_train.parquet"),
    "michel_val":   dict(channel="michel", split="val",   out_name="unmixed_michel_val.parquet"),
    "michel_eval":  dict(channel="michel", split="eval",  out_name="unmixed_michel_eval.parquet"),
    "mudif_train":  dict(channel="mudif",  split="train", out_name="unmixed_mudif_train.parquet"),
    "mudif_val":    dict(channel="mudif",  split="val",   out_name="unmixed_mudif_val.parquet"),
    "mudif_eval":   dict(channel="mudif",  split="eval",  out_name="unmixed_mudif_eval.parquet"),
    "pidif_train":  dict(channel="pidif",  split="train", out_name="unmixed_pidif_train.parquet"),
    "pidif_val":    dict(channel="pidif",  split="val",   out_name="unmixed_pidif_val.parquet"),
    "pidif_eval":   dict(channel="pidif",  split="eval",  out_name="unmixed_pidif_eval.parquet"),
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
        # Per-job seed: decorrelate the 12 jobs' smearing + file shuffle while
        # staying reproducible. --seed now also seeds the per-hit smearing RNG
        # (passed even without --shuffle_files).
        job_seed = (args.seed + sum(ord(c) for c in name)) & 0xFFFFFFFF
        cmd.extend(["--seed", str(job_seed)])
    # The piDIF channel is the ONLY one converted with kPidif kept; every other
    # channel keeps the default skim (drops pion-DIF contamination).
    if spec["channel"] == "pidif":
        cmd.append("--keep_pidif")

    if args.nprocs and args.nprocs > 1:
        cmd.extend(["--nprocs", str(args.nprocs)])

    print(f"\n=== [{name}] {spec['channel']}/{spec['split']}  "
          f"max_events={max_events}  ->  {output}", flush=True)
    print(f"    {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"    wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(JOBS.keys()), default=None,
                        help="Run a single job instead of all six.")
    parser.add_argument("--max_train", type=int, default=10000000,
                        help="Max events for the train splits.")
    parser.add_argument("--max_val",   type=int, default=500000,
                        help="Max events for the val splits.")
    parser.add_argument("--max_eval",  type=int, default=1000000,
                        help="Max events for the eval splits.")
    parser.add_argument("--output_dir", type=str, default=DATA_DIR,
                        help="Directory to write all 6 parquet files.")
    parser.add_argument("--shuffle_files", action="store_true", default=True,
                        help="Pass --shuffle_files to root_to_parquet.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed used with --shuffle_files.")
    parser.add_argument("--nprocs", type=int, default=1,
                        help="Worker processes per job, passed through to root_to_parquet "
                             "--nprocs (file-sharded parallel conversion). Default 1.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = [args.only] if args.only else list(JOBS.keys())
    for name in jobs:
        run_job(name, JOBS[name], args)


if __name__ == "__main__":
    main()
