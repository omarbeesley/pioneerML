"""Mix one michel shard (a group of gen_scratch chunk-parquets) into a pimu benchmark shard.

For the 10^7 michel run: the full pool (~44 GB) will not fit in RAM, so we shard.  gen_scratch
ships one parquet per (task,chunk); this concatenates all chunks of ONE gen task into a
~200k-event pool on node-local scratch, then runs the PileupMixer on it with the unbiased
BENCHMARK_OPTS (physical pileup, accidental labels).  1 mix task <-> 1 gen task.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from pileup_mixer import PileupMixer

ap = argparse.ArgumentParser()
ap.add_argument("--gen_job", required=True)
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--michel_dir", required=True)
ap.add_argument("--pie", required=True)
ap.add_argument("--lut_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scratch", default="/scratch")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
np.random.seed(args.seed & 0xFFFFFFFF)

files = sorted(glob.glob(f"{args.michel_dir}/michel_{args.gen_job}_t{args.task:03d}_c*.parquet"))
assert files, f"no michel shards for gen_job={args.gen_job} task={args.task} in {args.michel_dir}"
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
n = len(df)
tmp = os.path.join(args.scratch, f"michel_pool_{args.gen_job}_{args.task}.parquet")
os.makedirs(args.scratch, exist_ok=True)
df.to_parquet(tmp)
del df

mx = PileupMixer(michel_path=tmp, pie_path=args.pie, lut_dir=args.lut_dir)
mixed = mx.generate_batch(n, mode="michel",
                          biased_fraction=0.0, biased_sigma=100.0, cal_only_fraction=0.0,
                          radio_rate=2e7, enforce_window=False)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
mixed.to_parquet(args.out)
os.remove(tmp)
print(f"[mix_shard t{args.task:03d}] {n} michel ({len(files)} chunks) -> {len(mixed)} mixed -> {args.out}",
      flush=True)
