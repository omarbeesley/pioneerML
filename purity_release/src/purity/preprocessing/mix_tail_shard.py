"""Mix one tail-training shard or a pure-class tail evaluation pool.

Port of the external mix_shard_tail.py that produced the tail10 training data.
Mixed mode draws 30% pion-to-positron, 25% muon decay in flight, 15% pion decay
in flight and 30% Michel triggers. Radiative muon decays (kMurad, 0x200) are removed
from the Michel and muon-DIF pools. The four pools are the converted simulation
chunks of one generation-task group per channel; distinct groups keep training,
validation and evaluation disjoint.

The historical shards were mixed by the July 2026 pileup mixer; this command uses
the release mixer, whose accidental sampling was revised afterwards. A new run
therefore follows the recipe but is not the archived realization. See
docs/training.md.

    python -m purity.preprocessing.mix_tail_shard --task 0 --mode mixed \\
        --mgrp 0 --pgrp 0 --ugrp 0 --dgrp 0 --seed 7000 \\
        --pool_dir /data/parquet --lut_dir assets/luts \\
        --scratch /scratch/$USER --out outputs/tail10/train_000.parquet
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from purity.preprocessing._io import ensure_output, record_run
from purity.preprocessing.pileup_mixer import PileupMixer

KMURAD = 0x200
# Generation jobs of the converted pools used for tail10.
HISTORICAL_JOBS = {"michel": 1595422, "pie": 1596343, "mudif": 1596402, "pidif": 1596403}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--mode", default="mixed", choices=["mixed", "pie", "michel", "mudif", "pidif"])
    ap.add_argument("--mgrp", type=int, required=True, help="Michel generation-task group")
    ap.add_argument("--pgrp", type=int, required=True, help="Pion-to-positron group")
    ap.add_argument("--ugrp", type=int, required=True, help="Muon-DIF group")
    ap.add_argument("--dgrp", type=int, required=True, help="Pion-DIF group")
    ap.add_argument("--nev", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_dir", required=True,
                    help="Directory holding <channel>/<channel>_<job>_t<group>_c*.parquet")
    ap.add_argument("--lut_dir", required=True)
    for ch, job in HISTORICAL_JOBS.items():
        ap.add_argument(f"--{ch}_job", type=int, default=job)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/scratch")
    a = ap.parse_args()

    out = ensure_output(a.out)
    np.random.seed(a.seed & 0xFFFFFFFF)
    os.makedirs(a.scratch, exist_ok=True)
    inputs = []

    def pool(ch, grp, rmd_filter):
        job = getattr(a, f"{ch}_job")
        fs = sorted(glob.glob(os.path.join(a.pool_dir, ch, f"{ch}_{job}_t{grp:03d}_c*.parquet")))
        assert fs, f"no chunks {ch} grp {grp}"
        inputs.extend(fs)
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        if rmd_filter and "event_type" in df.columns:
            df = df[(df.event_type.astype("int64") & KMURAD) == 0].reset_index(drop=True)
        t = os.path.join(a.scratch, f"tail_{ch}_{a.task}.parquet")
        df.to_parquet(t)
        return t

    mp = pool("michel", a.mgrp, True)
    pp = pool("pie", a.pgrp, False)
    up = pool("mudif", a.ugrp, True)
    dp = pool("pidif", a.dgrp, False)
    mx = PileupMixer(michel_path=mp, pie_path=pp, mudif_path=up, pidif_path=dp, lut_dir=a.lut_dir)
    kw = dict(biased_fraction=0.0, cal_only_fraction=0.0, radio_rate=2e7)
    if a.mode == "mixed":
        mixed = mx.generate_batch(a.nev, mode="mixed", pie_mix_fraction=0.30,
                                  mudif_mix_fraction=0.25, pidif_mix_fraction=0.15, **kw)
    else:
        mixed = mx.generate_batch(a.nev, mode=a.mode, **kw)
    mixed.to_parquet(out)
    for t in (mp, pp, up, dp):
        os.remove(t)
    record_run(out, inputs, __file__, settings=vars(a))
    print(f"[tailmix t{a.task:03d} {a.mode}] -> {len(mixed)} events", flush=True)


if __name__ == "__main__":
    main()
