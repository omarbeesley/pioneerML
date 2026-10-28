"""Concatenate mixed shards into one training table, dropping non-finite events.

Port of the external concat_train_fixed.py that built train10M_fixed/train.parquet
and val.parquet. An event is dropped when any of its ATAR or LYSO hit coordinates,
times or energies is NaN or infinite. Each input file is written as one or more
row groups, which the trainer's chunked epochs shuffle.

    python -m purity.preprocessing.concat_clean \\
        --input '/data/train10M_fixed/mix_*/data.parquet' \\
        --output outputs/train10M/train.parquet
"""
import argparse
import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from purity.preprocessing._io import ensure_output, record_run

FLOAT_ARR = ["atar_x", "atar_y", "atar_z", "atar_t", "atar_truth_t", "atar_E",
             "lyso_x", "lyso_y", "lyso_z", "lyso_t", "lyso_E"]


def clean_concat(files, out):
    w = None
    tot = kept = 0
    for f in files:
        t = pq.read_table(f)
        bad = np.zeros(t.num_rows, bool)
        for c in FLOAT_ARR:
            if c not in t.column_names:
                continue
            col = t.column(c).combine_chunks()
            fl = col.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
            off = col.offsets.to_numpy()
            lens = np.diff(off)
            ev = np.repeat(np.arange(len(lens)), lens)
            nf = ~np.isfinite(fl)
            if nf.any():
                bad |= np.bincount(ev[nf], minlength=t.num_rows) > 0
        keep = t.filter(pa.array(~bad))
        if w is None:
            w = pq.ParquetWriter(out, keep.schema, compression="snappy")
        w.write_table(keep)
        tot += t.num_rows
        kept += keep.num_rows
        print(f"  {f}: {t.num_rows}->{keep.num_rows}", flush=True)
    w.close()
    print(f"{out}: kept {kept:,}/{tot:,}", flush=True)
    return {"rows_in": tot, "rows_kept": kept}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Quoted glob of shard parquets")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(a.input))
    if not files:
        raise SystemExit(f"no inputs match {a.input}")
    out = ensure_output(a.output)
    counts = clean_concat(files, out)
    record_run(out, files, __file__, settings={"input": a.input, **counts})


if __name__ == "__main__":
    main()
