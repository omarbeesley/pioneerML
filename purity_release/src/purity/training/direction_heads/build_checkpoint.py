"""Write a weights-only checkpoint of a network with a set of trained direction heads.

    python -m purity.training.direction_heads.build_checkpoint BASE.pth HEADS_DIR OUT.pth [--first N]

BASE.pth is a PURITYModel checkpoint (with or without optimizer state); HEADS_DIR holds
head_XX.pt files written by purity.training.direction_heads.train (one set). The heads, in the
order of their file names (the first N with --first), become dir_ensemble.0 ... dir_ensemble.K-1;
the model reads them with PURITY_DIR_HEADS = K. The network's own head is kept unchanged.
"""
import argparse
import glob
import hashlib
import os

import torch

PREFIX = ("dir_attn_pool.", "positron_dir_head.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("heads")
    ap.add_argument("out")
    ap.add_argument("--first", type=int)
    args = ap.parse_args()
    base = torch.load(args.base, map_location="cpu")
    sd = {k: v for k, v in base["model"].items() if not k.startswith("dir_ensemble.")}
    files = sorted(glob.glob(f"{args.heads}/head_*.pt"))[:args.first]
    assert files, f"no heads in {args.heads}"
    records = []
    for i, f in enumerate(files):
        h = torch.load(f, map_location="cpu")
        keys = [k for k in h["head"] if k.startswith(PREFIX)]
        assert len(keys) == 10 and all(k in sd for k in keys), f"{f}: not a direction head"
        for k in keys:
            assert h["head"][k].shape == sd[k].shape, f"{f}: {k} shape differs from the network's head"
            sd[f"dir_ensemble.{i}.{k}"] = h["head"][k].clone()
        records.append({"file": os.path.basename(f), **{k: v for k, v in h.get("record", {}).items()
                                                        if k not in ("reference",)}})
    sha = hashlib.sha256(open(args.base, "rb").read()).hexdigest()
    torch.save({"model": sd, "epoch": base.get("epoch"), "direction_heads": len(files),
                "base_checkpoint": os.path.basename(args.base), "base_sha256": sha,
                "head_records": records}, args.out)
    print(f"wrote {args.out}: base {os.path.basename(args.base)} (epoch {base.get('epoch')}) + {len(files)} heads "
          f"from {args.heads}; load with PURITY_DIR_HEADS={len(files)}")


if __name__ == "__main__":
    main()
