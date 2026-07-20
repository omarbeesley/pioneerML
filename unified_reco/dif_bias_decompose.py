"""
Isolate WHICH parts of the muDIF / piDIF veto heads bias the pi->e nu tail fraction.
Run inside pytorch.sif on the GPU node.

Each head's logit = fuse([pool_feat, scalar_feat]) where pool_feat is the learned
attention pool over the (GNN) trunk and scalar_feat is the MLP of the engineered
physics scalars. On accepted pi->e nu (E = deposited_energy + dead_E; tail = E<e_split):

(1) PATHWAY ABLATION  -- re-evaluate fuse with one pathway zeroed:
      full        = fuse([pool_feat , scalar_feat])     (== the real head score)
      pool_only   = fuse([pool_feat , 0          ])
      scalar_only = fuse([0         , scalar_feat])
    and report each one's tail-fraction bias eff_tail/eff_peak. Whichever pathway
    reproduces the full head's bias is where the bias lives: the learned POOL or the
    engineered SCALARS.

(2) PER-SCALAR  -- for each engineered scalar: mean(peak), mean(tail), their ratio,
    corr(scalar, head_logit), and the scalar's OWN standalone tail bias (cutting on it
    in the direction it pushes the veto). A scalar that BOTH differs peak-vs-tail AND
    drives the logit AND has a standalone bias != 1 is a culprit.

muDIF uses full acceptance; piDIF uses angle-only (<angle_max, no pion-stop fiducial).
Optional --theta_min/--theta_max restricts to an angle band (e.g. the forward-theta
muDIF events that the bias-vs-theta plot flagged).

Usage:
  apptainer exec --nv --cleanenv --contain \
    --bind /home/obeesley/pioneerML:/pioneerML --bind /data/nvme0/prod_ml_data:/data \
    /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
    python3 /pioneerML/unified_reco/dif_bias_decompose.py \
      --checkpoint /pioneerML/model_weights/<ckpt>.pth \
      --data /data/tail_reveal_pie_eval.parquet --max_events 300000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_reco.dataset import PURITYDataset
from unified_reco.models_tail import PURITYTailModel, assemble_tail_features
from torch_geometric.loader import DataLoader

# display name, model attribute, acceptance mode
HEADS = [("muDIF", "mudif_veto", "full"),
         ("piDIF", "pidif_veto", "angle")]


@torch.inference_mode()
def collect(model, loader, device):
    out = {nm: {"full": [], "pool_only": [], "scalar_only": [], "scalars": [],
                "P": [], "S": []}
           for nm, _, _ in HEADS}
    meta = {"E_tot": [], "theta": [], "is_pie": [], "acceptance": []}
    for batch in loader:
        batch = batch.to(device)
        bb = model.backbone(batch.x, batch.batch)
        if "h_atar" not in bb:
            continue
        f = assemble_tail_features(bb, batch.x, batch.batch, detach=model.freeze_trunk)
        for nm, attr, _ in HEADS:
            head = getattr(model, attr)
            logit, _node, parts = head(f, return_parts=True)
            scal = parts["scalars"]                       # [B, n_scalars]
            pool_feat = parts["pool_feat"]                # [B, h]
            scalar_feat = head.scalar_mlp(scal)           # [B, h]
            z_s = torch.zeros_like(scalar_feat)
            z_p = torch.zeros_like(pool_feat)
            full = head.fuse(torch.cat([pool_feat, scalar_feat], -1)).squeeze(-1)
            pool_only = head.fuse(torch.cat([pool_feat, z_s], -1)).squeeze(-1)
            scalar_only = head.fuse(torch.cat([z_p, scalar_feat], -1)).squeeze(-1)
            out[nm]["full"].append(full.cpu().numpy())
            out[nm]["pool_only"].append(pool_only.cpu().numpy())
            out[nm]["scalar_only"].append(scalar_only.cpu().numpy())
            out[nm]["scalars"].append(scal.cpu().numpy())
            # post-MLP pathway features, for the additive (functional-ANOVA) decomposition
            out[nm]["P"].append(pool_feat.cpu().numpy().astype(np.float32))
            out[nm]["S"].append(scalar_feat.cpu().numpy().astype(np.float32))
        live = batch.live_E_target.view(-1).float()
        dead = batch.dead_E_target.view(-1).float()
        meta["E_tot"].append((live + dead).cpu().numpy())
        meta["theta"].append(torch.rad2deg(batch.positron_theta_target.view(-1)).cpu().numpy()
                             if hasattr(batch, "positron_theta_target")
                             else np.full(live.shape[0], np.nan))
        meta["is_pie"].append(batch.is_pie_target.view(-1).cpu().numpy())
        meta["acceptance"].append(batch.acceptance_target.view(-1).cpu().numpy())
    res = {nm: {k: np.concatenate(v) for k, v in d.items()} for nm, d in out.items()}
    res["_meta"] = {k: np.concatenate(v) for k, v in meta.items()}
    return res


@torch.inference_mode()
def anova_shift(head, P, S, pie, peak, tail, device):
    """Additive functional-ANOVA of the tail-peak mean LOGIT shift on accepted pie:
       Delta_full = Delta_pool_main + Delta_scalar_main + Delta_interaction (exact).
    Each main effect varies one pathway with the OTHER held at its pie MEAN (on-
    distribution, unlike zeroing). Higher logit => more veto, so Delta_full < 0 means
    tail pie has a LOWER logit (over-survives => bias > 1); > 0 => bias < 1."""
    Pt = torch.tensor(P[pie], device=device)
    St = torch.tensor(S[pie], device=device)
    pbar = Pt.mean(0, keepdim=True)
    sbar = St.mean(0, keepdim=True)
    fu = lambda p, s: head.fuse(torch.cat([p, s], -1)).squeeze(-1)
    L_full = fu(Pt, St)
    L_pool = fu(Pt, sbar.expand_as(St))           # vary pool, scalar at mean
    L_scal = fu(pbar.expand_as(Pt), St)           # vary scalar, pool at mean
    L_base = fu(pbar, sbar).item()
    g_pool = (L_pool - L_base).cpu().numpy()
    g_scal = (L_scal - L_base).cpu().numpy()
    h = (L_full - L_pool - L_scal + L_base).cpu().numpy()   # interaction
    Lf = L_full.cpu().numpy()
    pk, tl = peak[pie], tail[pie]
    sh = lambda x: float(x[tl].mean() - x[pk].mean())
    return dict(full=sh(Lf), pool=sh(g_pool), scalar=sh(g_scal), inter=sh(h))


def veto_bias(values, pie, peak, tail, pie_eff, keep_low=True):
    """tail bias eff_tail/eff_peak treating `values` as a veto score."""
    v = values
    if keep_low:
        thr = np.quantile(v[pie], pie_eff); keep = v < thr
    else:
        thr = np.quantile(v[pie], 1.0 - pie_eff); keep = v > thr
    ep, et = keep[peak].mean(), keep[tail].mean()
    n = int(tail.sum())
    b = (et / ep) if ep > 0 else float("nan")
    be = (b * np.sqrt(et * (1 - et) / (n * et ** 2 + 1e-12)) if (et > 0 and n) else float("nan"))
    return b, be


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", nargs="+", required=True, help="pie eval parquet(s)")
    p.add_argument("--max_events", type=int, default=None)
    p.add_argument("--max_hits", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--pie_eff", type=float, default=0.5)
    p.add_argument("--e_split", type=float, default=56.0)
    p.add_argument("--e_max", type=float, default=75.0)
    p.add_argument("--accept_min", type=float, default=0.5)
    p.add_argument("--angle_max", type=float, default=120.0)
    p.add_argument("--theta_min", type=float, default=None, help="restrict to theta >= this (deg)")
    p.add_argument("--theta_max", type=float, default=None, help="restrict to theta < this (deg)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PURITYTailModel(dropout=0.0).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.checkpoint} ({len(miss)} missing / {len(unexp)} unexpected)", flush=True)

    parts = []
    for path in args.data:
        ds = PURITYDataset(path, max_hits=args.max_hits, max_events=args.max_events)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        parts.append(collect(model, loader, device))
    # merge across files
    res = {}
    for nm, _, _ in HEADS:
        res[nm] = {k: np.concatenate([pp[nm][k] for pp in parts]) for k in parts[0][nm]}
    meta = {k: np.concatenate([pp["_meta"][k] for pp in parts]) for k in parts[0]["_meta"]}

    E = meta["E_tot"]; th = meta["theta"]
    base = (meta["is_pie"] == 1) & np.isfinite(E) & (E <= args.e_max)
    if args.theta_min is not None:
        base &= (th >= args.theta_min)
    if args.theta_max is not None:
        base &= (th < args.theta_max)
    band = ""
    if args.theta_min is not None or args.theta_max is not None:
        band = f"  [theta in [{args.theta_min},{args.theta_max}) deg]"

    for nm, attr, acc_mode in HEADS:
        head = getattr(model, attr)
        acc = (th < args.angle_max) if acc_mode == "angle" else (meta["acceptance"] >= args.accept_min)
        pie = base & acc
        if pie.sum() < 100:
            print(f"\n=== {nm} [skip] only {int(pie.sum())} accepted pie ==="); continue
        peak = pie & (E >= args.e_split)
        tail = pie & (E < args.e_split)
        d = res[nm]
        print(f"\n=================== {nm} ({acc_mode} acc){band} ===================")
        print(f"accepted pie={int(pie.sum())}  peak={int(peak.sum())}  tail={int(tail.sum())}"
              f"  (@ pie_eff={args.pie_eff:g}, E=live+dead, tail<{args.e_split:g})")

        print("\n(1) PATHWAY ABLATION  (tail bias; NON-additive: each at its OWN threshold):")
        for key in ("full", "pool_only", "scalar_only"):
            b, be = veto_bias(d[key], pie, peak, tail, args.pie_eff, keep_low=True)
            print(f"    {key:12s} bias = {b:.3f} +/- {be:.3f}")

        a = anova_shift(head, d["P"], d["S"], pie, peak, tail, device)
        print("\n(1b) ADDITIVE functional-ANOVA  (tail-peak mean LOGIT shift; EXACT decomposition;"
              "\n     shift>0 => tail higher logit => over-veto => bias<1, and vice-versa):")
        print(f"    Delta_full     = {a['full']:+.4f}")
        print(f"    Delta_pool     = {a['pool']:+.4f}   (learned attention pool)")
        print(f"    Delta_scalar   = {a['scalar']:+.4f}   (engineered scalars)")
        print(f"    Delta_interact = {a['inter']:+.4f}   (pool x scalar, the cross-term)")
        print(f"    sum(parts)     = {a['pool']+a['scalar']+a['inter']:+.4f}  (should == Delta_full)")

        print("\n(2) PER-SCALAR  (peak/tail mean, ratio, corr-with-logit, standalone bias):")
        names = head.SCALAR_NAMES
        logit = d["full"]
        S = np.nan_to_num(d["scalars"])
        print(f"    {'scalar':18s} {'peak':>9s} {'tail':>9s} {'ratio':>7s} {'corr':>7s} {'st.bias':>9s}")
        for i, snm in enumerate(names):
            x = S[:, i]
            mp, mt = x[peak].mean(), x[tail].mean()
            ratio = (mt / mp) if abs(mp) > 1e-9 else float("nan")
            c = np.corrcoef(x[pie], logit[pie])[0, 1]
            b, _ = veto_bias(x, pie, peak, tail, args.pie_eff, keep_low=(c >= 0))
            print(f"    {snm:18s} {mp:9.4f} {mt:9.4f} {ratio:7.2f} {c:+7.2f} {b:9.3f}")
    print("\nRead-off: the PATHWAY whose bias matches 'full' is the seat; within SCALARS,"
          "\na feature with |corr|>~0.1 AND standalone bias far from 1.0 is a driver.")


if __name__ == "__main__":
    main()
