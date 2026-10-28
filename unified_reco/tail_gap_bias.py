"""A hit-level feature for the invisible-muon residue: the empty gap between the pion track
and the positron track.

The fitted-endpoint version of this (tail_vertex_cut.py) was blunted by the endpoint
regression's resolution. This measures the same physics directly from the hits: the distance
from the last pion hit to the nearest positron hit, and how much charge sits in between. A
pi -> e positron starts in the pion's own pixel, so the gap is a pixel wide and there is
nothing between. An ordinary Michel has the muon's Bragg deposit filling the gap. The
survivors have several millimetres of empty silicon, which is a signature neither of the
other two has.

Truth hit labels are used here to establish whether the separation exists at all; if it does,
the reconstructed version uses the model's own hit classification, which is near-perfect for
these events (positron hit IoU 1.000 on every survivor).
"""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle
tail_figstyle.apply()

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
EM, MAXH = 90.0, 250
ES = 56.0
PION, MUON, POSITRON = 1, 2, 4
C_PIE, C_MIC = "#d62728", "#1f77b4"
RAW = ["atar_x", "atar_y", "atar_z", "atar_E", "atar_pdg", "lyso_x"]
TCOLS = ["muon_score", "topo_score", "pie_score", "pileup_score", "muon_dif_score",
         "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "pred_accepted", "pred_positron_energy",
         "pred_dead_energy", "pred_positron_time_consensus_ns", "truth_theta", "pred_polar_angle"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def gap_features(row):
    """(gap, charge in the gap) between the pion track end and the positron track start."""
    pdg = np.asarray(row["atar_pdg"], int)
    if len(pdg) == 0:
        return np.nan, np.nan
    x = np.asarray(row["atar_x"], float); y = np.asarray(row["atar_y"], float)
    z = np.asarray(row["atar_z"], float); E = np.asarray(row["atar_E"], float)
    ip = (pdg & PION) > 0
    ie = (pdg & POSITRON) > 0
    if not ip.any() or not ie.any():
        return np.nan, np.nan
    pp = np.stack([x[ip], y[ip], z[ip]], 1)
    pe = np.stack([x[ie], y[ie], z[ie]], 1)
    d = np.sqrt(((pp[:, None, :] - pe[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmin(d), d.shape)
    gap = d[i, j]
    a, b = pp[i], pe[j]                                  # the two closest hits
    seg = b - a; L = np.sqrt((seg ** 2).sum())
    if L < 1e-6:
        return gap, 0.0
    u = seg / L
    allp = np.stack([x, y, z], 1)
    t = ((allp - a) @ u)
    perp = np.sqrt(np.maximum(((allp - a) ** 2).sum(1) - t ** 2, 0))
    between = (t > 0.15) & (t < L - 0.15) & (perp < 0.6) & ~ip & ~ie   # neither pion nor positron
    return gap, float(E[between].sum())


# ---------------- does a gap cut sculpt the pie tail? ----------------------------------
# The concern: a transverse positron, or a pion stopping in dead material / at a strip end,
# displaces the first recorded hit and fakes a gap -- and those are the events that populate
# the tail, so the cut would remove the tail preferentially.
rows = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet"))[:6]:
    g = ev.split("shard_")[-1].split("/")[0]
    raw_f = f"{P}/norad_mixed2_rs/mix_{g}/data.parquet"
    if not os.path.exists(raw_f):
        continue
    m = pd.read_parquet(ev, columns=MCOLS + ["truth_is_pie"])
    raw = pq.read_table(raw_f, columns=RAW).to_pandas()
    na = raw.atar_x.apply(lambda v: 0 if v is None else len(v)).to_numpy()
    nl = raw.lyso_x.apply(lambda v: 0 if v is None else len(v)).to_numpy()
    raw = raw[((na + nl) > 0) & ((na + nl) <= MAXH)].reset_index(drop=True)
    if len(raw) != len(m):
        continue
    E = m.pred_positron_energy.values + m.pred_dead_energy.values
    acc = (m.truth_is_pie.values == 1) & (m.pred_accepted.values >= 1) & np.isfinite(E) & (E <= EM) & (E >= 0)
    th = np.degrees(m.truth_theta.values)
    idx = np.where(acc)[0]
    # keep ALL tail events (they are rare and are the point of the study) + a peak subsample
    tail_idx = idx[E[idx] < ES]
    peak_idx = idx[E[idx] >= ES][:40000]
    for i in np.concatenate([tail_idx, peak_idx]):
        gp, q = gap_features(raw.iloc[i])
        if np.isfinite(gp):
            rows.append((gp, q, float(E[i]), float(th[i])))
    if len(rows) > 150000:
        break
d = pd.DataFrame(rows, columns=["gap", "q", "E", "theta"])
tail = d.E < ES; peak = ~tail
print(f"pie sample: {len(d):,}  ({int(tail.sum()):,} tail, {int(peak.sum()):,} peak)", flush=True)
print(f"  gap  peak q50 {d.gap[peak].median():.3f}  q99 {d.gap[peak].quantile(0.99):.3f}  "
      f"q99.9 {d.gap[peak].quantile(0.999):.3f} mm", flush=True)
print(f"  gap  TAIL q50 {d.gap[tail].median():.3f}  q99 {d.gap[tail].quantile(0.99):.3f}  "
      f"q99.9 {d.gap[tail].quantile(0.999):.3f} mm", flush=True)

print("\n=== gap vs positron angle (accepted pie) ===", flush=True)
for lo, hi in ((0,30),(30,60),(60,80),(80,100),(100,120)):
    m_ = (d.theta >= lo) & (d.theta < hi)
    if m_.sum() < 50: continue
    print(f"  theta {lo:3}-{hi:3} deg  n {int(m_.sum()):>7,}  gap q50 {d.gap[m_].median():6.3f}  "
          f"q99 {d.gap[m_].quantile(0.99):6.3f}  frac>1.5mm {100*(d.gap[m_]>1.5).mean():6.3f}%", flush=True)

print("\n=== induced tail bias of a gap cut (efficiency tail / efficiency peak) ===", flush=True)
print(f"{'cut':>7} {'eff peak':>9} {'eff tail':>9} {'bias':>16}")
for cut in (5.0, 3.0, 2.0, 1.5, 1.0, 0.7):
    ep = (d.gap[peak] < cut).mean(); et = (d.gap[tail] < cut).mean()
    npk, ntl = int(peak.sum()), int(tail.sum())
    r = et / ep if ep > 0 else 0
    err = 100 * r * np.sqrt(abs(1-et)/max(et*ntl,1) + abs(1-ep)/max(ep*npk,1))
    print(f"{cut:7.1f} {100*ep:8.3f}% {100*et:8.3f}%  {100*(r-1):+7.2f}+/-{err:5.2f}%", flush=True)
print("\nALL DONE", flush=True)
