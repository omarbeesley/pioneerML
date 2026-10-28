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
PION, MUON, POSITRON = 1, 2, 4
C_PIE, C_MIC = "#d62728", "#1f77b4"
RAW = ["atar_x", "atar_y", "atar_z", "atar_E", "atar_pdg", "lyso_x"]
TCOLS = ["muon_score", "topo_score", "pie_score", "pileup_score", "muon_dif_score",
         "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "pred_accepted", "pred_positron_energy",
         "pred_dead_energy", "pred_positron_time_consensus_ns"]


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


# ------------------------------------------------------------------ pie reference
pie_gap, pie_q = [], []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet"))[:4]:
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
    acc = (m.truth_is_pie.values == 1) & (m.pred_accepted.values >= 1) & np.isfinite(E) & (E <= EM)
    idx = np.where(acc)[0][:60000]
    for i in idx:
        gp, q = gap_features(raw.iloc[i])
        if np.isfinite(gp):
            pie_gap.append(gp); pie_q.append(q)
pie_gap = np.array(pie_gap); pie_q = np.array(pie_q)
print(f"pie: {len(pie_gap):,} events  gap q50 {np.median(pie_gap):.3f} q99 {np.quantile(pie_gap,0.99):.3f} "
      f"q99.9 {np.quantile(pie_gap,0.999):.3f} max {pie_gap.max():.3f} mm", flush=True)
for c in (1.0, 1.5, 2.0, 3.0):
    print(f"   pie with gap > {c:.1f} mm: {100*(pie_gap>c).mean():.4f}%", flush=True)

# ------------------------------------------------------------------ michel: all + survivors
mic_gap, mic_q, surv_rows = [], [], []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    raw_f = f"{P}/michel_inwin/{piece}.parquet"
    if not (os.path.exists(tp) and os.path.exists(raw_f)):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t) or (f32(t.deposited_energy.values) != f32(m.truth_live_E.values)).mean() > 0.001:
        continue
    raw = pq.read_table(raw_f, columns=RAW).to_pandas()
    na = raw.atar_x.apply(lambda v: 0 if v is None else len(v)).to_numpy()
    nl = raw.lyso_x.apply(lambda v: 0 if v is None else len(v)).to_numpy()
    raw = raw[((na + nl) > 0) & ((na + nl) <= MAXH)].reset_index(drop=True)
    if len(raw) != len(m):
        continue
    E = m.pred_positron_energy.values + m.pred_dead_energy.values
    et = m.truth_event_type.values.astype(np.int64)
    acc = ((et & 1) == 0) & ((et & 0x1000) == 0) & ((et & 0x200) == 0) \
        & (m.pred_accepted.values >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM)
    surv = acc & (t.topo_score.values > 0.99) & (t.muon_score.values < 0.02) \
        & (t.muon_dif_score.values < 0.46) & (t.pion_dif_score.values < 0.01)
    take = np.where(acc)[0][:4000]
    for i in take:
        gp, q = gap_features(raw.iloc[i])
        if np.isfinite(gp):
            mic_gap.append(gp); mic_q.append(q)
    for i in np.where(surv)[0]:
        gp, q = gap_features(raw.iloc[i])
        surv_rows.append(dict(t=float(m.pred_positron_time_consensus_ns.values[i]),
                              E=float(E[i]), gap=float(gp), q_between=float(q),
                              mudif=float(t.muon_dif_score.values[i]),
                              pileup=float(t.pileup_score.values[i])))
mic_gap = np.array(mic_gap); mic_q = np.array(mic_q)
print(f"\nmichel (subsample): {len(mic_gap):,} events  gap q50 {np.median(mic_gap):.3f} "
      f"q90 {np.quantile(mic_gap,0.90):.3f} mm   charge between: q50 {np.median(mic_q):.4f} MeV", flush=True)

print(f"\n=== the {len(surv_rows)} surviving rows ===", flush=True)
seen = {}
for r in sorted(surv_rows, key=lambda r: r["t"]):
    k = round(r["t"], 1)
    seen.setdefault(k, []).append(r)
for k, g in seen.items():
    r = g[0]
    keep_now = (r["mudif"] < 0.40) and (r["pileup"] < 0.99)
    tag = "STILL SURVIVES muDIF<0.40 & pileup<0.99" if keep_now else "removed by muDIF<0.40 / pileup<0.99"
    pct = 100.0 * (pie_gap > r["gap"]).mean()
    print(f"  t={k:7.1f} ns (x{len(g)})  gap {r['gap']:6.3f} mm  charge between {r['q_between']:.4f} MeV"
          f"   [{100-pct:6.3f}% of pie below this gap]   {tag}", flush=True)

fig, ax = plt.subplots(figsize=(7.6, 5.2))
b = np.linspace(0, 8, 81)
ax.hist(pie_gap, bins=b, weights=np.full(len(pie_gap), 1/len(pie_gap)), histtype="step",
        lw=1.9, color=C_PIE, label=r"$\pi\to e$ (accepted)")
ax.hist(mic_gap, bins=b, weights=np.full(len(mic_gap), 1/len(mic_gap)), histtype="step",
        lw=1.9, color=C_MIC, label=r"$\pi\to\mu\to e$ (accepted)")
for k, g in seen.items():
    ax.axvline(g[0]["gap"], color="k", ls=":", lw=1.1)
ax.plot([], [], color="k", ls=":", lw=1.1, label="survivors of the topological chain")
ax.set_yscale("log"); ax.set_xlabel("gap from the last pion hit to the nearest positron hit [mm]")
ax.set_ylabel("fraction of events / bin"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_gap_feature.png"); plt.close(fig)
print(f"\nwrote {OUT}/tail_gap_feature.png\nALL DONE", flush=True)
