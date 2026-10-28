"""A reconstructed vertex cut: distance from the positron track's nearer endpoint to the
reconstructed pion stop, applied to the timing-free chain.

Nothing in the present chain uses this. A pi -> e positron begins at the pion stop; a Michel
positron begins wherever the muon stopped, and the events that survive the topological cuts
are those whose muon travelled several millimetres through dead material without depositing.
The offset is therefore a handle on exactly the residue the topology cannot see, and unlike
the prompt window it costs the signal nothing by construction.

Everything here is reconstructed: the pion stop from the event summary, the positron slice
from the role head, its endpoints from the quantile regression. The endpoint convention orders
the two ends by longitudinal coordinate rather than by time, so the vertex is taken as the
endpoint nearer the pion stop.

Outputs figures/AI/tail_vertex_reco.png and the 'vertex_cut' block of tail_stats.json.
"""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle
tail_figstyle.apply()
import tail_rates as TR

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
ES, EM = 56.0, 90.0
C_PIE, C_MIC, C_PID, C_MUD = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
KMUDIF, KRAD = 0x1000, 0x200

ECOLS = ["event_id", "truth_live_E", "pred_accepted", "pred_positron_energy", "pred_dead_energy",
         "pred_positron_time_consensus_ns", "pred_pion_stop_x", "pred_pion_stop_y", "pred_pion_stop_z"]
SCOLS = ["event_id", "role_prob_positron", "is_anchor",
         "pred_start_x", "pred_start_y", "pred_start_z", "pred_stop_x", "pred_stop_y", "pred_stop_z"]
TCOLS = ["muon_score", "topo_score", "muon_dif_score", "pion_dif_score", "deposited_energy"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def vertex_offset(ev_file, sl_file):
    """Per-event distance from the positron slice's nearer endpoint to the reco pion stop."""
    e = pd.read_parquet(ev_file, columns=ECOLS)
    s = pd.read_parquet(sl_file, columns=SCOLS)
    s = s[s.is_anchor == 0] if "is_anchor" in s else s
    s = s.sort_values("role_prob_positron").groupby("event_id", sort=False).tail(1)   # best positron slice
    s = s.set_index("event_id")
    idx = e.event_id.values
    have = s.reindex(idx)
    ps = np.stack([e.pred_pion_stop_x.values, e.pred_pion_stop_y.values, e.pred_pion_stop_z.values], 1)
    a = np.stack([have.pred_start_x.values, have.pred_start_y.values, have.pred_start_z.values], 1)
    b = np.stack([have.pred_stop_x.values, have.pred_stop_y.values, have.pred_stop_z.values], 1)
    da = np.sqrt(((a - ps) ** 2).sum(1)); db = np.sqrt(((b - ps) ** 2).sum(1))
    off = np.fmin(da, db)
    off[~np.isfinite(off)] = np.nan
    return e, off


def load(kind):
    rows = []
    if kind == "pie":
        for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
            g = ev.split("shard_")[-1].split("/")[0]
            sl = ev.replace("_events.parquet", "_slices.parquet")
            tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
            if not (os.path.exists(sl) and os.path.exists(tp)):
                continue
            e, off = vertex_offset(ev, sl)
            t = pd.read_parquet(tp, columns=TCOLS + ["is_pie"])
            if len(t) != len(e) or (f32(t.deposited_energy.values) != f32(e.truth_live_E.values)).mean() > 0.001:
                continue
            keep = t.is_pie.values == 1
            rows.append(_pack(e, t, off, keep))
    elif kind == "michel":
        for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
            piece = ev.split("/")[-2]
            sl = ev.replace("_events.parquet", "_slices.parquet")
            tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
            if not (os.path.exists(sl) and os.path.exists(tp)):
                continue
            et = pd.read_parquet(ev, columns=["truth_event_type"]).truth_event_type.values.astype(np.int64)
            e, off = vertex_offset(ev, sl)
            t = pd.read_parquet(tp, columns=TCOLS)
            if len(t) != len(e) or (f32(t.deposited_energy.values) != f32(e.truth_live_E.values)).mean() > 0.001:
                continue
            keep = ((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0)
            rows.append(_pack(e, t, off, keep))
    else:
        flag = "is_pidif" if kind == "pidif" else "is_mudif"
        ev = f"{P}/purity_eval/tail_osl/{kind}_osl_events.parquet"
        sl = f"{P}/purity_eval/tail_osl/{kind}_osl_slices.parquet"
        e, off = vertex_offset(ev, sl)
        mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                             columns=TCOLS + [flag])
        mt = mt[mt[flag] == 1].reset_index(drop=True)
        tE, mE = f32(mt.deposited_energy.values), f32(e.truth_live_E.values)
        ti, pt, pm = 0, [], []
        for mi in range(len(e)):
            while ti < len(mt) and tE[ti] != mE[mi]:
                ti += 1
            if ti >= len(mt):
                break
            pt.append(ti); pm.append(mi); ti += 1
        e = e.iloc[pm].reset_index(drop=True); off = off[pm]
        t = mt.iloc[pt].reset_index(drop=True)
        rows.append(_pack(e, t, off, np.ones(len(e), bool)))
    d = pd.concat(rows, ignore_index=True)
    print(f"{kind:7}: {len(d):>10,} rows, vertex offset defined for {100*np.isfinite(d.off).mean():.1f}%", flush=True)
    return d


def _pack(e, t, off, keep):
    E = e.pred_positron_energy.values + e.pred_dead_energy.values
    return pd.DataFrame(dict(E=E, off=off, acc=(e.pred_accepted.values >= 1),
                             t=e.pred_positron_time_consensus_ns.values,
                             topo=t.topo_score.values, muon=t.muon_score.values,
                             mudif=t.muon_dif_score.values, pidif=t.pion_dif_score.values))[keep]


pie, mic = load("pie"), load("michel")
pid, mud = load("pidif"), load("mudif")


def acc_of(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def topo_chain(d):
    return acc_of(d) & (d.topo.to_numpy() > 0.99) & (d.muon.to_numpy() < 0.02) \
        & (d.mudif.to_numpy() < 0.46) & (d.pidif.to_numpy() < 0.01)


Ep = pie.E.to_numpy(); acc_pie = acc_of(pie); base_pie = topo_chain(pie)
w_pie = TR.weights(len(pie), TR.BR_PIE)
TS = TR.tail_rescale(Ep[acc_pie])
onset = np.quantile(pie.t.to_numpy()[acc_pie & np.isfinite(pie.t.to_numpy())], 0.005)
TLO, THI = onset + 2.0, onset + 32.0
time_pie = base_pie & (pie.t.to_numpy() >= TLO) & (pie.t.to_numpy() <= THI)

print("\n=== reconstructed vertex offset (mm) ===", flush=True)
for nm, d, m in (("pie", pie, acc_pie), ("michel", mic, acc_of(mic)),
                 ("piDIF", pid, acc_of(pid)), ("muDIF", mud, acc_of(mud))):
    o = d.off.to_numpy()[m]; o = o[np.isfinite(o)]
    print(f"  {nm:7} n {len(o):>10,}  q50 {np.median(o):6.3f}  q90 {np.quantile(o,0.9):6.3f}  "
          f"q99 {np.quantile(o,0.99):6.3f}  frac>2mm {100*(o>2).mean():6.3f}%", flush=True)

print("\n=== vertex cut applied to the TIMING-FREE chain ===", flush=True)
print(f"{'cut':>6} | {'pie eff':>8} {'bias':>16} | {'michel':>7} {'piDIF':>6} {'muDIF':>7} {'muDIF contam':>13}")
rlo = (acc_pie & (Ep < ES)).sum(); rhi = (acc_pie & (Ep >= ES)).sum()
scan = []
for cut in (99.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5):
    kp = base_pie & (np.nan_to_num(pie.off.to_numpy(), nan=1e9) < cut)
    ns = {}
    for nm, d in (("michel", mic), ("piDIF", pid), ("muDIF", mud)):
        ns[nm] = int((topo_chain(d) & (np.nan_to_num(d.off.to_numpy(), nan=1e9) < cut)).sum())
    eff = 100 * kp.sum() / max(acc_pie.sum(), 1)
    slo = (kp & (Ep < ES)).sum(); shi = (kp & (Ep >= ES)).sum()
    r = (slo / max(shi, 1)) / (rlo / max(rhi, 1))
    el, eh = slo / max(rlo, 1), shi / max(rhi, 1)
    err = 100 * r * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
    km = topo_chain(mud) & (np.nan_to_num(mud.off.to_numpy(), nan=1e9) < cut)
    w_mud = TR.weights(len(mud), TR.R_MUDIF_AT_ACC * w_pie[acc_pie].sum(), mask=acc_of(mud))
    contam = TR.contamination(w_mud[km & (mud.E.to_numpy() < ES)].sum(),
                              w_pie[kp & (Ep < ES)].sum(), TS)
    scan.append(dict(cut=cut, pie_eff=float(eff), bias=float(100*(r-1)), bias_err=float(err),
                     mudif_contam=float(contam), **{f"n_{k}": v for k, v in ns.items()}))
    lab = "none" if cut > 50 else f"{cut:.1f}"
    print(f"{lab:>6} | {eff:7.2f}% {100*(r-1):+7.2f}+/-{err:4.2f}% | {ns['michel']:>7} {ns['piDIF']:>6} "
          f"{ns['muDIF']:>7,} {contam:12.2f}%", flush=True)
print(f"\nfor reference, the prompt window on the same chain: pie eff "
      f"{100*time_pie.sum()/acc_pie.sum():.2f}%", flush=True)

fig, ax = plt.subplots(figsize=(7.6, 5.4))
b = np.linspace(0, 12, 61)
for nm, d, m, c in (("$\\pi\\to e$", pie, acc_pie, C_PIE), ("$\\pi\\to\\mu\\to e$", mic, acc_of(mic), C_MIC),
                    ("$\\pi$DIF", pid, acc_of(pid), C_PID), ("$\\mu$DIF", mud, acc_of(mud), C_MUD)):
    o = d.off.to_numpy()[m]; o = o[np.isfinite(o)]
    ax.hist(o, bins=b, weights=np.full(len(o), 1/len(o)), histtype="step", lw=1.8, color=c, label=nm)
o = mic.off.to_numpy()[topo_chain(mic)]; o = o[np.isfinite(o)]
if len(o):
    ax.plot(o, np.full(len(o), 3e-6), "kv", ms=7, label="Michel survivors of the topological chain")
ax.set_yscale("log"); ax.set_xlabel("reconstructed distance from pion stop to positron endpoint [mm]")
ax.set_ylabel("fraction of accepted events / bin"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_vertex_reco.png"); plt.close(fig)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["vertex_cut"] = scan
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print(f"wrote {OUT}/tail_vertex_reco.png\nALL DONE", flush=True)
