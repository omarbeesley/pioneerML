"""Does a positron-to-pion-stop vertex compatibility separate the invisible-muon survivors?

The displays show the surviving Michel events share one geometry: the muon leaves the pion
stop travelling within a plane of constant z, deposits nothing (it is in dead material between
the silicon layers), and decays several millimetres away. The positron therefore starts far
from the pion stop, whereas a genuine pi -> e positron starts exactly at it. This measures the
separation that offset provides, on truth quantities, to say whether it is worth reconstructing.
"""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle
tail_figstyle.apply()

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
EM, MAXH = 90.0, 250
C_PIE, C_MIC = "#d62728", "#1f77b4"

TCOLS = ["muon_score", "topo_score", "muon_dif_score", "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "pred_accepted", "pred_positron_energy",
         "pred_dead_energy", "pred_positron_time_consensus_ns"]
RAWC = ["live_E", "truth_pion_stop_x", "truth_pion_stop_y", "truth_pion_stop_z",
        "truth_positron_start_x", "truth_positron_start_y", "truth_positron_start_z"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def offsets(raw):
    d = np.sqrt((raw.truth_positron_start_x.values - raw.truth_pion_stop_x.values) ** 2
                + (raw.truth_positron_start_y.values - raw.truth_pion_stop_y.values) ** 2
                + (raw.truth_positron_start_z.values - raw.truth_pion_stop_z.values) ** 2)
    return d


# ---------------------------------------------------------------- michel: offset vs survival
off_all, off_surv, off_novis = [], [], []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    raw_f = f"{P}/michel_inwin/{piece}.parquet"
    if not (os.path.exists(tp) and os.path.exists(raw_f)):
        continue
    m = pd.read_parquet(ev, columns=MCOLS + ["truth_has_muon"])
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t):
        continue
    if (f32(t.deposited_energy.values) != f32(m.truth_live_E.values)).mean() > 0.001:
        continue
    raw = pq.read_table(raw_f, columns=RAWC + ["atar_x", "lyso_x"]).to_pandas()
    na = raw.atar_x.apply(lambda x: 0 if x is None else len(x)).to_numpy()
    nl = raw.lyso_x.apply(lambda x: 0 if x is None else len(x)).to_numpy()
    raw = raw[((na + nl) > 0) & ((na + nl) <= MAXH)].reset_index(drop=True)
    if len(raw) != len(m):
        continue
    E = m.pred_positron_energy.values + m.pred_dead_energy.values
    et = m.truth_event_type.values.astype(np.int64)
    acc = ((et & 1) == 0) & ((et & 0x1000) == 0) & ((et & 0x200) == 0) \
        & (m.pred_accepted.values >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM)
    surv = acc & (t.topo_score.values > 0.99) & (t.muon_score.values < 0.02) \
        & (t.muon_dif_score.values < 0.46) & (t.pion_dif_score.values < 0.01)
    d = offsets(raw)
    off_all.append(d[acc]); off_surv.append(d[surv])
    off_novis.append(d[acc & (m.truth_has_muon.values == 0)])
off_all = np.concatenate(off_all); off_surv = np.concatenate(off_surv)
off_novis = np.concatenate(off_novis)
off_all = off_all[np.isfinite(off_all)]

# ---------------------------------------------------------------- pie reference
pie_off = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet"))[:8]:
    g = ev.split("shard_")[-1].split("/")[0]
    raw_f = f"{P}/norad_mixed2_rs/mix_{g}/data.parquet"
    if not os.path.exists(raw_f):
        continue
    m = pd.read_parquet(ev, columns=MCOLS + ["truth_is_pie"])
    raw = pq.read_table(raw_f, columns=RAWC + ["atar_x", "lyso_x"]).to_pandas()
    na = raw.atar_x.apply(lambda x: 0 if x is None else len(x)).to_numpy()
    nl = raw.lyso_x.apply(lambda x: 0 if x is None else len(x)).to_numpy()
    raw = raw[((na + nl) > 0) & ((na + nl) <= MAXH)].reset_index(drop=True)
    if len(raw) != len(m):
        continue
    E = m.pred_positron_energy.values + m.pred_dead_energy.values
    acc = (m.truth_is_pie.values == 1) & (m.pred_accepted.values >= 1) & np.isfinite(E) & (E <= EM)
    d = offsets(raw)
    pie_off.append(d[acc])
pie_off = np.concatenate(pie_off) if pie_off else np.array([])
pie_off = pie_off[np.isfinite(pie_off)]

print(f"pie accepted        {len(pie_off):>10,}  offset q50 {np.median(pie_off):.3f} mm  "
      f"q99 {np.quantile(pie_off, 0.99):.3f} mm  frac>2mm {100*(pie_off>2).mean():.4f}%", flush=True)
print(f"michel accepted     {len(off_all):>10,}  offset q50 {np.median(off_all):.3f} mm  "
      f"frac>2mm {100*(off_all>2).mean():.3f}%", flush=True)
print(f"michel no muon hit  {len(off_novis):>10,}  offset q50 {np.median(off_novis):.3f} mm  "
      f"frac>2mm {100*(off_novis>2).mean():.1f}%", flush=True)
print(f"michel SURVIVORS    {len(off_surv):>10,}  offsets: " + " ".join(f"{v:.2f}" for v in np.sort(off_surv)), flush=True)
for cut in (1.0, 1.5, 2.0, 3.0):
    print(f"  vertex cut at {cut:.1f} mm: keeps {100*(pie_off<=cut).mean():6.3f}% of pie, "
          f"removes {100*(off_surv>cut).mean():5.1f}% of the survivors", flush=True)

fig, ax = plt.subplots(figsize=(7.4, 5.2))
b = np.linspace(0, 15, 76)
ax.hist(pie_off, bins=b, weights=np.full(len(pie_off), 1/len(pie_off)), histtype="step",
        lw=1.9, color=C_PIE, label=r"$\pi\to e$ (accepted)")
ax.hist(off_all, bins=b, weights=np.full(len(off_all), 1/len(off_all)), histtype="step",
        lw=1.9, color=C_MIC, label=r"$\pi\to\mu\to e$ (accepted)")
if len(off_novis):
    ax.hist(off_novis, bins=b, weights=np.full(len(off_novis), 1/len(off_novis)),
            histtype="stepfilled", color="0.75", ec="0.4", alpha=0.7,
            label=r"$\pi\to\mu\to e$, no muon hit")
for v in np.sort(off_surv):
    ax.axvline(v, color="k", ls=":", lw=1.1)
ax.plot([], [], color="k", ls=":", lw=1.1, label="survivors of the timing-free chain")
ax.set_yscale("log"); ax.set_xlabel("distance from pion stop to positron origin [mm]")
ax.set_ylabel("fraction of events / bin"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_vertex_offset.png"); plt.close(fig)
print(f"wrote {OUT}/tail_vertex_offset.png\nALL DONE", flush=True)
