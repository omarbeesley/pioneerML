"""ATAR event displays of the Michel events that survive the timing-free topological chain.

These are the invisible-muon events: a stopped pion whose muon leaves no ATAR hit, either
because it escaped the active volume before decaying (no kMutar) or deposited below
threshold, followed by a positron that looks exactly like a pi -> e positron.

Locates each survivor by its piece and row index, reads the raw hits from michel_inwin under
the same dataset filter the evaluation applied, and draws the two ATAR projections with the
truth pion stop, muon track and positron track overlaid, in the style of
plot_mudif_survivors.py. Writes to thesis_ref/figures/AI/survivors/.
"""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tail_figstyle
tail_figstyle.apply()

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI/survivors"
os.makedirs(OUT, exist_ok=True)
EM, MAXH = 90.0, 250
PION, MUON, POSITRON, ELECTRON, GAMMA = 1, 2, 4, 8, 16
KMUTAR, KMUDAR, KPITAR, KPIDAR = 0x4000, 0x2000, 0x40, 0x20

TCOLS = ["muon_score", "topo_score", "muon_dif_score", "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "pred_accepted", "pred_positron_energy",
         "pred_dead_energy", "pred_positron_time_consensus_ns", "truth_positron_t",
         "truth_positron_energy", "truth_theta", "truth_has_muon"]
RAW = ["atar_x", "atar_y", "atar_z", "atar_E", "atar_view", "atar_pdg", "lyso_x",
       "truth_pion_stop_x", "truth_pion_stop_y", "truth_pion_stop_z",
       "truth_muon_start_x", "truth_muon_start_y", "truth_muon_start_z",
       "truth_muon_stop_x", "truth_muon_stop_y", "truth_muon_stop_z",
       "truth_positron_start_x", "truth_positron_start_y", "truth_positron_start_z",
       "truth_positron_stop_x", "truth_positron_stop_y", "truth_positron_stop_z",
       "truth_positron_energy", "truth_theta", "event_type", "live_E"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def hit_color(pdg):
    if pdg & MUON: return "tab:red"
    if pdg & POSITRON: return "tab:blue"
    if pdg & PION: return "dimgray"
    if pdg & ELECTRON: return "tab:cyan"
    if pdg & GAMMA: return "tab:orange"
    return "tab:green"


def fin(*v):
    return all(np.isfinite(x) for x in v)


# ------------------------------------------------------------------ find the survivors
found = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t):
        continue
    if (f32(t.deposited_energy.values) != f32(m.truth_live_E.values)).mean() > 0.001:
        continue
    E = m.pred_positron_energy.values + m.pred_dead_energy.values
    et = m.truth_event_type.values.astype(np.int64)
    keep = ((et & 1) == 0) & ((et & 0x1000) == 0) & ((et & 0x200) == 0)
    surv = keep & (m.pred_accepted.values >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM) \
        & (t.topo_score.values > 0.99) & (t.muon_score.values < 0.02) \
        & (t.muon_dif_score.values < 0.46) & (t.pion_dif_score.values < 0.01)
    for i in np.where(surv)[0]:
        found.append(dict(piece=piece, row=int(i), Ereco=float(E[i]),
                          t=float(m.pred_positron_time_consensus_ns.values[i]),
                          t_true=float(m.truth_positron_t.values[i]),
                          E_true=float(m.truth_positron_energy.values[i]),
                          et=int(et[i]), has_mu=int(m.truth_has_muon.values[i]),
                          topo=float(t.topo_score.values[i]), muon=float(t.muon_score.values[i]),
                          mudif=float(t.muon_dif_score.values[i]),
                          live_E=float(m.truth_live_E.values[i])))
print(f"survivor rows located: {len(found)}", flush=True)

# one display per DISTINCT decay (trigger reuse duplicates the same decay across mixed events)
seen, uniq = set(), []
for s in found:
    k = round(s["t_true"], 3)
    if k in seen:
        continue
    seen.add(k); uniq.append(s)
print(f"distinct decays: {len(uniq)}", flush=True)

# ------------------------------------------------------------------ draw
made = 0
by_piece = {}
for s in uniq:
    by_piece.setdefault(s["piece"], []).append(s)
for piece, items in by_piece.items():
    tb = pq.read_table(f"{P}/michel_inwin/{piece}.parquet", columns=RAW).to_pandas()
    na = tb.atar_x.apply(lambda x: 0 if x is None else len(x)).to_numpy()
    nl = tb.lyso_x.apply(lambda x: 0 if x is None else len(x)).to_numpy()
    tb = tb[((na + nl) > 0) & ((na + nl) <= MAXH)].reset_index(drop=True)
    for s in items:
        if s["row"] >= len(tb):
            print(f"  row {s['row']} beyond piece {piece} ({len(tb)})", flush=True); continue
        r = tb.iloc[s["row"]]
        if abs(float(r.live_E) - s["live_E"]) > 1e-3:
            print(f"  live_E mismatch in {piece} row {s['row']}: {float(r.live_E):.4f} vs {s['live_E']:.4f}", flush=True)
            continue
        az = np.asarray(r.atar_z, float)
        if len(az) == 0:
            continue
        ax_, ay_ = np.asarray(r.atar_x, float), np.asarray(r.atar_y, float)
        aE = np.asarray(r.atar_E, float); aview = np.asarray(r.atar_view, float)
        apdg = np.asarray(r.atar_pdg, int)
        sizes = 15.0 + 280.0 * (aE / max(aE.max(), 1e-6))
        colors = np.array([hit_color(int(p)) for p in apdg])
        is_yz = aview > 0.5
        T = dict(pion=(r.truth_pion_stop_x, r.truth_pion_stop_y, r.truth_pion_stop_z),
                 mu0=(r.truth_muon_start_x, r.truth_muon_start_y, r.truth_muon_start_z),
                 mu1=(r.truth_muon_stop_x, r.truth_muon_stop_y, r.truth_muon_stop_z),
                 e0=(r.truth_positron_start_x, r.truth_positron_start_y, r.truth_positron_start_z),
                 e1=(r.truth_positron_stop_x, r.truth_positron_stop_y, r.truth_positron_stop_z))
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), sharex=True)
        transv = {0: ax_, 1: ay_}
        for ax, sel, tc, tl in [(axes[0], ~is_yz, 0, "x"), (axes[1], is_yz, 1, "y")]:
            if sel.any():
                ax.scatter(az[sel], transv[tc][sel], s=sizes[sel], c=list(colors[sel]),
                           edgecolors="k", linewidths=0.3, alpha=0.85, zorder=3)
            px, py, pz = T["pion"]
            if fin(pz, px, py):
                ax.scatter([pz], [px if tc == 0 else py], marker="*", s=260, c="k", zorder=4)

            def seg(a, b, c):
                if fin(a[2], b[2], a[tc], b[tc]):
                    ax.plot([a[2], b[2]], [a[tc], b[tc]], color=c, ls="--", lw=1.6, alpha=0.8, zorder=2)
            seg(T["mu0"], T["mu1"], "tab:red"); seg(T["e0"], T["e1"], "tab:blue")
            ax.set_ylabel(f"{tl} [mm]"); ax.set_ylim(-11.5, 11.5)
        axes[1].set_xlabel("z [mm]"); axes[1].set_xlim(-0.2, 7.0)
        n_mu = int(((apdg & MUON) > 0).sum()); n_pos = int(((apdg & POSITRON) > 0).sum())
        mdist = (np.sqrt(sum((T["mu1"][k] - T["mu0"][k]) ** 2 for k in range(3)))
                 if fin(*T["mu0"], *T["mu1"]) else float("nan"))
        tags = " ".join(n for b, n in ((KPIDAR, "kPidar"), (KPITAR, "kPitar"),
                                       (KMUDAR, "kMudar"), (KMUTAR, "kMutar")) if s["et"] & b)
        fig.suptitle(
            f"Michel survivor of the timing-free chain   [{tags}]\n"
            f"muon ATAR hits {n_mu}   muon truth range {mdist:.2f} mm   truth has\\_muon {s['has_mu']}\n"
            f"$e^+$ {s['E_true']:.1f} MeV at {s['t_true']:.0f} ns, reconstructed {s['Ereco']:.1f} MeV at {s['t']:.0f} ns\n"
            f"topology {s['topo']:.3f}   muon veto {s['muon']:.3f}   $\\mu$DIF veto {s['mudif']:.3f}",
            fontsize=8.6)
        handles = [Line2D([0], [0], marker='o', ls='', mfc=c, mec='k', label=l) for c, l in
                   [("tab:red", "muon hit"), ("tab:blue", "$e^+$ hit"), ("dimgray", "pion hit"),
                    ("tab:cyan", "electron"), ("tab:orange", "photon")]]
        handles += [Line2D([0], [0], marker='*', ls='', mfc='k', mec='k', label='pion stop (truth)'),
                    Line2D([0], [0], color='tab:red', ls='--', label='muon track (truth)'),
                    Line2D([0], [0], color='tab:blue', ls='--', label='$e^+$ track (truth)')]
        axes[0].legend(handles=handles, fontsize=7, loc="best", ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = f"{OUT}/michel_survivor_{made:02d}.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  wrote {path}  ({tags}, muon hits {n_mu}, range {mdist:.2f} mm)", flush=True)
        made += 1
print(f"\n{made} displays written to {OUT}\nALL DONE", flush=True)
