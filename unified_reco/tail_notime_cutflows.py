"""Topological suppression measured without the prompt time window.

The chains of tail_reco_cutflows.py open with the prompt window, which removes most of the
Michel population before any network score acts, so the suppression attributed to the cuts is
measured on the surviving few percent. Here the order is inverted: the topological cuts act on
the whole accepted population, and the prompt window is applied last, as a factor that stacks
on top. This separates what the network achieves from what the clock achieves.

Stages: acceptance -> + pie topology > 0.99 -> + muon veto -> + pion-DIF veto -> + prompt window.

Outputs figures/AI/tail_cutflow_notime.png (one panel per background) and the
'notime_cutflows' block of figures/AI/tail_stats.json.
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle
tail_figstyle.apply()
import tail_rates as TR

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
C_PIE, C_MIC, C_PID, C_MUD = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
ES, EM = 56.0, 90.0
KMUDIF, KRAD = 0x1000, 0x200
bins = np.linspace(0, EM, 61)
CL90 = 2.3

TCOLS = ["muon_score", "topo_score", "muon_dif_score", "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "truth_gen_weight", "pred_accepted",
         "pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def paired(tail, main):
    tE, mE = f32(tail.deposited_energy.values), f32(main.truth_live_E.values)
    ti, pt, pm = 0, [], []
    for mi in range(len(main)):
        while ti < len(tail) and tE[ti] != mE[mi]:
            ti += 1
        if ti >= len(tail):
            break
        pt.append(ti); pm.append(mi); ti += 1
    if len(pt) < 0.999 * len(main):
        return None
    j = tail.iloc[pt].reset_index(drop=True)
    mm = main.iloc[pm].reset_index(drop=True)
    for c in mm.columns:
        j["m_" + c] = mm[c].values
    return j


def finish(j, w_gen=False):
    d = pd.DataFrame({
        "E": j.m_pred_positron_energy.values + j.m_pred_dead_energy.values,
        "acc": (j.m_pred_accepted.values >= 1),
        "t": j.m_pred_positron_time_consensus_ns.values,
        "muon": j.muon_score.values, "topo": j.topo_score.values,
        "mudif": j.muon_dif_score.values, "pidif": j.pion_dif_score.values,
    })
    d["w"] = j.m_truth_gen_weight.values if w_gen else 1.0
    return d


# ------------------------------------------------------------------ samples
rows = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
    g = ev.split("shard_")[-1].split("/")[0]
    tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS + ["truth_is_pie"])
    t = pd.read_parquet(tp, columns=TCOLS + ["is_pie"])
    if len(m) != len(t):
        continue
    j = paired(t, m)
    if j is not None:
        rows.append(finish(j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)]))
pie = pd.concat(rows, ignore_index=True)

rows, MIC_GW_ALL = [], 0.0
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t):
        continue
    j = paired(t, m)
    if j is None:
        continue
    et = j.m_truth_event_type.values.astype(np.int64)
    gwa = j.m_truth_gen_weight.values.astype(float)
    MIC_GW_ALL += gwa[np.isfinite(gwa)].sum()
    keep = ((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0) & np.isfinite(gwa)
    rows.append(finish(j[keep], w_gen=True))
mic = pd.concat(rows, ignore_index=True)

mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                     columns=TCOLS + ["is_pidif", "is_mudif"])
bkg = {}
for flag, mfile in (("is_pidif", "pidif_osl"), ("is_mudif", "mudif_osl")):
    tt = mt[mt[flag] == 1].reset_index(drop=True)
    m = pd.read_parquet(f"{P}/purity_eval/tail_osl/{mfile}_events.parquet", columns=MCOLS)
    bkg[flag] = finish(paired(tt, m))
pid, mud = bkg["is_pidif"], bkg["is_mudif"]
print(f"pie {len(pie):,}  michel {len(mic):,}  piDIF {len(pid):,}  muDIF {len(mud):,}", flush=True)

# ------------------------------------------------------------------ weights and stages
onset = np.quantile(pie.t.to_numpy()[pie.acc.to_numpy() & np.isfinite(pie.t.to_numpy())], 0.005)
TLO, THI = onset + 2.0, onset + 32.0


def acc_of(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def stages_of(d):
    a = acc_of(d)
    topo = a & (d.topo.to_numpy() > 0.99)
    muon = topo & (d.muon.to_numpy() < 0.02) & (d.mudif.to_numpy() < 0.46)
    pdv = muon & (d.pidif.to_numpy() < 0.01)
    tw = pdv & (d.t.to_numpy() >= TLO) & (d.t.to_numpy() <= THI)
    return [("(0) acceptance", a), ("(1) + topology $>0.99$", topo),
            ("(2) + muon veto", muon), ("(3) + pion-DIF veto", pdv),
            ("(4) + prompt time window", tw)]


Ep = pie.E.to_numpy()
acc_pie = acc_of(pie)
w_pie = TR.weights(len(pie), TR.BR_PIE)
w_mic = mic.w.to_numpy() * (TR.F_MIC_WIN / MIC_GW_ALL)
w_pid = TR.weights(len(pid), TR.R_PIDIF_ABS)
w_mud = TR.weights(len(mud), TR.R_MUDIF_AT_ACC * w_pie[acc_pie].sum(), mask=acc_of(mud))
TS = TR.tail_rescale(Ep[acc_pie])
print(f"[rates] ts = {TS:.3f}   prompt window [{TLO:.1f}, {THI:.1f}] ns", flush=True)

sp = stages_of(pie)
CH = [("$\\pi\\to\\mu\\to e$", mic, w_mic, C_MIC, 1.0),
      ("$\\pi$DIF", pid, w_pid, C_PID, TR.R_PIDIF_ABS),
      ("$\\mu$DIF", mud, w_mud, C_MUD, None)]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.6), sharey=True)
tab = {}
for ax, (name, d, w, col, _r) in zip(axes, CH):
    sb = stages_of(d)
    E = d.E.to_numpy()
    r0 = w[sb[0][1]].sum()
    ax.hist(E[sb[0][1]], bins=bins, weights=w[sb[0][1]], histtype="stepfilled",
            color="0.88", ec="0.6", label="at acceptance")
    ax.hist(E[sb[3][1]], bins=bins, weights=w[sb[3][1]], histtype="step", lw=1.9, color=col,
            label="+ topology and vetoes")
    ax.hist(E[sb[4][1]], bins=bins, weights=w[sb[4][1]], histtype="step", lw=1.5, ls="--",
            color=col, label="+ prompt window")
    ax.hist(Ep[sp[4][1]], bins=bins, weights=w_pie[sp[4][1]], histtype="step", lw=1.5,
            color=C_PIE, label=r"$\pi\to e$ (full chain)")
    ax.axvline(ES, color="k", ls=":", lw=0.9); ax.set_yscale("log")
    ax.set_ylim(3e-10, 3e-1); ax.set_xlim(0, EM)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("reconstructed energy [MeV]")
    st = []
    for (nm, mb), (_, mp) in zip(sb, sp):
        blo = w[mb & (E < ES)].sum(); plo = w_pie[mp & (Ep < ES)].sum()
        st.append(dict(stage=nm, bkg_events=int(mb.sum()), bkg_rate=float(w[mb].sum()),
                       contam_lo=float(TR.contamination(blo, plo, TS)),
                       pie_eff=float(100 * mp.sum() / max(sp[0][1].sum(), 1))))
    tab[name] = st
    n_topo, n_all = st[3]["bkg_events"], st[0]["bkg_events"]
    supp_topo = r0 / st[3]["bkg_rate"] if st[3]["bkg_rate"] > 0 else r0 / (CL90 * w.mean())
    supp_full = r0 / st[4]["bkg_rate"] if st[4]["bkg_rate"] > 0 else r0 / (CL90 * w.mean())
    extra = supp_full / supp_topo if supp_topo > 0 else 1.0
    lim = "" if n_topo else " (limit)"
    ax.text(0.97, 0.97,
            f"topology + vetoes:\n  {n_all:,} $\\to$ {n_topo:,} events\n"
            f"  suppression $>{supp_topo:,.0f}\\times${lim}\n"
            f"  $\\pi\\to e$ eff {st[3]['pie_eff']:.1f}%\n"
            f"+ prompt window:\n  $\\times${extra:,.1f} further\n"
            f"  $\\pi\\to e$ eff {st[4]['pie_eff']:.1f}%",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.4, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.93))
    if name == "$\\pi\\to\\mu\\to e$":
        ax.legend(fontsize=8.6, loc="upper left")
        ax.set_ylabel("rate per stopped pion / bin")
    print(f"\n== {name}", flush=True)
    for s in st:
        print(f"   {s['stage'][:28]:28} n {s['bkg_events']:>10,}  rate {s['bkg_rate']:.3e}  "
              f"contam {s['contam_lo']:6.2f}%  pie eff {s['pie_eff']:5.1f}%", flush=True)
    print(f"   topology+vetoes suppression > {supp_topo:,.0f}x ; prompt window adds x{extra:,.1f}", flush=True)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_cutflow_notime.png"); plt.close(fig)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["notime_cutflows"] = tab
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print("\nALL DONE", flush=True)
