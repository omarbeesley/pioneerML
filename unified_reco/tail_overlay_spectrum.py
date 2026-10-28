"""Summary overlay: all three backgrounds and the signal in the reconstructed energy
spectrum, before and after the retuned tail chain.

Both panels weight every channel to its physical rate per stopped pion, so the curves may be
read against each other directly:
  pie    : the branching ratio, 1.2352e-4, over the whole generated sample
  michel : the in-window muon-decay-at-rest rate, 0.194 per stopped pion, gen-weighted
  piDIF  : three per pie in the window (external rate input), over the whole sample
  muDIF  : one per twenty pie AT ACCEPTANCE (external rate input), the convention in which
           that ratio is quoted
Left panel: after the reconstructed acceptance. Right panel: after the full retuned chain
(prompt window, pie topology > 0.99, muon veto, pion-DIF veto). Channels that reach zero
survivors are quoted as 90% CL upper limits (2.3 events) in the annotation.

Outputs figures/AI/tail_overlay_spectrum.png and the 'overlay_spectrum' block of
figures/AI/tail_stats.json.
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
import tail_figstyle
tail_figstyle.apply()
C_PIE, C_MIC, C_PID, C_MUD = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
ES, EM = 56.0, 90.0        # no upper energy cap in the analysis; show the pileup-inflated region
import tail_rates as TR
BR_PIE, F_MIC_WIN = TR.BR_PIE, TR.F_MIC_WIN
KMUDIF, KRAD = 0x1000, 0x200
bins = np.linspace(0, EM, 61)      # 1.5 MeV bins
CL90 = 2.3          # 90% CL upper limit on a Poisson mean for zero observed events

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
    if j is None:
        continue
    rows.append(finish(j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)]))
pie = pd.concat(rows, ignore_index=True)

rows = []
MIC_GW_ALL = [0.0]
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
    MIC_GW_ALL[0] += gwa[np.isfinite(gwa)].sum()      # denominator before the RMD removal, so the
    keep = ((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0) & np.isfinite(gwa)
    rows.append(finish(j[keep], w_gen=True))          # removed radiative rate is lost, not redistributed
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

# ------------------------------------------------------------------ masks
onset = np.quantile(pie.t.to_numpy()[pie.acc.to_numpy() & np.isfinite(pie.t.to_numpy())], 0.005)
TLO, THI = onset + 2.0, onset + 32.0
print(f"prompt window [{TLO:.1f}, {THI:.1f}] ns", flush=True)


def acc_mask(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def time_mask(d):
    tt = d.t.to_numpy()
    return acc_mask(d) & (tt >= TLO) & (tt <= THI)


def full_mask(d):
    t = d.t.to_numpy()
    return acc_mask(d) & (t >= TLO) & (t <= THI) & (d.topo.to_numpy() > 0.99) \
        & (d.muon.to_numpy() < 0.02) & (d.mudif.to_numpy() < 0.46) \
        & (d.pidif.to_numpy() < 0.01)


# ------------------------------------------------------------------ physical weights
acc_pie, acc_mud = acc_mask(pie), acc_mask(mud)
w_pie = TR.weights(len(pie), BR_PIE)                        # BR per stopped pion, pre-selection
w_mic = mic.w.to_numpy() * (F_MIC_WIN / MIC_GW_ALL[0])      # in-window rate, radiative share removed
w_pid = TR.weights(len(pid), TR.R_PIDIF_ABS)                # 1e-3 per stopped pion = 8.1x pie
w_mud = TR.weights(len(mud), TR.R_MUDIF_AT_ACC * w_pie[acc_pie].sum(), mask=acc_mud)  # 20:1 at acceptance
TS = TR.tail_rescale(pie.E.to_numpy()[acc_pie])             # simulated pie tail -> physical 0.6%
print(f"[rates] piDIF/pie = {TR.PIDIF_OVER_PIE:.2f}  tail rescale ts = {TS:.3f}", flush=True)

CH = [("$\\pi\\to e$", pie, w_pie, C_PIE, 1.9),
      ("$\\pi\\to\\mu\\to e$", mic, w_mic, C_MIC, 1.6),
      ("$\\mu$DIF", mud, w_mud, C_MUD, 1.6),
      ("$\\pi$DIF", pid, w_pid, C_PID, 1.6)]

# ---- diagnostic: is the piDIF spectrum shape real, or a sample/join artifact? ----------
for nm, d, w in (("piDIF", pid, w_pid), ("muDIF", mud, w_mud), ("michel", mic, w_mic)):
    k = acc_mask(d); E = d.E.to_numpy()
    n_unique = len(np.unique(np.round(E[k], 4)))
    q = np.percentile(E[k], [5, 25, 50, 75, 95]) if k.sum() else [np.nan] * 5
    print(f"[diag] {nm:7} accepted {k.sum():>9,}  unique E {n_unique:>9,} "
          f"({100*n_unique/max(k.sum(),1):5.1f}%)  E q05/25/50/75/95 = "
          + "/".join(f"{v:.1f}" for v in q)
          + f"  frac>56 {100*(E[k]>=ES).mean():5.2f}%", flush=True)

Epie = pie.E.to_numpy()
k_time, k_full = time_mask(pie), full_mask(pie)
rlo, rhi = (k_time & (Epie < ES)).sum(), (k_time & (Epie >= ES)).sum()
slo, shi = (k_full & (Epie < ES)).sum(), (k_full & (Epie >= ES)).sum()
tf_time, tf_full = rlo / max(rhi, 1), slo / max(shi, 1)
el, eh = slo / max(rlo, 1), shi / max(rhi, 1)
bias = 100 * (tf_full / tf_time - 1) if tf_time else 0.0
bias_err = 100 * (tf_full / tf_time) * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
TF = {"acceptance": 100 * (k_acc_pie_lo := (acc_pie & (Epie < ES)).sum()) / max((acc_pie & (Epie >= ES)).sum(), 1),
      "full_chain": 100 * tf_full}
BIAS = {"acceptance": None, "full_chain": (bias, bias_err)}
print(f"[bias] pie tail fraction: acceptance {TF['acceptance']:.3f}%  full chain {TF['full_chain']:.3f}%  "
      f"bias vs post-time {bias:+.2f} +/- {bias_err:.2f}%", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 8.6), sharey=True)
stats = {}
for ax, (title, mask_fn, tag) in zip(axes, [
        ("after the reconstructed acceptance", acc_mask, "acceptance"),
        ("after the full retuned chain", full_mask, "full_chain")]):
    lows, block = {}, []
    for name, d, w, col, lw in CH:
        k = mask_fn(d)
        E = d.E.to_numpy()
        if k.sum():
            ax.hist(E[k], bins=bins, weights=w[k], histtype="step", lw=lw, color=col, label=name)
        lo = w[k & (E < ES)].sum()
        if name == "$\\pi\\to e$":
            lo *= TS                                        # physical tail, per tail_rates
        lows[name] = float(lo)
        if name != "$\\pi\\to e$" and k.sum() == 0:
            # 90% CL upper limit for zero observed: 2.3 events, each carrying the mean weight
            # of the population that could have survived (events inside the prompt window)
            k_ref = time_mask(d)
            w_ref = w[k_ref].mean() if k_ref.sum() else w.mean()
            lows[name] = float(CL90 * w_ref)
            block.append((name, col, 0, lows[name], True))
        else:
            block.append((name, col, int(k.sum()), float(lo), False))
    tot_lo = sum(v for (nm, _c, _n, v, u) in block if not u)      # observed channels only
    ax.axvline(ES, color="k", ls=":", lw=0.9)
    ax.set_yscale("log"); ax.set_xlabel("reconstructed energy [MeV]")
    ax.set_title(title, fontsize=11.5)
    txt = []
    for name, col, n, lo, ul in block:
        if ul:
            txt.append(f"{name}: $<${lo:.1e}  (90% CL)")
            continue
        frac = 100 * lo / tot_lo if tot_lo > 0 else 0.0
        fs = f"{frac:.1f}%" if frac >= 1 else f"{frac:.2g}%"
        txt.append(f"{name}: {lo:.2e}  ({fs})")
    txt.append("")
    txt.append(f"$\\pi\\to e$ tail fraction: {TF[tag]:.2f}%")
    if BIAS[tag] is not None:
        txt.append(f"tail-fraction bias: ${BIAS[tag][0]:+.1f} \\pm {BIAS[tag][1]:.1f}$%")
    ax.text(0.975, 0.975, "low-bin rate ($E<56$ MeV):\n" + "\n".join(txt), transform=ax.transAxes,
            va="top", ha="right", fontsize=8.2, family="serif", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.93))
    stats[tag] = {n: dict(events=int(nn), low_bin_rate=float(l), upper_limit=bool(u))
                  for n, _, nn, l, u in block}
for ax in axes:
    ax.set_ylim(3e-10, 3e-1)         # trimmed to the populated decades, with box headroom
    ax.set_xlim(0, EM)
axes[0].set_ylabel("rate per stopped pion / bin")
axes[0].legend(fontsize=9.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/tail_overlay_spectrum.png"); plt.close(fig)
print(json.dumps(stats, indent=1), flush=True)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["overlay_spectrum"] = dict(window=[float(TLO), float(THI)],
                                    pie_tail_fraction_pct=TF, tail_bias_pct=list(BIAS["full_chain"]),
                                    **stats)
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print("ALL DONE", flush=True)
