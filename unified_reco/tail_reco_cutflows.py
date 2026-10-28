"""Thesis-style tail cutflows on the MAIN model's reconstructed quantities.

Same chains, weights, annotations, and page-format layouts as the truth-quantity cutflows of
tail_thesis_figures.py, with three swaps per event: the reconstructed acceptance decision for
the truth flag, the reconstructed positron energy plus the reconstructed dead-material
estimate for the true deposit, and the consensus positron time for the true time. The prompt
window is anchored data-driven at the reconstructed pie onset (+2 to +32 ns), so the time
frame convention of each generation cannot shift the cut.

Inputs (all positional joins, verified per piece by float32 equality of the tail predictions'
deposited_energy against the main summaries' truth_live_E):
  pie    : norad_eval2_rs tail predictions x purity_eval/nodrop_ep12_pie main summaries
  michel : michel_eval_inwin_rs tail predictions x purity_eval/nodrop_ep12_michel_inwin
           main summaries (RMD removed via truth_event_type bit 0x200, per analysis policy)
  piDIF  : eval10 eval-mode tail predictions x purity_eval/tail_osl pidif summaries

Outputs: figures/AI/tail_cutflow_michel_reco.png, tail_cutflow_pidif_reco.png, and the
'mudar_cutflow_reco' / 'pidif_cutflow_reco' blocks in figures/AI/tail_stats.json.
"""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
import tail_figstyle
tail_figstyle.apply()
C_PIE, C_MIC, C_PID = "#d62728", "#1f77b4", "#2ca02c"
ES, EM = 56.0, 90.0        # no upper energy cap in the analysis
import tail_rates as TR
BR_PIE, F_MIC_WIN = TR.BR_PIE, TR.F_MIC_WIN
KMUDIF, KRAD = 0x1000, 0x200
bins = np.linspace(0, EM, 61)
STATS = {}

TCOLS = ["muon_score", "pileup_score", "topo_score", "muon_dif_score", "pion_dif_score",
         "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "truth_gen_weight", "pred_accepted",
         "pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def paired(tail, main, label):
    """Positional pairing with a two-pointer resync on float32 live-energy equality."""
    tE, mE = f32(tail.deposited_energy.values), f32(main.truth_live_E.values)
    ti, pt, pm = 0, [], []
    for mi in range(len(main)):
        while ti < len(tail) and tE[ti] != mE[mi]:
            ti += 1
        if ti >= len(tail):
            break
        pt.append(ti); pm.append(mi); ti += 1
    if len(pt) < 0.999 * len(main):
        print(f"  [{label}] pairing failed ({len(pt)}/{len(main)}), SKIP", flush=True)
        return None
    j = tail.iloc[pt].reset_index(drop=True)
    mm = main.iloc[pm].reset_index(drop=True)
    for c in mm.columns:
        j["m_" + c] = mm[c].values
    return j


def finish(j, w_gen=False):
    d = pd.DataFrame({
        "Ereco": j.m_pred_positron_energy.values + j.m_pred_dead_energy.values,
        "acc": (j.m_pred_accepted.values >= 1),
        "t": j.m_pred_positron_time_consensus_ns.values,
        "muon": j.muon_score.values, "pileup": j.pileup_score.values,
        "topo": j.topo_score.values, "mudif": j.muon_dif_score.values,
        "pidif": j.pion_dif_score.values,
    })
    d["w"] = j.m_truth_gen_weight.values if w_gen else 1.0
    return d


# ------------------------------------------------------------------ load the three samples
pie_rows, mic_rows = [], []
MIC_GW_ALL = [0.0]
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
    g = ev.split("shard_")[-1].split("/")[0]
    tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS + ["truth_is_pie"])
    t = pd.read_parquet(tp, columns=TCOLS + ["is_pie"])
    if len(m) != len(t):
        continue
    j = paired(t, m, f"pie {g}")
    if j is None:
        continue
    j = j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)]
    pie_rows.append(finish(j))
pie = pd.concat(pie_rows, ignore_index=True)
print(f"pie: {len(pie_rows)} shards, {len(pie):,} rows", flush=True)

for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t):
        continue
    j = paired(t, m, f"mic {piece}")
    if j is None:
        continue
    et = j.m_truth_event_type.values.astype(np.int64)
    gw_all = j.m_truth_gen_weight.values.astype(float)
    gw_all = gw_all[np.isfinite(gw_all)]
    MIC_GW_ALL[0] += gw_all.sum()                     # denominator BEFORE the RMD removal,
    j = j[((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0)]   # so removed
    mic_rows.append(finish(j, w_gen=True))            # rate is lost, not redistributed
mic = pd.concat(mic_rows, ignore_index=True)
print(f"michel: {len(mic_rows)} pieces, {len(mic):,} rows (kPienu/kMudif/RMD removed)", flush=True)

mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                     columns=TCOLS + ["is_pidif"])
mt = mt[mt.is_pidif == 1].reset_index(drop=True)
mm = pd.read_parquet(f"{P}/purity_eval/tail_osl/pidif_osl_events.parquet", columns=MCOLS)
j = paired(mt, mm, "pidif")
pid = finish(j)
print(f"pidif: {len(pid):,} rows", flush=True)

# ------------------------------------------------------------------ common machinery
onset = np.quantile(pie.t.to_numpy()[pie.acc.to_numpy() & np.isfinite(pie.t.to_numpy())], 0.005)
TLO, THI = onset + 2.0, onset + 32.0
print(f"reco pie onset {onset:.2f} ns -> prompt window [{TLO:.1f}, {THI:.1f}] ns", flush=True)


TOPO_CUT = 0.99   # hardened pie-identity cut; the pileup veto is dropped, subsumed by it


def masks(d):
    E = d.Ereco.to_numpy()
    ok = d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)
    tw = (d.t.to_numpy() >= TLO) & (d.t.to_numpy() <= THI)
    return E, dict(
        acc=ok,
        time=ok & tw,
        topo=ok & tw & (d.topo.to_numpy() > TOPO_CUT),
        muon=ok & tw & (d.topo.to_numpy() > TOPO_CUT)
             & (d.muon.to_numpy() < 0.02) & (d.mudif.to_numpy() < 0.46),
        pdveto=ok & tw & (d.pidif.to_numpy() < 0.01),
        full_pidif=ok & tw & (d.pidif.to_numpy() < 0.01) & (d.muon.to_numpy() < 0.02)
                   & (d.mudif.to_numpy() < 0.46) & (d.topo.to_numpy() > TOPO_CUT),
    )


Ep, mp = masks(pie)
wpie_v = TR.weights(len(pie), BR_PIE)                    # per stopped pion, pre-selection
TS = TR.tail_rescale(Ep[mp["acc"]])
print(f"[rates] piDIF/pie = {TR.PIDIF_OVER_PIE:.2f}  tail rescale ts = {TS:.3f}", flush=True)


def cutflow(dbkg, stages, wnorm, bkg_color, bkg_label, grid, out, statkey, at_acceptance=False):
    """`wnorm` is the channel rate; imposed over accepted rows if the external input is
    quoted at acceptance (muDIF), else over all rows (michel, piDIF)."""
    Eb, mb = masks(dbkg)
    wb = TR.weights(len(dbkg), wnorm, gw=dbkg.w.to_numpy(),
                    mask=mb["acc"] if at_acceptance else None)
    nrow, ncol = grid
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol + 0.1, 5.4 * nrow), sharex=True, sharey=True)
    axes = axs.ravel()[:len(stages)]
    for a in axs.ravel()[len(stages):]:
        a.axis("off")
    TF0 = None; stab = []
    for i, (ax, (key, title)) in enumerate(zip(axes, stages)):
        pk, bk = mp[key], mb[key]
        ax.hist(Ep[pk], bins=bins, weights=wpie_v[pk], histtype="step", lw=1.6, color=C_PIE,
                label=r"$\pi\to e$" if i == 0 else None)
        ax.hist(Eb[bk], bins=bins, weights=wb[bk], histtype="step", lw=1.8, color=bkg_color,
                label=bkg_label if i == 0 else None)
        ax.axvline(ES, color="k", ls=":", lw=0.9); ax.set_yscale("log")
        ax.set_ylim(3e-10, 3e-1)
        ax.set_title(title, fontsize=11)
        blo = wb[bk & (Eb < ES)].sum(); plo = wpie_v[pk & (Ep < ES)].sum()
        bhi = wb[bk & (Eb >= ES)].sum(); phi = wpie_v[pk & (Ep >= ES)].sum()
        clo = TR.contamination(blo, plo, TS)
        chi = 100 * bhi / (bhi + phi) if (bhi + phi) > 0 else 0.0
        peff = 100 * pk.sum() / max(mp["acc"].sum(), 1)
        tf = 100 * (Ep[pk] < ES).sum() / max((Ep[pk] >= ES).sum(), 1)
        if TF0 is None and key == "time":
            TF0 = tf
        bias = 100 * (tf - TF0) / TF0 if TF0 else 0.0
        ax.text(0.96, 0.96, f"contam ($E\\geq$56): {chi:.2f}%\ncontam ($E<$56): {clo:.1f}%\n"
                            f"$\\pi\\to e$ eff: {peff:.1f}%\ntail bias: {bias:+.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92))
        if i == 0:
            ax.legend(fontsize=9, loc="upper left")
        stab.append(dict(stage=title, contam_lo=clo, contam_hi=chi, pie_eff=peff,
                         tail_bias=bias, bkg_events=int(bk.sum())))
    for a in axs[-1] if nrow > 1 else [axes[-1]]:
        if a.axison:
            a.set_xlabel("reconstructed energy [MeV]")
    if nrow > 1 and len(stages) < nrow * ncol:
        axs.ravel()[ncol - 1].tick_params(labelbottom=True)
        axs.ravel()[ncol - 1].set_xlabel("reconstructed energy [MeV]")
    for r in range(nrow):
        axs[r][0].set_ylabel("branching-weighted rate [a.u.]") if ncol > 1 else None
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    STATS[statkey] = stab
    print(f"wrote {out}", flush=True)
    for s in stab:
        print(f"  {s['stage'][:36]:36} contam_hi {s['contam_hi']:6.2f}%  contam_lo {s['contam_lo']:5.1f}%  "
              f"evts {s['bkg_events']:>9,}  pie eff {s['pie_eff']:5.1f}%  bias {s['tail_bias']:+.2f}%", flush=True)


cutflow(mic,
        [("acc", "(0) after acceptance"), ("time", "(1) + prompt time window"),
         ("topo", "(2) + $\\pi\\to e$ topology $>0.99$"), ("muon", "(3) + muon veto")],
        F_MIC_WIN * (mic.w.to_numpy().sum() / MIC_GW_ALL[0]), C_MIC, r"$\pi\to\mu\to e$", (2, 2),
        f"{OUT}/tail_cutflow_michel_reco.png", "mudar_cutflow_reco")

cutflow(pid,
        [("acc", "(0) acceptance"), ("time", "(1) + prompt time window"),
         ("pdveto", "(2) + pion-DIF veto $<0.01$"), ("full_pidif", "(3) + all $\\mu$DAR cuts")],
        TR.R_PIDIF_ABS, C_PID, r"$\pi$DIF", (2, 2),
        f"{OUT}/tail_cutflow_pidif_reco.png", "pidif_cutflow_reco")

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats.update(STATS)
allstats["reco_cutflow_meta"] = dict(window_ns=[float(TLO), float(THI)], onset_ns=float(onset),
                                     pie_rows=int(len(pie)), michel_rows=int(len(mic)),
                                     pidif_rows=int(len(pid)))
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print("ALL DONE", flush=True)
