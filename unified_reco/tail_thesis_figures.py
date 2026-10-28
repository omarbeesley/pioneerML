"""Thesis-style tail-reveal figures for the AI chapter, generated from the settled eval parquets.

Ports the SETTLED analysis logic verbatim from:
  cutflow_settled.py   (muDAR 5-panel chain, gen-weighted michel, kPienu+kMudif removed)
  cutflow_pidif.py     (piDIF 4-panel chain, absolute [6,36] ns window on norad/eval10 samples)
  dif_mudif_corrected.py (muDIF suppression spectrum + delta-ray-corrected tail-fraction sweep)
  mudif_tradeoff.py    (bias vs contamination working-point curve, pie5M Ecorr, 20:1 norm)
  mudif_surv_analysis.py (survivor muon E-deposition / decay-KE, suppression vs angle)
and renders in the thesis figure style (figures/ml_pedagogy/figstyle.py conventions).
Outputs PNGs to thesis_ref/figures/AI/ and a stats JSON alongside.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches

ap = argparse.ArgumentParser()
ap.add_argument("--skip-cutflows", action="store_true",
                help="Skip the michel/pidif cutflows (their input pools await eval-mode re-evaluation).")
args = ap.parse_args()

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
# eval-mode (settled-convention) predictions; the plain predictions.parquet of Jul 25 ran with
# dropout active and is invalid
EV10 = f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet"
OUT = "/home/obeesley/thesis_ref/figures/AI"
os.makedirs(OUT, exist_ok=True)

# thesis figure style (figstyle.py conventions: serif + CM mathtext, no system LaTeX)
import tail_figstyle
tail_figstyle.apply()
C_PIE, C_MIC, C_PID, C_MUD = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
STATS = {}
RNG = np.random.default_rng(20260728)

import tail_rates as TR
BR_PIE, F_MIC_WIN = TR.BR_PIE, TR.F_MIC_WIN
KMUDIF = 0x1000
ES, EM = 56.0, 75.0
bins = np.linspace(0, EM, 51)
R_PIDIF, R_PIE = TR.R_PIDIF_ABS, TR.BR_PIE   # 8.1:1 per stopped pion, pre-selection


def auc_rank(sig, bkg):
    """Mann-Whitney AUC: P(score_sig > score_bkg), ties get half credit."""
    s = np.concatenate([sig, bkg])
    r = pd.Series(s).rank().to_numpy()
    ns, nb = len(sig), len(bkg)
    return (r[:ns].sum() - ns * (ns + 1) / 2) / (ns * nb)


def roc_curve(sig, bkg, n=400):
    thr = np.quantile(np.concatenate([sig, bkg]), np.linspace(0, 1, n))
    se = [(sig >= t).mean() for t in thr]
    be = [(bkg >= t).mean() for t in thr]
    return np.array(se), np.array(be)


# ---------------------------------------------------------------- sample loaders
GW_ALL = [0.0]


def load_michel(nmax=int(os.environ.get("TAIL_MICHEL_NMAX", "60"))):
    cols = ["muon_score", "pileup_score", "topo_score", "pie_score", "muon_dif_score",
            "pion_dif_score", "acceptance", "positron_t", "deposited_energy"]
    rows, used = [], 0
    for pf in sorted(glob.glob(f"{P}/michel_eval_inwin_rs/*/predictions.parquet")):
        if used >= nmax: break
        piece = pf.split("/")[-2]; inp = f"{P}/michel_inwin/{piece}.parquet"
        if not os.path.exists(inp): continue
        t = pq.read_table(inp, columns=["event_type", "live_E", "gen_weight"]).to_pandas()
        t["k"] = np.round(t.live_E.values.astype(np.float32).astype(np.float64), 4)
        g = t.groupby("k").agg(nun=("event_type", "nunique"), et=("event_type", "first"),
                               gw=("gen_weight", "first"))
        good = g[g["nun"] == 1]
        pr = pd.read_parquet(pf, columns=cols); pr = pr[pr.acceptance == 1].copy()
        key = np.round(pr.deposited_energy.values.astype(np.float32).astype(np.float64), 4)
        et = pd.Series(key, index=pr.index).map(good["et"])
        pr["w_gen"] = pd.Series(key, index=pr.index).map(good["gw"]).values
        ei = et.fillna(-1).astype("int64")
        keep = et.notna() & ((ei & 1) == 0) & ((ei & KMUDIF) == 0) & ((ei & 0x200) == 0) & pr.w_gen.notna()   # 0x200 = radiative decay, excluded by analysis policy
        GW_ALL[0] += pr.w_gen[et.notna() & pr.w_gen.notna()].sum()    # denominator before the removal
        rows.append(pr[keep]); used += 1
    d = pd.concat(rows, ignore_index=True)
    print(f"michel: {len(d):,} pure accepted ({used} pieces)", flush=True)
    return d


def load_pie_rs(cols):
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in
                   sorted(glob.glob(f"{P}/norad_eval2_rs/ev_*/predictions.parquet"))],
                  ignore_index=True)
    print(f"pie_rs: {len(d):,}", flush=True)
    return d


# =========================================================== 1) settled muDAR cutflow
print("== muDAR cutflow ==", flush=True)
SKIP = args.skip_cutflows
if SKIP:
    print("  skipped (pools await eval-mode re-evaluation)", flush=True)
if not SKIP:
    dm = load_michel()
    dp = load_pie_rs(["muon_score", "pileup_score", "topo_score", "pie_score", "muon_dif_score",
                      "acceptance", "positron_t", "deposited_energy", "is_pie"])
    dp = dp[(dp.is_pie == 1) & (dp.acceptance == 1)].reset_index(drop=True)
    dm = dm[dm.deposited_energy <= EM].reset_index(drop=True)
    dp = dp[dp.deposited_energy <= EM].reset_index(drop=True)
    Em_, Ep_ = dm.deposited_energy.values, dp.deposited_energy.values
    wmic = dm.w_gen.values * (F_MIC_WIN / GW_ALL[0])      # radiative share leaves with its events
    wpie = np.full(len(dp), BR_PIE / len(dp))
    TS = TR.tail_rescale(Ep_)
    print(f"[rates] piDIF/pie = {TR.PIDIF_OVER_PIE:.2f}  tail rescale ts = {TS:.3f}", flush=True)
    bins = np.linspace(0, EM, 51)

    STAGES_M = [("(0) after acceptance", lambda d: np.ones(len(d), bool)),
                ("(1) + time $[2,32]$ ns", lambda d: (d.positron_t.values >= 2) & (d.positron_t.values <= 32)),
                ("(2) + pileup veto $<0.5$", lambda d: d.pileup_score.values < 0.5),
                ("(3) + $\\pi\\to e$ topology $>0.9$", lambda d: d.topo_score.values > 0.9),
                ("(4) + muon veto", lambda d: (d.muon_score.values < 0.02) & (d.muon_dif_score.values < 0.46))]

    fig, axs2 = plt.subplots(2, 3, figsize=(13.0, 10.6), sharex=True, sharey=True)
    axes = axs2.ravel()[:5]
    axs2.ravel()[5].axis("off")
    mm = np.ones(len(dm), bool); pm = np.ones(len(dp), bool)
    TF0 = None; stab = []
    pm_time = None
    for j, (ax, (name, expr)) in enumerate(zip(axes, STAGES_M)):
        mm = mm & expr(dm); pm = pm & expr(dp)
        if j == 1: pm_time = pm.copy()
        ax.hist(Ep_[pm], bins=bins, weights=wpie[pm], histtype="step", lw=1.6, color=C_PIE,
                label=r"$\pi\to e$" if j == 0 else None)
        ax.hist(Em_[mm], bins=bins, weights=wmic[mm], histtype="step", lw=1.8, color=C_MIC,
                label=r"$\pi\to\mu\to e$" if j == 0 else None)
        ax.axvline(ES, color="k", ls=":", lw=0.9); ax.set_yscale("log")
        ax.set_ylim(3e-10, 3e-1)
        ax.set_title(name, fontsize=11)
        mic_lo = wmic[mm & (Em_ < ES)].sum(); pie_lo = wpie[pm & (Ep_ < ES)].sum()
        mic_hi = wmic[mm & (Em_ >= ES)].sum(); pie_hi = wpie[pm & (Ep_ >= ES)].sum()
        blo = TR.contamination(mic_lo, pie_lo, TS)
        bhi = 100 * mic_hi / (mic_hi + pie_hi) if (mic_hi + pie_hi) > 0 else 0.0
        peff = 100 * pm.sum() / len(dp)
        tf = 100 * (Ep_[pm] < ES).sum() / max((Ep_[pm] >= ES).sum(), 1)
        if TF0 is None and j == 1: TF0 = tf
        bias = 100 * (tf - TF0) / TF0 if TF0 else 0.0
        ax.text(0.96, 0.96, f"contam ($E\\geq$56): {bhi:.2f}%\ncontam ($E<$56): {blo:.1f}%\n"
                            f"$\\pi\\to e$ eff: {peff:.1f}%\ntail bias: {bias:+.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92))
        if j == 0: ax.legend(fontsize=9, loc="upper left")
        stab.append(dict(stage=name, michel_events=int(mm.sum()), michel_contam_lo=blo,
                         michel_contam_hi=bhi, pie_eff=peff, tail_bias=bias))
    for a in (axes[2], axes[3], axes[4]):
        a.set_xlabel("deposited energy [MeV]")
    axes[2].tick_params(labelbottom=True)
    for a in (axes[2], axes[3]):
        a.set_xlabel("deposited energy [MeV]")
    axes[0].set_ylabel("branching-weighted rate [a.u.]")
    axes[2].set_ylabel("branching-weighted rate [a.u.]")
    axes[3].set_ylabel("branching-weighted rate [a.u.]")
    fig.tight_layout(); fig.savefig(f"{OUT}/tail_cutflow_michel.png"); plt.close(fig)

    # bootstrap uncertainty on the final tail bias (resample pie events; ref = after time cut)
    lo_t = (Ep_ < ES) & pm_time; hi_t = (Ep_ >= ES) & pm_time
    lo_f = (Ep_ < ES) & pm; hi_f = (Ep_ >= ES) & pm
    idx = np.arange(len(dp)); boots = []
    for _ in range(400):
        b = RNG.choice(idx, len(idx), replace=True)
        t_lo, t_hi = lo_t[b].sum(), hi_t[b].sum(); f_lo, f_hi = lo_f[b].sum(), hi_f[b].sum()
        if min(t_hi, f_hi, t_lo) > 0:
            boots.append(100 * ((f_lo / f_hi) / (t_lo / t_hi) - 1))
    STATS["mudar_cutflow"] = dict(
        stages=stab, michel_sample=int(len(dm)), pie_sample=int(len(dp)),
        final_bias_pct=stab[-1]["tail_bias"], final_bias_unc_pct=float(np.std(boots)),
        final_michel_survivors=int(mm.sum()))
    print(json.dumps(STATS["mudar_cutflow"], indent=1), flush=True)

    # =========================================================== 2) piDIF cutflow
    print("== piDIF cutflow ==", flush=True)
    dd = pd.read_parquet(EV10,
                         columns=["is_pidif", "muon_score", "muon_dif_score", "topo_score",
                                  "pileup_score", "pion_dif_score", "deposited_energy",
                                  "acceptance", "positron_t"])
    dd = dd[dd.is_pidif == 1].reset_index(drop=True)
    pie2 = load_pie_rs(["muon_score", "muon_dif_score", "topo_score", "pileup_score",
                        "pion_dif_score", "deposited_energy", "acceptance", "positron_t", "is_pie"])
    pie2 = pie2[pie2.is_pie == 1].reset_index(drop=True)
    Ep2 = pie2.deposited_energy.values.astype(float); Ed2 = dd.deposited_energy.values.astype(float)
    k1 = np.isfinite(Ep2) & (Ep2 >= 0) & (Ep2 <= EM); k2 = np.isfinite(Ed2) & (Ed2 >= 0) & (Ed2 <= EM)
    pie2 = pie2[k1].reset_index(drop=True); Ep2 = Ep2[k1]
    dd = dd[k2].reset_index(drop=True); Ed2 = Ed2[k2]
    pacc = pie2.acceptance.values >= 1; dacc = dd.acceptance.values >= 1
    TW_LO, TW_HI = 6.0, 36.0    # absolute window = pion-stop-referenced [2,32] ns (stop ~4 ns)
    twin = (pie2.positron_t.values >= TW_LO) & (pie2.positron_t.values <= TW_HI)
    dwin = (dd.positron_t.values >= TW_LO) & (dd.positron_t.values <= TW_HI)
    wpie2 = R_PIE / len(pie2); wdif2 = R_PIDIF / len(dd)
    mu_p = (pie2.muon_score.values < 0.02) & (pie2.muon_dif_score.values < 0.46)
    mu_d = (dd.muon_score.values < 0.02) & (dd.muon_dif_score.values < 0.46)
    cl_p = (pie2.pileup_score.values < 0.5) & (pie2.topo_score.values > 0.9)
    cl_d = (dd.pileup_score.values < 0.5) & (dd.topo_score.values > 0.9)
    pd_p = pie2.pion_dif_score.values < 0.01; pd_d = dd.pion_dif_score.values < 0.01

    STAGES_P = [("(0) acceptance", pacc, dacc),
                ("(1) + time $[2,32]$ ns", pacc & twin, dacc & dwin),
                ("(2) + pion-DIF veto $<0.01$", pacc & twin & pd_p, dacc & dwin & pd_d),
                ("(3) + all $\\mu$DAR cuts", pacc & twin & pd_p & mu_p & cl_p,
                 dacc & dwin & pd_d & mu_d & cl_d)]
    fig, axs2 = plt.subplots(2, 2, figsize=(11.0, 10.6), sharex=True, sharey=True)
    axes = axs2.ravel()
    TF0 = None; stab = []
    for j, (ax, (name, pmask, dmask)) in enumerate(zip(axes, STAGES_P)):
        ax.hist(Ep2[pmask], bins=bins, weights=np.full(pmask.sum(), wpie2), histtype="step",
                lw=1.6, color=C_PIE, label=r"$\pi\to e$" if j == 0 else None)
        ax.hist(Ed2[dmask], bins=bins, weights=np.full(dmask.sum(), wdif2), histtype="step",
                lw=1.8, color=C_PID, label=r"$\pi$DIF" if j == 0 else None)
        ax.axvline(ES, color="k", ls=":", lw=0.9); ax.set_yscale("log")
        ax.set_ylim(3e-10, 3e-1)
        ax.set_title(name, fontsize=11)
        plo = wpie2 * (pmask & (Ep2 < ES)).sum(); phi = wpie2 * (pmask & (Ep2 >= ES)).sum()
        dlo = wdif2 * (dmask & (Ed2 < ES)).sum(); dhi = wdif2 * (dmask & (Ed2 >= ES)).sum()
        blo = TR.contamination(dlo, plo, TS)
        bhi = 100 * dhi / (dhi + phi) if (dhi + phi) > 0 else 0.0
        peff = 100 * pmask.sum() / len(pie2)
        tf = 100 * (Ep2[pmask] < ES).sum() / max((Ep2[pmask] >= ES).sum(), 1)
        if TF0 is None and j == 1: TF0 = tf
        bias = 100 * (tf - TF0) / TF0 if TF0 else 0.0
        bx, bha = (0.96, "right") if j == 0 else (0.04, "left")
        ax.text(bx, 0.96, f"contam ($E\\geq$56): {bhi:.2f}%\ncontam ($E<$56): {blo:.1f}%\n"
                          f"$\\pi\\to e$ eff: {peff:.1f}%\ntail bias: {bias:+.1f}%",
                transform=ax.transAxes, va="top", ha=bha, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92))
        if j == 0: ax.legend(fontsize=9, loc="upper left")
        stab.append(dict(stage=name, pidif_contam_lo=blo, pidif_contam_hi=bhi,
                         pie_eff=peff, tail_bias=bias, pidif_events=int(dmask.sum())))
    axes[0].set_ylabel("branching-weighted rate [a.u.]")
    fig.tight_layout(); fig.savefig(f"{OUT}/tail_cutflow_pidif.png"); plt.close(fig)
    STATS["pidif_cutflow"] = dict(stages=stab, pidif_sample=int(len(dd)), pie_sample=int(len(pie2)))
    print(json.dumps(STATS["pidif_cutflow"], indent=1), flush=True)
    del pie2, dd

# =========================================================== 3+4) muDIF spectrum & trade-off
print("== muDIF spectrum / trade-off ==", flush=True)
RATIO, REAL_TAIL = 20.0, 0.006
pie5 = pd.concat([pd.read_parquet(f"{P}/norad_joined_v2.parquet"),
                  pd.read_parquet(f"{P}/norad_joined2_v2.parquet")], ignore_index=True)
Ec = (pie5.deposited_energy + pie5.dead_E + pie5.Epure).to_numpy()
pa5 = ((pie5.acceptance >= 1) & (pie5.is_pie == 1)).to_numpy() & np.isfinite(Ec) & (Ec <= EM)
psc5 = pie5.muon_dif_score.to_numpy()
md = pd.read_parquet(EV10,
                     columns=["is_mudif", "muon_dif_score", "deposited_energy", "dead_E",
                              "acceptance", "positron_theta", "muon_decay_ke"])
md = md[md.is_mudif == 1].reset_index(drop=True)
Emd = (md.deposited_energy + md.dead_E).to_numpy()
ma = (md.acceptance >= 1).to_numpy() & np.isfinite(Emd) & (Emd <= EM)
msc = md.muon_dif_score.to_numpy()
cutq = lambda e: np.quantile(psc5[pa5], e / 100.0)

EFFS = [99, 90, 50, 30, 10]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[2, 1]))
axL.hist(Emd[ma], bins=bins, histtype="stepfilled", color="0.88", ec="0.6", zorder=0,
         label="no veto")
cmap = plt.get_cmap("Purples")
cval = lambda e: cmap(0.35 + 0.55 * (1 - e / 100.0))
wp_tab = []
for e in EFFS:
    surv = ma & (msc < cutq(e))
    axL.hist(Emd[surv], bins=bins, histtype="step", lw=1.8, color=cval(e), zorder=3)
    wp_tab.append(dict(pie_eff=e, suppression=float(ma.sum() / max(surv.sum(), 1)),
                       mudif_events=int(surv.sum())))
axL.axvline(ES, color="k", ls=":", lw=0.9); axL.set_yscale("log")
axL.set_xlabel("deposited energy [MeV]"); axL.set_ylabel("accepted $\\mu$DIF events / bin")
cb_cmap = colors.LinearSegmentedColormap.from_list("cb", [cval(e) for e in np.linspace(10, 99, 256)])
sm = cm.ScalarMappable(norm=colors.Normalize(10, 99), cmap=cb_cmap)
fig.colorbar(sm, ax=axL, pad=0.01).set_label(r"$\pi\to e$ efficiency [%]")
thr = np.quantile(psc5[pa5], np.linspace(0.005, 1.0, 200))
ef, tf_ = [], []
for t in thr:
    k = pa5 & (psc5 < t)
    if k.sum() == 0: continue
    ef.append(100.0 * k.sum() / pa5.sum()); tf_.append((Ec[k] < ES).sum() / k.sum())
ef, tf_ = np.array(ef), np.array(tf_)
tf_ = tf_ / tf_[np.argmax(ef)]
keep = ef >= 5.0
axR.axhline(1.0, color="0.6", ls="--", lw=1.0)
axR.plot(ef[keep], tf_[keep], "-", color=C_MUD, lw=2.0)
axR.set_xlabel(r"$\pi\to e$ efficiency [%]"); axR.set_ylabel("relative $\\pi\\to e$ tail fraction")
axR.invert_xaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/tail_mudif_spectrum.png"); plt.close(fig)
STATS["mudif_working_points"] = wp_tab
print(json.dumps(wp_tab, indent=1), flush=True)

# trade-off: bias vs contamination (corrected energy, 20:1 acceptance norm, real 0.6% tail)
wpie5 = R_PIE / pa5.sum(); wdif5 = (R_PIE / RATIO) / ma.sum()
ts = REAL_TAIL / ((Ec[pa5] < ES).sum() / pa5.sum())
rlo = (pa5 & (Ec < ES)).sum(); rhi = (pa5 & (Ec >= ES)).sum()
effs = [99, 95, 90, 80, 70, 50, 30]
X, Y, YE, tab = [], [], [], []
for e in effs:
    c = cutq(e)
    pmk = pa5 & (psc5 < c); mmk = ma & (msc < c)
    plo = ts * wpie5 * (pmk & (Ec < ES)).sum(); mlo = wdif5 * (mmk & (Emd < ES)).sum()
    contam = 100 * mlo / (mlo + plo)
    slo = (pmk & (Ec < ES)).sum(); shi = (pmk & (Ec >= ES)).sum()
    el, eh = slo / rlo, shi / rhi; r = el / eh
    err = 100 * r * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
    X.append(contam); Y.append(100 * (r - 1)); YE.append(err)
    tab.append(dict(pie_eff=e, contam_pct=float(contam), bias_pct=float(100 * (r - 1)),
                    bias_unc_pct=float(err)))
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.add_patch(patches.Rectangle((0, -1), 1, 2, fc="#2ca02c", alpha=0.13, ec="none", zorder=0))
ax.text(1.15, -0.9, "target:\n$<1\\%$ contam, $|$bias$|<1\\%$", fontsize=9, color="#1a7a1a", va="bottom")
ax.errorbar(X, Y, yerr=YE, fmt="o-", color=C_MUD, lw=1.8, ms=5.5, capsize=3)
for x, y, e in zip(X, Y, effs):
    ax.annotate(f"{e}%", (x, y), textcoords="offset points", xytext=(8, 7), fontsize=9.5)
ax.axhline(0, color="0.6", ls="--", lw=1.0)
ax.set_xlabel(r"$\mu$DIF contamination in the low bin [%]")
ax.set_ylabel(r"$\pi\to e$ tail bias [%]")
ax.set_xlim(left=0)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_bias_vs_contamination.png"); plt.close(fig)
STATS["mudif_tradeoff"] = tab
print(json.dumps(tab, indent=1), flush=True)

# =========================================================== 5) suppression vs angle
# pre-acceptance, as in the canonical study: the acceptance cut empties every bin beyond 120 deg
c50 = cutq(50)
deg = np.degrees(md.positron_theta.to_numpy())
kept = msc < c50
mth = np.isfinite(Emd) & (Emd <= EM) & np.isfinite(deg)
edges = np.arange(0, 181, 15); ctr = (edges[:-1] + edges[1:]) / 2
supp, serr = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = mth & (deg >= lo) & (deg < hi)
    n, nk = m.sum(), (m & kept).sum()
    supp.append(n / max(nk, 1)); serr.append((n / max(nk, 1)) / np.sqrt(max(nk, 1)))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.errorbar(ctr, supp, yerr=serr, fmt="o-", color=C_MUD, lw=1.8, ms=5, capsize=3)
ax.axvline(120, color="k", ls=":", lw=0.9)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"positron polar angle $\theta$ [deg]")
ax.set_ylabel(r"$\mu$DIF suppression ($1/$leakage)")
fig.tight_layout(); fig.savefig(f"{OUT}/tail_mudif_angular.png"); plt.close(fig)
STATS["mudif_angular"] = dict(theta=ctr.tolist(), suppression=[float(s) for s in supp],
                              pie_eff_working_point=50)

# =========================================================== 6) irreducible component
print("== survivors ==", flush=True)
key = lambda x: np.round(np.asarray(x).astype(np.float32).astype(np.float64), 4)
md_full = pd.read_parquet(EV10,
                          columns=["is_mudif", "muon_score", "muon_dif_score", "pion_dif_score",
                                   "topo_score", "pileup_score", "deposited_energy", "acceptance"])
md_full = md_full[md_full.is_mudif == 1].reset_index(drop=True)
accm = ((md_full.acceptance >= 1) & (md_full.deposited_energy >= 0) &
        (md_full.deposited_energy <= EM)).to_numpy()
survm = accm & (md_full.muon_dif_score < c50).to_numpy() & (md_full.pion_dif_score < 0.01).to_numpy() & \
        (md_full.muon_score < 0.02).to_numpy() & (md_full.pileup_score < 0.5).to_numpy() & \
        (md_full.topo_score > 0.9).to_numpy()
surv_keys = set(key(md_full[survm].deposited_energy.values))
acc_keys = set(key(md_full[accm].deposited_energy.values))
print(f"survivors={int(survm.sum())}  accepted muDIF={int(accm.sum())}", flush=True)
# live_E key join with collisions tolerated, the same matching mudif_surv_analysis.py uses;
# collisions mix non-survivors into the survivor set, so the survivor contrast shown is a
# LOWER bound on the true separation
MUON, POSITRON = 2, 4
t = pq.read_table(f"{P}/tail10/eval_mudif_osl.parquet",
                  columns=["atar_pdg", "atar_origin", "atar_E", "live_E", "muon_decay_ke"])
pdg = t.column("atar_pdg").combine_chunks(); org = t.column("atar_origin").combine_chunks()
aE = t.column("atar_E").combine_chunks()
fp = pdg.flatten().to_numpy(False).astype(np.int64)
fo = org.flatten().to_numpy(False).astype(np.int64)
fE = aE.flatten().to_numpy(False).astype(float)
lens = np.diff(pdg.offsets.to_numpy()); nE = len(lens); ev = np.repeat(np.arange(nE), lens)
mmask = ((fp & MUON) > 0) & (fo == 0)
n_muon = np.bincount(ev[mmask], minlength=nE)
muon_E = np.bincount(ev[mmask], weights=fE[mmask], minlength=nE)
pmask_h = ((fp & POSITRON) > 0) & (fo == 0)
hE = fE[pmask_h]; hh, hedge = np.histogram(hE[hE < 0.2], bins=80)
mip_mpv = float((hedge[np.argmax(hh)] + hedge[np.argmax(hh) + 1]) / 2)
mip_mean = float(hE[hE < 0.2].mean())
d6 = pd.DataFrame({"k": key(t.column("live_E").to_numpy(False)), "n_muon": n_muon,
                   "muon_E": muon_E, "dke": t.column("muon_decay_ke").to_numpy(False)})
A = d6[d6.k.isin(acc_keys)]; Sv = d6[d6.k.isin(surv_keys)]
print(f"key-matched: accepted={len(A)}, survivors={len(Sv)} (collisions dilute the contrast)", flush=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
b1 = np.linspace(0, 0.2, 41)
a1.hist(A.muon_E, bins=b1, weights=np.full(len(A), 1 / len(A)), histtype="step", lw=1.8,
        color="0.45", label=f"all accepted $\\mu$DIF (n={len(A):,})")
a1.hist(Sv.muon_E, bins=b1, weights=np.full(len(Sv), 1 / len(Sv)), histtype="stepfilled",
        color=C_MUD, alpha=0.45, ec=C_MUD, lw=1.8, label=f"survivors (n={len(Sv)})")
a1.axvline(mip_mpv, color=C_PIE, ls="--", lw=1.2)
a1.text(mip_mpv, a1.get_ylim()[1], " MIP MPV", color=C_PIE, fontsize=9, va="top")
a1.axvline(mip_mean, color=C_MIC, ls=":", lw=1.2)
a1.text(mip_mean, 0.85 * a1.get_ylim()[1], " MIP mean", color=C_MIC, fontsize=9, va="top")
a1.set_xlim(0, 0.2)
a1.set_xlabel("true muon energy deposited in the ATAR [MeV]")
a1.set_ylabel("fraction of events / bin"); a1.legend(fontsize=9)
b2 = np.linspace(0, 4.2, 29)
a2.hist(A.dke, bins=b2, weights=np.full(len(A), 1 / len(A)), histtype="step", lw=1.8, color="0.45")
a2.hist(Sv.dke, bins=b2, weights=np.full(len(Sv), 1 / len(Sv)), histtype="stepfilled",
        color=C_MUD, alpha=0.45, ec=C_MUD, lw=1.8)
a2.set_xlim(0, 4.2)
a2.set_xlabel("muon kinetic energy at decay [MeV]")
a2.set_ylabel("fraction of events / bin")
fig.tight_layout(); fig.savefig(f"{OUT}/tail_mudif_irreducible.png"); plt.close(fig)
STATS["mudif_survivors"] = dict(
    n_survivors=int(survm.sum()), n_matched_survivor_rows=int(len(Sv)), n_accepted=int(len(A)),
    frac_full_ke=float((Sv.dke > 3.9).mean()), median_muon_E_MeV=float(Sv.muon_E.median()),
    median_n_muon_hits=float(Sv.n_muon.median()), mip_mpv_MeV=mip_mpv, mip_mean_MeV=mip_mean,
    full_chain_suppression=float(accm.sum() / max(survm.sum(), 1)),
    frac_le1_muon_hit=float((Sv.n_muon <= 1).mean()),
    match_method="live_E key join with collisions (mudif_surv_analysis.py convention); survivor contrast is a lower bound")
print(json.dumps(STATS["mudif_survivors"], indent=1), flush=True)
del t, d6

# =========================================================== 7) ROC curves
print("== ROC ==", flush=True)
ev10 = pd.read_parquet(EV10,
                       columns=["is_pie", "is_mudif", "is_pidif", "muon_present",
                                "pileup_present", "pie_score", "topo_score", "muon_score",
                                "pileup_score", "muon_dif_score", "pion_dif_score", "acceptance"])
ev10 = ev10[ev10.acceptance >= 1].reset_index(drop=True)
is_mic = (ev10.is_pie == 0) & (ev10.is_mudif == 0) & (ev10.is_pidif == 0)
pie_m = ev10.is_pie == 1
curves = [
    (r"$\pi\to e$ tagger vs $\pi\to\mu\to e$", ev10.pie_score[pie_m], ev10.pie_score[is_mic], C_PIE, "-"),
    (r"$\pi\to e$ topology vs $\pi\to\mu\to e$", ev10.topo_score[pie_m], ev10.topo_score[is_mic], C_PIE, "--"),
    ("muon veto (own truth)", ev10.muon_score[ev10.muon_present == 1], ev10.muon_score[ev10.muon_present == 0], C_MIC, "-"),
    ("pileup veto (own truth)", ev10.pileup_score[ev10.pileup_present == 1], ev10.pileup_score[ev10.pileup_present == 0], C_MIC, "--"),
    (r"$\mu$DIF veto vs $\pi\to e$", ev10.muon_dif_score[ev10.is_mudif == 1], ev10.muon_dif_score[pie_m], C_MUD, "-"),
    (r"$\pi$DIF veto vs $\pi\to e$", ev10.pion_dif_score[ev10.is_pidif == 1], ev10.pion_dif_score[pie_m], C_PID, "-"),
]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
auc_tab = {}
for name, sig, bkg, col, ls in curves:
    sig, bkg = sig.to_numpy(), bkg.to_numpy()
    a = auc_rank(sig, bkg)
    se, be = roc_curve(sig, bkg)
    ax.plot(se, be, ls, color=col, lw=1.7, label=f"{name}  (AUC {a:.4f})")
    auc_tab[name] = float(a)
ax.set_yscale("log"); ax.set_ylim(1e-5, 1.2); ax.set_xlim(0, 1.02)
ax.set_xlabel("signal-class efficiency")
ax.set_ylabel("background-class efficiency")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/tail_roc.png"); plt.close(fig)
STATS["roc_auc"] = auc_tab
print(json.dumps(auc_tab, indent=1), flush=True)

allstats = {}
if os.path.exists(f"{OUT}/tail_stats.json"):
    allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats.update(STATS)                      # merge: other scripts own the rest of the file
with open(f"{OUT}/tail_stats.json", "w") as f:
    json.dump(allstats, f, indent=1)
print(f"wrote {OUT}/tail_stats.json\nALL DONE", flush=True)
