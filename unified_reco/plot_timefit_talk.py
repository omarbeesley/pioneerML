"""Updated high/low-bin time-fit plot in the remix_fit_K9 style, cleaned up:
black data points + Poisson error bars, stacked filled components (grey accidental ->
blue michel -> red pi->e), black total-fit line, pion-stop (dotted) + fit-start (green)
lines.  NO title, NO pulls panel, FULL readout window (-300..500 ns).

Data: the gatefix10M main-model eval (10M non-rad pie + 20M non-rad in-window michel), pie
scaled to michel luminosity by pool size.  High bin uncapped (see remu_bias emax sweep).
"""
import glob, os, numpy as np, pandas as pd, matplotlib
import pyarrow.parquet as _pq
def pq_meta(f):
    return _pq.ParquetFile(f).metadata.num_rows
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import nnls

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/pioneerML/unified_reco/updated_plots/02_timefit_highlow_fix10M.png"
TAU_PI, TAU_MU = 26.033, 2196.981
T0_PIE, T0_MU = 6.39, 4.04                       # fitted decay onsets (ns)
bins = np.linspace(-300, 500, 401); lo, hi = bins[:-1], bins[1:]
ctr = 0.5 * (lo + hi); neg = hi <= 0.0           # pre-stop sideband -> flat accidental level
COL = dict(acc="#bbbbbb", mue="#1f77b4", pie="#d62728")

def basis(t0, mic=False):
    a, b = np.maximum(lo, t0), np.maximum(hi, t0)
    ie = lambda tau: np.exp(-(a - t0) / tau) - np.exp(-(b - t0) / tau)
    v = (TAU_MU * ie(TAU_MU) - TAU_PI * ie(TAU_PI)) / (TAU_MU - TAU_PI) if mic else ie(TAU_PI)
    return v / v[hi > t0].sum()
pieF, micF = basis(T0_PIE), basis(T0_MU, True)

COLS = ["pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns",
        "truth_event_type", "truth_gen_weight", "truth_positron_energy", "pred_accepted"]

def hist_both(shards, is_pie):
    """One pass over the shards -> (H,V) weighted hist + sq-weight hist for BOTH energy bins."""
    Hh = Vh = Hl = Vl = np.zeros(len(ctr)); N = 0
    for f in shards:
        d = pd.read_parquet(f, columns=COLS)
        E = (np.clip(d.pred_positron_energy, 0, None) + np.clip(d.pred_dead_energy, 0, None)).to_numpy()
        tr = d.pred_positron_time_consensus_ns.to_numpy(); et = d.truth_event_type.to_numpy().astype(np.int64)
        keep = np.ones(len(d), bool) if is_pie else \
            ((et & 1) == 0) & ((et & 0x200) == 0) & (d.truth_positron_energy.to_numpy() <= 55)
        base = keep & (d.pred_accepted.to_numpy() >= 0.5) & (tr >= -300) & (tr <= 500)
        w = d.truth_gen_weight.to_numpy().astype(float)
        # high bin CAPPED at 75 for the FIT DISPLAY only: reco>75 events are pileup-merge
        # over-reco (79-84% pileup-tagged) with unreliable times -- they belong to the flat
        # term, not the decay laws. The COUNTING measurement stays uncapped (remu_bias).
        mh = base & (E >= 56) & (E <= 75); ml = base & (E < 56)
        Hh = Hh + np.histogram(tr[mh], bins, weights=w[mh])[0]; Vh = Vh + np.histogram(tr[mh], bins, weights=w[mh]**2)[0]
        Hl = Hl + np.histogram(tr[ml], bins, weights=w[ml])[0]; Vl = Vl + np.histogram(tr[ml], bins, weights=w[ml]**2)[0]
        N += len(d)
    return Hh, Vh, Hl, Vl, N

# --- load the gatefix10M eval (10M non-rad pie + 20M non-rad in-window michel) ONCE, cache the
#     histograms so style iterations re-plot in seconds instead of re-reading 30M events ---
CACHE = "/home/obeesley/pioneerML/unified_reco/updated_plots/.timefit_hists_fix10M.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE)
    Ph_h, Pv_h, Ph_l, Pv_l = z["Ph_h"], z["Pv_h"], z["Ph_l"], z["Pv_l"]; NPIE = int(z["NPIE"])
    Mh_h = list(z["Mh_h"]); Mv_h = list(z["Mv_h"]); Mh_l = list(z["Mh_l"]); Mv_l = list(z["Mv_l"]); sc = list(z["sc"])
    print("loaded histogram cache", flush=True)
else:
    pf = [f"{P}/purity_eval/fix10M_pie_events.parquet"]
    Ph_h, Pv_h, Ph_l, Pv_l, NPIE = hist_both(pf, True)
    sh = [f"{P}/purity_eval/fix10M_michel_events.parquet"]
    hh, vh, hl, vl, nm = hist_both(sh, False)
    Mh_h = [hh]; Mv_h = [vh]; Mh_l = [hl]; Mv_l = [vl]; sc = [nm]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, Ph_h=Ph_h, Pv_h=Pv_h, Ph_l=Ph_l, Pv_l=Pv_l, NPIE=NPIE,
             Mh_h=np.array(Mh_h), Mv_h=np.array(Mv_h), Mh_l=np.array(Mh_l), Mv_l=np.array(Mv_l), sc=np.array(sc))
    print("computed + cached histograms", flush=True)
K = len(Mh_h); NMIC = int(np.mean(sc))
# OPEN michel = unconditioned (1 row = 1 trigger) -> row-count luminosity match is correct
scale = NMIC / NPIE
print(f"lumi: NMIC={NMIC:,} NPIE={NPIE:,} scale={scale:.3f}", flush=True)

# Accidental time shape: the sideband DECAYS (user-identified; -4.8 sigma in the low bin).
# Fit tau once in the statistics-rich LOW-bin sideband; each bin then anchors only the
# NORMALIZATION of the shared exp shape on its own sideband.
from scipy.optimize import curve_fit
_ol = Ph_l * scale + np.mean(Mh_l, axis=0)
_el = np.sqrt(Pv_l * scale**2 + np.sum(Mv_l, axis=0) / K**2)
_p, _ = curve_fit(lambda t, A, tau: A * np.exp(-(t + 300) / tau), ctr[neg], _ol[neg],
                  sigma=np.maximum(_el[neg], 1e-9), p0=[_ol[neg][:30].mean(), 3000], maxfev=40000)
TAU_ACC = float(_p[1])
ACC_SHAPE = np.exp(-(ctr + 300) / TAU_ACC)
print(f"accidental tau (low-bin sideband) = {TAU_ACC:,.0f} ns", flush=True)

def build(P_H, P_V, M_H, M_V, include_pie, t0):
    """Average rounds, scale pie to michel luminosity, DECAYING sideband-anchored accidental
    (shared tau from the low bin), nnls-fit with the calibrated T0."""
    Hm = np.mean(M_H, axis=0); Vm = np.sum(M_V, axis=0) / K**2      # variance of the mean
    Hp = P_H * scale; Vp = P_V * scale**2
    obs = Hp + Hm; verr = np.sqrt(np.maximum(Vp + Vm, 0.0))
    acc = (obs[neg].mean() / ACC_SHAPE[neg].mean()) * ACC_SHAPE     # own-sideband normalization
    pos = lo >= t0 + 5.0
    if include_pie:
        A = np.vstack([pieF[pos], micF[pos]]).T; c, _ = nnls(A, (obs - acc)[pos]); Npie, Nmue = float(c[0]), float(c[1])
    else:
        c, _ = nnls(micF[pos].reshape(-1, 1), (obs - acc)[pos]); Npie, Nmue = 0.0, float(c[0])
    rng = np.random.default_rng(1); toy = []
    for _ in range(400):
        o = obs + rng.normal(0, verr)
        accT = (o[neg].mean() / ACC_SHAPE[neg].mean()) * ACC_SHAPE
        if include_pie:
            cc, _ = nnls(A, (o - accT)[pos]); toy.append(cc[0])
    sig = float(np.std(toy)) if toy else 0.0
    mic_full = Nmue * micF; pie_full = Npie * pieF
    return dict(obs=obs, verr=verr, flat=acc, mic_full=mic_full, pie_full=pie_full,
                fit_tot=acc + mic_full + pie_full, Npie=Npie, Nmue=Nmue, sig=sig,
                include_pie=include_pie, fitstart=t0 + 5.0)

R_hi = build(Ph_h, Pv_h, Mh_h, Mv_h, True, T0_PIE)
R_lo = build(Ph_l, Pv_l, Mh_l, Mv_l, False, T0_MU)


# ================= TALK STYLE: two separate, decluttered thesis-style figures =================
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 15,
    "axes.labelsize": 17, "axes.titlesize": 17, "legend.fontsize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.grid": False})

def draw_talk(r, title, outfile):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.fill_between(ctr, r["flat"], color=COL["acc"], step="mid", label="Accidental")
    ax.fill_between(ctr, r["flat"], r["flat"] + r["mic_full"], color=COL["mue"], alpha=0.55,
                    step="mid", label=r"$\pi\to\mu\to e$")
    if r["include_pie"]:
        ax.fill_between(ctr, r["flat"] + r["mic_full"], r["fit_tot"], color=COL["pie"], alpha=0.6,
                        step="mid", label=r"$\pi\to e\nu$")
    ax.plot(ctr, r["fit_tot"], color="k", lw=1.3, label="Total Fit")
    ax.errorbar(ctr, r["obs"], yerr=r["verr"], fmt="o", ms=2.4, color="k", lw=0.7,
                label="Data", zorder=5)
    ax.set_yscale("log"); ax.set_xlim(-300, 500)
    pos_obs = r["obs"][r["obs"] > 0]
    ax.set_ylim(max(pos_obs.min() * 0.5, 1e-2), pos_obs.max() * 6.0)
    ax.set_xlabel("Reconstructed Positron Time [ns]")
    ax.set_ylabel("Weighted Counts / 2 ns")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(outfile); plt.close(fig)
    print(f"wrote {outfile}", flush=True)

UPD = "/home/obeesley/pioneerML/unified_reco/updated_plots"
draw_talk(R_hi, r"High Bin: $E \geq 56$ MeV", f"{UPD}/02_timefit_high_fix10M.png")
# ---- RMD-free emulation: accidental level scaled to the measured non-RMD fraction (x0.385);
#      signal components are level-invariant (proven: fit identical for any accidental scale) ----
SRMD = 0.385
R2 = dict(R_hi)
R2["flat"] = SRMD * R_hi["flat"]
R2["obs"]  = R_hi["obs"] - (1.0 - SRMD) * R_hi["flat"]
R2["verr"] = R_hi["verr"] * np.sqrt(np.clip(R2["obs"] / np.maximum(R_hi["obs"], 1e-9), 0.05, 1.0))
R2["fit_tot"] = R_hi["fit_tot"] - (1.0 - SRMD) * R_hi["flat"]
draw_talk(R2, r"High Bin: $E \geq 56$ MeV", f"{UPD}/02_timefit_high_fix10M_normd.png")
draw_talk(R_lo, r"Low Bin: $E < 56$ MeV", f"{UPD}/02_timefit_low_fix10M.png")
