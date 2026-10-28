"""Tail analysis on the MAIN model's reconstructed quantities.

Joins, per class, the tail-model scores (tail_reveal_eval10 eval-mode predictions) with the
main model's event summaries (nodrop_ep12 benchmark.py evals of the same osl files), then
re-runs the veto chains using the RECONSTRUCTED acceptance decision and RECONSTRUCTED
positron energy in place of the truth acceptance flag and truth deposited energy.

Join: both orderings derive from PURITYDataset order over the same osl file. The tail rows
carry 16 extra events (its hit counting differed at the 250-hit boundary), so a two-pointer
walk pairs rows on float32 equality of tail.deposited_energy vs main.truth_live_E and skips
unmatched tail rows; the pair count must equal the main row count exactly.

Outputs: thesis_ref/figures/AI/tail_reco_cutflows.png, tail_reco_bias.png, and a
'reco_analysis' block appended to figures/AI/tail_stats.json.
"""
import json
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
EV10 = f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet"
MOUT = f"{P}/purity_eval/tail_osl"
OUT = "/home/obeesley/thesis_ref/figures/AI"

import tail_figstyle
tail_figstyle.apply()
C_PIE, C_MIC, C_PID, C_MUD = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
ES, EM = 56.0, 75.0
STATS = {}

TAIL_COLS = ["pie_score", "topo_score", "muon_score", "pileup_score", "muon_dif_score",
             "pion_dif_score", "deposited_energy", "positron_t", "acceptance",
             "is_pie", "is_mudif", "is_pidif"]
CLASSES = [("pie", "is_pie", "pie_osl"), ("michel", None, "michel_osl"),
           ("mudif", "is_mudif", "mudif_osl"), ("pidif", "is_pidif", "pidif_osl")]

tail = pd.read_parquet(EV10, columns=TAIL_COLS)
is_mic = (tail.is_pie == 0) & (tail.is_mudif == 0) & (tail.is_pidif == 0)

ev0 = pq.ParquetFile(f"{MOUT}/pie_osl_events.parquet")
print("main events columns:", " ".join(ev0.schema_arrow.names), flush=True)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def join_class(tail_cls, main_file):
    m = pd.read_parquet(f"{MOUT}/{main_file}_events.parquet")
    tE = f32(tail_cls.deposited_energy.values)
    mE = f32(m.truth_live_E.values)
    ti, pairs_t, pairs_m = 0, [], []
    for mi in range(len(m)):
        while ti < len(tail_cls) and tE[ti] != mE[mi]:
            ti += 1
        if ti >= len(tail_cls):
            break
        pairs_t.append(ti); pairs_m.append(mi); ti += 1
    print(f"{main_file}: main={len(m):,} tail={len(tail_cls):,} paired={len(pairs_t):,} "
          f"(tail skipped {len(tail_cls) - len(pairs_t)})", flush=True)
    assert len(pairs_t) >= 0.999 * len(m), f"{main_file}: alignment failed"
    j = tail_cls.iloc[pairs_t].reset_index(drop=True)
    mm = m.iloc[pairs_m].reset_index(drop=True)
    for c in mm.columns:
        j["m_" + c] = mm[c].values
    return j


joined = {}
for name, flag, mfile in CLASSES:
    tc = tail[is_mic].reset_index(drop=True) if flag is None else tail[tail[flag] == 1].reset_index(drop=True)
    joined[name] = join_class(tc, mfile)

# ---- reconstructed quantities ----------------------------------------------------------
# reco energy: the main model's assembled positron energy including its dead-material
# estimate; reco acceptance: the main model's own accept decision; reco time: the main
# model's positron time (sentinel -> fail the window).
def reco_cols(j):
    Ereco = j["m_pred_positron_energy"].to_numpy() + j["m_pred_dead_energy"].to_numpy()
    acc = j["m_pred_accepted"].to_numpy()
    t = j["m_pred_positron_time_consensus_ns"].to_numpy()
    return Ereco, acc, t


for name in joined:
    Ereco, acc, t = reco_cols(joined[name])
    j = joined[name]
    j["Ereco"] = Ereco
    j["acc_reco"] = (np.asarray(acc) >= 1).astype(int)
    j["t_reco"] = t

pie_j, mic_j, mud_j, pid_j = joined["pie"], joined["michel"], joined["mudif"], joined["pidif"]
STATS["join"] = {k: int(len(v)) for k, v in joined.items()}

# ---- chains on reconstructed quantities ------------------------------------------------
def chain_masks(j, use_reco=True):
    E = j.Ereco.values if use_reco else j.deposited_energy.values
    acc = (j.acc_reco.values == 1) if use_reco else (j.acceptance.values >= 1)
    t = j.t_reco.values if use_reco else j.positron_t.values
    ok = np.isfinite(E) & (E >= 0) & (E <= EM)
    twin = (t >= 6.0) & (t <= 36.0)     # absolute window = pion-stop-referenced [2,32] ns
    mu = (j.muon_score.values < 0.02) & (j.muon_dif_score.values < 0.46)
    cl = (j.pileup_score.values < 0.5) & (j.topo_score.values > 0.9)
    pdv = j.pion_dif_score.values < 0.01
    return E, dict(acc=acc & ok, time=acc & ok & twin, pileup=acc & ok & twin & (j.pileup_score.values < 0.5),
                   topo=acc & ok & twin & (j.pileup_score.values < 0.5) & (j.topo_score.values > 0.9),
                   muon=acc & ok & twin & cl & mu, full=acc & ok & twin & cl & mu & pdv)


R = {"pie": 1e-4, "michel": 1.0, "mudif": 1e-4 / 20.0, "pidif": 3e-4}
bins = np.linspace(0, EM, 51)
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True)
stages_show = [("acc", "(0) acceptance"), ("time", "(1) + time"), ("topo", "(2) + pileup & topology"),
               ("full", "(3) + muon & $\\pi$DIF vetoes")]
tab = {}
for (variant, use_reco), col in [(("truth", False), 0), (("reco", True), 1)]:
    Ep, mp = chain_masks(pie_j, use_reco)
    tab[variant] = {}
    for row, (bname, bj, bcol) in enumerate([("michel", mic_j, C_MIC), ("mudif", mud_j, C_MUD)]):
        Eb, mb = chain_masks(bj, use_reco)
        ax = axs[row][col]
        wp = R["pie"] / max(mp["acc"].sum(), 1); wb = R[bname] / max(mb["acc"].sum(), 1)
        st = {}
        for key, label in stages_show:
            plo = wp * (mp[key] & (Ep < ES)).sum(); phi = wp * (mp[key] & (Ep >= ES)).sum()
            blo = wb * (mb[key] & (Eb < ES)).sum(); bhi = wb * (mb[key] & (Eb >= ES)).sum()
            st[key] = dict(pie_eff=100 * mp[key].sum() / max(mp["acc"].sum(), 1),
                           contam_lo=100 * blo / (blo + plo) if (blo + plo) > 0 else 0.0,
                           contam_hi=100 * bhi / (bhi + phi) if (bhi + phi) > 0 else 0.0,
                           bkg_events=int(mb[key].sum()))
        tab[variant][bname] = st
        ax.hist(Ep[mp["full"]], bins=bins, weights=np.full(mp["full"].sum(), wp),
                histtype="step", lw=1.6, color=C_PIE, label=r"$\pi\to e$ (full chain)")
        ax.hist(Eb[mb["acc"]], bins=bins, weights=np.full(mb["acc"].sum(), wb),
                histtype="stepfilled", color="0.9", ec="0.6", label=f"{bname} (acceptance)")
        ax.hist(Eb[mb["full"]], bins=bins, weights=np.full(mb["full"].sum(), wb),
                histtype="step", lw=1.8, color=bcol, label=f"{bname} (full chain)")
        ax.axvline(ES, color="k", ls=":", lw=0.9); ax.set_yscale("log")
        ttl = "truth energy and acceptance" if variant == "truth" else "reconstructed energy and acceptance"
        ax.set_title(f"{bname} vs \\pie: {ttl}" if False else f"{ttl}", fontsize=10.5)
        c = st["full"]
        ax.text(0.03, 0.03, f"{bname} contam ($E<$56): {c['contam_lo']:.1f}%\n"
                            f"$\\pi\\to e$ eff (of acc): {c['pie_eff']:.1f}%",
                transform=ax.transAxes, va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92))
        if row == 1:
            ax.set_xlabel("energy [MeV]")
        if col == 0:
            ax.set_ylabel("branching-weighted rate [a.u.]")
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/tail_reco_cutflows.png"); plt.close(fig)
STATS["chains"] = tab
print(json.dumps(tab, indent=1), flush=True)

# ---- muDIF bias vs contamination on RECO energy ----------------------------------------
Ep, mp = chain_masks(pie_j, True); Em, mm_ = chain_masks(mud_j, True)
psc = pie_j.muon_dif_score.values; msc = mud_j.muon_dif_score.values
pa = mp["acc"]; ma = mm_["acc"]
rlo = (pa & (Ep < ES)).sum(); rhi = (pa & (Ep >= ES)).sum()
wpie = R["pie"] / pa.sum(); wdif = R["mudif"] / ma.sum()
effs = [99, 95, 90, 80, 70, 50, 30]
X, Y, YE, wp_tab = [], [], [], []
for e in effs:
    c = np.quantile(psc[pa], e / 100.0)
    pk = pa & (psc < c); mk = ma & (msc < c)
    plo = wpie * (pk & (Ep < ES)).sum(); mlo = wdif * (mk & (Em < ES)).sum()
    contam = 100 * mlo / (mlo + plo) if (mlo + plo) > 0 else 0.0
    slo = (pk & (Ep < ES)).sum(); shi = (pk & (Ep >= ES)).sum()
    el, eh = slo / max(rlo, 1), shi / max(rhi, 1); r = el / max(eh, 1e-12)
    err = 100 * r * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
    X.append(contam); Y.append(100 * (r - 1)); YE.append(err)
    wp_tab.append(dict(pie_eff=e, contam_pct=float(contam), bias_pct=float(100 * (r - 1)),
                       bias_unc_pct=float(err), suppression=float(ma.sum() / max(mk.sum(), 1))))
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.add_patch(patches.Rectangle((0, -1), 1, 2, fc="#2ca02c", alpha=0.13, ec="none", zorder=0))
ax.errorbar(X, Y, yerr=YE, fmt="o-", color=C_MUD, lw=1.8, ms=5.5, capsize=3)
for x, y, e in zip(X, Y, effs):
    ax.annotate(f"{e}%", (x, y), textcoords="offset points", xytext=(8, 7), fontsize=9.5)
ax.axhline(0, color="0.6", ls="--", lw=1.0)
ax.set_xlabel(r"$\mu$DIF contamination in the low bin [%] (reconstructed energy)")
ax.set_ylabel(r"$\pi\to e$ tail bias [%] (reconstructed energy)")
ax.set_xlim(left=0)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_reco_bias.png"); plt.close(fig)
STATS["mudif_tradeoff_reco"] = wp_tab
print(json.dumps(wp_tab, indent=1), flush=True)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["reco_analysis"] = STATS
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print("ALL DONE", flush=True)
