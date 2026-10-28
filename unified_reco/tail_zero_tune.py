"""Survivor autopsy + cut retuning for zero Michel and zero piDIF survivors, on the
fully reconstructed chains of tail_reco_cutflows.py.

Part 1: dump every event that survives the current chains (Michel and piDIF), with all six
tail scores, the margin to each cut, energies, and truth flags, so the failure mode of each
event is visible.

Part 2: grid-scan the thresholds (pileup, topology, stopped-muon, muon-DIF, pion-DIF, and the
so-far-unused pie tagger) over the combined chain, keep every combination that leaves zero
Michel AND zero piDIF survivors, and report the Pareto front in (pie efficiency, |tail bias|),
with the muDIF residual at each point for context.

Outputs: figures/AI/tail_zero_tune.json + printed tables.
"""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
ES, EM = 56.0, 75.0
KMUDIF, KRAD = 0x1000, 0x200

TCOLS = ["muon_score", "pileup_score", "topo_score", "pie_score", "muon_dif_score",
         "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_dead_E", "truth_event_type", "truth_gen_weight", "pred_accepted",
         "pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns",
         "truth_has_atar_pileup", "truth_has_accidental", "truth_positron_energy"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def paired(tail, main, label):
    tE, mE = f32(tail.deposited_energy.values), f32(main.truth_live_E.values)
    ti, pt, pm = 0, [], []
    for mi in range(len(main)):
        while ti < len(tail) and tE[ti] != mE[mi]:
            ti += 1
        if ti >= len(tail):
            break
        pt.append(ti); pm.append(mi); ti += 1
    if len(pt) < 0.999 * len(main):
        print(f"  [{label}] pairing failed, skip", flush=True)
        return None
    j = tail.iloc[pt].reset_index(drop=True)
    mm = main.iloc[pm].reset_index(drop=True)
    for c in mm.columns:
        j["m_" + c] = mm[c].values
    return j


def finish(j, w_gen=False):
    d = pd.DataFrame({
        "Ereco": j.m_pred_positron_energy.values + j.m_pred_dead_energy.values,
        "Edep": j.m_truth_live_E.values + j.m_truth_dead_E.values,
        "acc": (j.m_pred_accepted.values >= 1),
        "t": j.m_pred_positron_time_consensus_ns.values,
        "muon": j.muon_score.values, "pileup": j.pileup_score.values,
        "topo": j.topo_score.values, "pie_s": j.pie_score.values,
        "mudif": j.muon_dif_score.values, "pidif": j.pion_dif_score.values,
        "atar_pu": j.m_truth_has_atar_pileup.values, "accid": j.m_truth_has_accidental.values,
        "posE": j.m_truth_positron_energy.values,
    })
    d["w"] = j.m_truth_gen_weight.values if w_gen else 1.0
    return d


# ----------------------------------------------------------------- load joined samples
pie_rows, mic_rows = [], []
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
    pie_rows.append(finish(j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)]))
pie = pd.concat(pie_rows, ignore_index=True)

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
    mic_rows.append(finish(j[((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0)], w_gen=True))
mic = pd.concat(mic_rows, ignore_index=True)

mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                     columns=TCOLS + ["is_pidif", "is_mudif"])
mm_all = {}
for flag, mfile in (("is_pidif", "pidif_osl"), ("is_mudif", "mudif_osl")):
    tt = mt[mt[flag] == 1].reset_index(drop=True)
    m = pd.read_parquet(f"{P}/purity_eval/tail_osl/{mfile}_events.parquet", columns=MCOLS)
    mm_all[flag] = finish(paired(tt, m, mfile))
pid, mud = mm_all["is_pidif"], mm_all["is_mudif"]
print(f"loaded: pie {len(pie):,}  michel {len(mic):,}  pidif {len(pid):,}  mudif {len(mud):,}", flush=True)

onset = np.quantile(pie.t.to_numpy()[pie.acc.to_numpy() & np.isfinite(pie.t.to_numpy())], 0.005)
TLO, THI = onset + 2.0, onset + 32.0


def base(d):
    E = d.Ereco.to_numpy()
    ok = d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)
    return ok & (d.t.to_numpy() >= TLO) & (d.t.to_numpy() <= THI)


CUR = dict(pileup=0.5, topo=0.9, muon=0.02, mudif=0.46, pidif=0.01, pie_s=-1.0)


def chain(d, c, with_pidif=True):
    k = base(d) & (d.pileup.to_numpy() < c["pileup"]) & (d.topo.to_numpy() > c["topo"]) \
        & (d.muon.to_numpy() < c["muon"]) & (d.mudif.to_numpy() < c["mudif"]) \
        & (d.pie_s.to_numpy() > c["pie_s"])
    if with_pidif:
        k = k & (d.pidif.to_numpy() < c["pidif"])
    return k


# ----------------------------------------------------------------- part 1: autopsy
print(f"\nprompt window [{TLO:.1f}, {THI:.1f}] ns", flush=True)
for name, d, wp in (("MICHEL", mic, False), ("PIDIF", pid, True)):
    surv = chain(d, CUR, with_pidif=wp)
    print(f"\n== {name} survivors of the current chain: {int(surv.sum())} ==", flush=True)
    cols = ["Ereco", "Edep", "posE", "t", "muon", "pileup", "topo", "pie_s", "mudif", "pidif",
            "atar_pu", "accid"]
    print(d[surv][cols].to_string(float_format=lambda v: f"{v:8.3f}"), flush=True)

# ----------------------------------------------------------------- part 2: grid scan
GRID = dict(
    pileup=[0.5, 1.01],
    topo=[0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99],
    muon=[0.005, 0.01, 0.02],
    mudif=[0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.46],
    pie_s=[-1.0, 0.9],
)
pidif_cut = 0.01

def precut(d):
    b = base(d)
    return dict(b=b,
                pileup={v: d.pileup.to_numpy() < v for v in GRID["pileup"]},
                topo={v: d.topo.to_numpy() > v for v in GRID["topo"]},
                muon={v: d.muon.to_numpy() < v for v in GRID["muon"]},
                mudif={v: d.mudif.to_numpy() < v for v in GRID["mudif"]},
                pie_s={v: d.pie_s.to_numpy() > v for v in GRID["pie_s"]},
                pidif=d.pidif.to_numpy() < pidif_cut)

pm = precut(pie); mm_ = precut(mic); pdm = precut(pid); mdm = precut(mud)
Ep = pie.Ereco.to_numpy()
acc_pie = pie.acc.to_numpy() & np.isfinite(Ep) & (Ep >= 0) & (Ep <= EM)
n_acc = acc_pie.sum()
t_pie = pm["b"]
tf_ref = (Ep[t_pie] < ES).sum() / max((Ep[t_pie] >= ES).sum(), 1)

results = []
for pu in GRID["pileup"]:
    for to in GRID["topo"]:
        for mu in GRID["muon"]:
            for md in GRID["mudif"]:
                for ps in GRID["pie_s"]:
                    km = mm_["b"] & mm_["pileup"][pu] & mm_["topo"][to] & mm_["muon"][mu] \
                         & mm_["mudif"][md] & mm_["pie_s"][ps]
                    nm = int(km.sum())
                    if nm > 0:
                        continue
                    kp = pdm["b"] & pdm["pileup"][pu] & pdm["topo"][to] & pdm["muon"][mu] \
                         & pdm["mudif"][md] & pdm["pie_s"][ps] & pdm["pidif"]
                    npd = int(kp.sum())
                    if npd > 0:
                        continue
                    ke = t_pie & pm["pileup"][pu] & pm["topo"][to] & pm["muon"][mu] \
                         & pm["mudif"][md] & pm["pie_s"][ps] & pm["pidif"]
                    eff = 100 * ke.sum() / max(n_acc, 1)
                    slo = (ke & (Ep < ES)).sum(); shi = (ke & (Ep >= ES)).sum()
                    tf = slo / max(shi, 1)
                    bias = 100 * (tf / tf_ref - 1)
                    berr = 100 * (tf / tf_ref) * np.sqrt(1 / max(slo, 1) + 1 / max(shi, 1))
                    kd = mdm["b"] & mdm["pileup"][pu] & mdm["topo"][to] & mdm["muon"][mu] \
                         & mdm["mudif"][md] & mdm["pie_s"][ps] & mdm["pidif"]
                    results.append(dict(pileup=pu, topo=to, muon=mu, mudif=md, pie_s=ps,
                                        pie_eff=float(eff), bias=float(bias), bias_err=float(berr),
                                        mudif_surv=int(kd.sum())))
print(f"\nzero-zero combinations: {len(results)}", flush=True)
results.sort(key=lambda r: -r["pie_eff"])
print(f"{'pileup':>7} {'topo':>6} {'muon':>7} {'mudif':>6} {'pie_s':>6} | {'pie eff':>8} {'bias':>14} {'muDIF left':>10}")
for r in results[:25]:
    print(f"{r['pileup']:7.2f} {r['topo']:6.2f} {r['muon']:7.3f} {r['mudif']:6.2f} {r['pie_s']:6.2f} | "
          f"{r['pie_eff']:7.2f}% {r['bias']:+7.2f}+/-{r['bias_err']:4.2f}% {r['mudif_surv']:>10,}", flush=True)

print("\nsoftest-topology frontier (best pie eff per topology value):")
best_by_topo = {}
for r in results:
    k = r["topo"]
    if k not in best_by_topo or r["pie_eff"] > best_by_topo[k]["pie_eff"]:
        best_by_topo[k] = r
for k in sorted(best_by_topo):
    r = best_by_topo[k]
    print(f"  topo>{k:5.3f}: pileup<{r['pileup']:4.2f} muon<{r['muon']:5.3f} mudif<{r['mudif']:4.2f} pie_s>{r['pie_s']:5.2f} | "
          f"eff {r['pie_eff']:6.2f}%  bias {r['bias']:+.2f}+/-{r['bias_err']:.2f}%  muDIF left {r['mudif_surv']:,}", flush=True)

json.dump(dict(window=[float(TLO), float(THI)], tf_ref=float(100 * tf_ref),
               current=CUR, zero_zero=results),
          open(f"{OUT}/tail_zero_tune_fine.json", "w"), indent=1)
print("ALL DONE", flush=True)
