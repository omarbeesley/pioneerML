"""Build the final overlap-masked reconstructed-energy spectrum and irreducible cache.

Ported from tail_reveal_cache.py; see docs/tail_analysis.md for the historical population.
"""
from ._runtime import command


def run(*, pipeline_dir=None, cache_dir=None, output_dir, record,
        skip_cutflows=False, michel_nmax=107):
    import glob, json, os
    import numpy as np, pandas as pd, pyarrow.parquet as pq
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = str(pipeline_dir)
    OUT = str(output_dir)


    def _save(fig, name):
        """Write both formats; the PDF is what the thesis includes."""
        for _ext in ("pdf", "png"):
            fig.savefig(f"{OUT}/{name}.{_ext}")
        print(f"wrote {OUT}/{name}.pdf", flush=True)
    from . import style as tail_figstyle
    tail_figstyle.apply(scale=0.86)
    C_PIE, C_MIC, C_PID = "#0072B2", "#D55E00", "#009E73"
    ES, EM = 56.0, 90.0        # historical implemented reconstructed-energy cap: 90 MeV
    from . import rates as TR
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



    def leak_flags(piece, j):
        """Leak mask for the paired rows, or None if unavailable/misaligned.

        michel_inwin carries ~25% trigger decays the tail model trained on (the
        combine-prefix leak); masks from michel_leak_audit.py, keyed by raw row order,
        are aligned here by the same float32 live-energy walk used for the pairing."""
        mf = f"{P}/michel_inwin_leakmask/{piece}.parquet"
        if not os.path.exists(mf):
            raise FileNotFoundError(f"Required Michel overlap mask is missing: {mf}")
        mk = record.read_parquet(mf)
        lE = np.asarray(mk.live_E, np.float32)
        jE = np.asarray(j.deposited_energy, np.float32)
        li, idx = 0, []
        for v in jE:
            while li < len(lE) and lE[li] != v:
                li += 1
            if li >= len(lE):
                return None
            idx.append(li)
            li += 1
        if len(idx) < 0.999 * len(jE):
            return None
        return mk.leaked.values[np.array(idx)]

    # ------------------------------------------------------------------ load the three samples
    pie_rows, mic_rows = [], []
    MIC_GW_ALL = [0.0]
    for ev in sorted(record.glob(f"{P}/purity_eval/filmg2_ep19tg_pie/shard_*/pie_eval_events.parquet")):
        g = ev.split("shard_")[-1].split("/")[0]
        tp = f"{P}/norad_eval2_drop/ev_{g}/predictions.parquet"
        if not os.path.exists(tp):
            raise FileNotFoundError(f"Missing paired tail predictions: {tp}")
        m = record.read_parquet(ev, columns=MCOLS + ["truth_is_pie"])
        t = record.read_parquet(tp, columns=TCOLS + ["is_pie"])
        if len(m) != len(t):
            raise ValueError(f"Tail/main row-count mismatch: {ev}")
        j = paired(t, m, f"pie {g}")
        if j is None:
            raise ValueError("Tail/main positional pairing failed")
        j = j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)]
        pie_rows.append(finish(j))
    pie = pd.concat(pie_rows, ignore_index=True)
    print(f"pie: {len(pie_rows)} shards, {len(pie):,} rows", flush=True)

    for ev in sorted(record.glob(f"{P}/purity_eval/filmg2_ep19_michel_inwin/*/mic_events.parquet")):
        piece = ev.split("/")[-2]
        tp = f"{P}/michel_eval_inwin_drop/{piece}/predictions.parquet"
        if not os.path.exists(tp):
            raise FileNotFoundError(f"Missing paired tail predictions: {tp}")
        m = record.read_parquet(ev, columns=MCOLS)
        t = record.read_parquet(tp, columns=TCOLS)
        if len(m) != len(t):
            raise ValueError(f"Tail/main row-count mismatch: {ev}")
        j = paired(t, m, f"mic {piece}")
        if j is None:
            raise ValueError("Tail/main positional pairing failed")
        lk = leak_flags(piece, j)
        if lk is None:
            raise ValueError(f"Michel overlap mask is not aligned for piece {piece}")
        j = j[~lk].reset_index(drop=True)                 # drop trained-on trigger decays; the
        et = j.m_truth_event_type.values.astype(np.int64) # clean subsample represents the full
        gw_all = j.m_truth_gen_weight.values.astype(float)  # window rate, so gw renormalizes
        gw_all = gw_all[np.isfinite(gw_all)]
        MIC_GW_ALL[0] += gw_all.sum()                     # denominator BEFORE the RMD removal,
        j = j[((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0)]   # so removed
        mic_rows.append(finish(j, w_gen=True))            # rate is lost, not redistributed
    mic = pd.concat(mic_rows, ignore_index=True)
    print(f"michel: {len(mic_rows)} pieces, {len(mic):,} rows (kPienu/kMudif/RMD removed)", flush=True)

    mt = record.read_parquet(f"{P}/tail_reveal_eval10/predictions_drop.parquet",
                         columns=TCOLS + ["is_pidif"])
    mt = mt[mt.is_pidif == 1].reset_index(drop=True)
    mm = record.read_parquet(f"{P}/purity_eval/tail_osl/pidif_osl_events.parquet", columns=MCOLS)
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



    # ================= cache the per-stage spectra for the reveal-design figures =================
    import pyarrow.parquet as pq

    CACHE = {"bins": bins, "TLO": TLO, "THI": THI, "TS": TS}


    def stage_cache(tag, dbkg, order, wnorm, at_acceptance=False):
        Eb, mb = masks(dbkg)
        wb = TR.weights(len(dbkg), wnorm, gw=dbkg.w.to_numpy(),
                        mask=mb["acc"] if at_acceptance else None)
        for key in order:
            h, _ = np.histogram(Eb[mb[key]], bins=bins, weights=wb[mb[key]])
            CACHE[f"{tag}_{key}_h"] = h
            CACHE[f"{tag}_{key}_n"] = int(mb[key].sum())
        print(f"{tag}: " + "  ".join(f"{k}={int(mb[k].sum()):,}" for k in order), flush=True)


    for key in ("acc", "time", "topo", "muon", "full_pidif"):
        h, _ = np.histogram(Ep[mp[key]], bins=bins, weights=wpie_v[mp[key]])
        CACHE[f"pie_{key}_h"] = h
        CACHE[f"pie_{key}_n"] = int(mp[key].sum())

    stage_cache("mic", mic, ["acc", "time", "topo", "muon", "full_pidif"],
                F_MIC_WIN * (mic.w.to_numpy().sum() / MIC_GW_ALL[0]))
    stage_cache("pid", pid, ["acc", "time", "pdveto", "full_pidif"], TR.R_PIDIF_ABS)

    # ================= muDIF: joined chain stages + full-chain survivors ========================
    mdt = record.read_parquet(f"{P}/tail_reveal_eval10/predictions_drop.parquet",
                          columns=TCOLS + ["is_mudif", "muon_decay_ke"])
    mdt = mdt[mdt.is_mudif == 1].reset_index(drop=True)
    mdm = record.read_parquet(f"{P}/purity_eval/tail_osl/mudif_osl_events.parquet", columns=MCOLS)
    jm = paired(mdt, mdm, "mudif")
    mud = finish(jm)
    mud["dep"] = jm.deposited_energy.values
    mud["dke"] = jm.muon_decay_ke.values
    print(f"mudif joined: {len(mud):,} rows", flush=True)

    Emu, mmu = masks(mud)
    wnorm_mud = wpie_v[mp["acc"]].sum() * TR.R_MUDIF_AT_ACC   # pie/18 AT ACCEPTANCE
    wmud = TR.weights(len(mud), wnorm_mud, mask=mmu["acc"])
    for key in ("acc", "time", "topo", "muon", "full_pidif"):
        h, _ = np.histogram(Emu[mmu[key]], bins=bins, weights=wmud[mmu[key]])
        CACHE[f"mud_{key}_h"] = h
        CACHE[f"mud_{key}_n"] = int(mmu[key].sum())
    print("mudif stages: " + "  ".join(f"{k}={int(mmu[k].sum()):,}"
                                       for k in ("acc", "time", "topo", "muon", "full_pidif")),
          flush=True)

    # full-chain survivors, tight veto: the irreducible population
    dtw = (mud.t.to_numpy() >= TLO) & (mud.t.to_numpy() <= THI)
    tight = (mmu["acc"] & dtw & (mud.topo.to_numpy() > TOPO_CUT)
             & (mud.muon.to_numpy() < 0.02) & (mud.mudif.to_numpy() < 0.05))
    print(f"full-chain tight-veto muDIF survivors: {int(tight.sum()):,}", flush=True)


    def key(a):
        return np.round(np.asarray(a, np.float32).astype(np.float64), 4)


    surv_keys = set(key(mud.dep.values[tight]))
    acc_keys = set(key(mud.dep.values[mmu["acc"]]))

    t = record.read_table(f"{P}/tail10/eval_mudif_osl.parquet",
                      columns=["atar_pdg", "atar_origin", "atar_E", "live_E", "muon_decay_ke"])
    pdg = t.column("atar_pdg").combine_chunks()
    org = t.column("atar_origin").combine_chunks()
    aE = t.column("atar_E").combine_chunks()
    fp = pdg.flatten().to_numpy(False).astype(np.int64)
    fo = org.flatten().to_numpy(False).astype(np.int64)
    fE = aE.flatten().to_numpy(False).astype(float)
    lens = np.diff(pdg.offsets.to_numpy())
    nE = len(lens)
    ev = np.repeat(np.arange(nE), lens)
    MUON, POSITRON = 2, 4
    mmask = ((fp & MUON) > 0) & (fo == 0)
    muon_E = np.bincount(ev[mmask], weights=fE[mmask], minlength=nE)
    pmask = ((fp & POSITRON) > 0) & (fo == 0)
    hE = fE[pmask]
    hh, hedge = np.histogram(hE[hE < 0.2], bins=80)
    d6 = pd.DataFrame({"k": key(t.column("live_E").to_numpy(False)), "muon_E": muon_E,
                       "dke": t.column("muon_decay_ke").to_numpy(False)})
    A = d6[d6.k.isin(acc_keys)]
    Sv = d6[d6.k.isin(surv_keys)]
    CACHE.update(acc_muE=A.muon_E.values, acc_dke=A.dke.values,
                 surv_muE=Sv.muon_E.values, surv_dke=Sv.dke.values,
                 mip_mpv=float((hedge[np.argmax(hh)] + hedge[np.argmax(hh) + 1]) / 2),
                 mip_mean=float(hE[hE < 0.2].mean()))
    print(f"irreducible cache: accepted {len(A):,}, survivors {len(Sv):,}", flush=True)

    np.savez_compressed(f"{OUT}/tail_reveal_cache.npz", **CACHE)
    print(f"wrote {OUT}/tail_reveal_cache.npz", flush=True)
    print("CACHE_DONE", flush=True)


def main(argv=None):
    return command('cache', 'pipeline', run, __doc__, argv=argv,
                   diagnostics=False)


if __name__ == "__main__":
    main()
