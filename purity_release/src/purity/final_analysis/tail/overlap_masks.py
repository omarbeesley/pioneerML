"""Build the historical Michel source-overlap masks and report the three-way cutflow.

Ported from michel_leak_audit.py; see docs/tail_analysis.md for the historical population.
"""
from ._runtime import command


def run(*, pipeline_dir=None, cache_dir=None, output_dir, record,
        skip_cutflows=False, michel_nmax=107):
    import glob
    import os

    import numpy as np
    import pandas as pd

    P = str(pipeline_dir)
    OUTDIR = str(output_dir)
    os.makedirs(OUTDIR, exist_ok=True)

    KEYC = ["live_E", "truth_pion_stop_x", "truth_pion_stop_y", "truth_pion_stop_z"]


    def keyview(df, cols):
        a = np.stack([np.asarray(df[c], np.float32) for c in cols], axis=1)
        a = np.ascontiguousarray(a)
        return a.view([("", np.float32)] * 4).ravel()


    # ---- 1. the leaked-decay key set -----------------------------------------------------------
    src = sorted(f for f in record.glob(f"{P}/parquet/michel/michel_1595422_t*_c*.parquet")
                 if int(f.split("_t")[1][:3]) <= 44)
    print(f"leaked source shards (job 1595422, t000-t044): {len(src)}", flush=True)
    keys = []
    for f in src:
        d = record.read_parquet(f, columns=KEYC)
        keys.append(keyview(d, KEYC))
    leak_keys = np.unique(np.concatenate(keys))
    print(f"leaked decay keys: {len(leak_keys):,}", flush=True)


    def paired_idx(tailE, mainE):
        tE = np.asarray(tailE, np.float32)
        mE = np.asarray(mainE, np.float32)
        ti, out = 0, []
        for mi in range(len(mE)):
            while ti < len(tE) and tE[ti] != mE[mi]:
                ti += 1
            if ti >= len(tE):
                break
            out.append((ti, mi))
            ti += 1
        return out


    # ---- 2 + 3. per-piece mask and three-way chain ---------------------------------------------
    TLO, THI = 8.3, 38.3
    INKEY = list(KEYC)
    KPIENU, KMUDIF, KMURAD = 0x1, 0x1000, 0x200

    stages = {s: np.zeros(3, np.int64) for s in ("acc", "time", "topo", "muon")}
    resid = {"all": 0, "clean": 0, "leaked": 0}
    resid_t_clean = []
    tot = np.zeros(3, np.int64)          # [all, clean, leaked] rows
    topo_scores = {"clean": [], "leaked": []}   # in-window, genuine, accepted

    for pf in sorted(record.glob(f"{P}/michel_eval_inwin_drop/*/predictions.parquet")):
        piece = pf.split("/")[-2]
        inp = f"{P}/michel_inwin/{piece}.parquet"
        mf = f"{P}/purity_eval/filmg2_ep19_michel_inwin/{piece}/mic_events.parquet"
        if not (os.path.exists(inp) and os.path.exists(mf)):
            raise FileNotFoundError(f"Missing raw or main Michel input: {inp}, {mf}")
        raw = record.read_parquet(inp, columns=INKEY + ["event_type"])
        kv = keyview(raw, KEYC)
        leak = np.isin(kv, leak_keys)
        pd.DataFrame({"live_E": np.asarray(raw.live_E, np.float32), "leaked": leak}).to_parquet(
            f"{OUTDIR}/{piece}.parquet", index=False)

        t = record.read_parquet(pf, columns=["topo_score", "muon_score", "muon_dif_score",
                                         "pion_dif_score", "deposited_energy"])
        m = record.read_parquet(mf, columns=["truth_live_E", "truth_event_type", "pred_accepted",
                                         "pred_positron_time_consensus_ns"])
        if len(t) != len(m):
            raise ValueError(f"Tail/main row-count mismatch: {piece}")
        pairs = paired_idx(t.deposited_energy.values, m.truth_live_E.values)
        if len(pairs) < 0.999 * len(m):
            raise ValueError(f"Tail/main pairing failed: {piece}")
        ti = np.array([a for a, _ in pairs])
        mi = np.array([b for _, b in pairs])
        # leak flag for the paired rows, via the same key walk against the raw file
        rp = paired_idx(np.asarray(raw.live_E, np.float32), m.truth_live_E.values[mi])
        if len(rp) < 0.999 * len(mi):
            raise ValueError(f"Raw/main pairing failed: {piece}")
        lk = leak[np.array([a for a, _ in rp])]

        et = m.truth_event_type.values[mi].astype(np.int64)
        genuine = (et & (KPIENU | KMUDIF | KMURAD)) == 0
        acc = (m.pred_accepted.values[mi] >= 1) & genuine
        tw = ((m.pred_positron_time_consensus_ns.values[mi] >= TLO)
              & (m.pred_positron_time_consensus_ns.values[mi] <= THI))
        topo = t.topo_score.values[ti] > 0.99
        muon = (t.muon_score.values[ti] < 0.02) & (t.muon_dif_score.values[ti] < 0.46)
        chain = {"acc": acc, "time": acc & tw, "topo": acc & tw & topo,
                 "muon": acc & tw & topo & muon}
        for s, msk in chain.items():
            stages[s] += [msk.sum(), (msk & ~lk).sum(), (msk & lk).sum()]
        tot += [len(mi), int((~lk).sum()), int(lk.sum())]

        # in-window topo-score comparison (memorization probe)
        for nm, sel in (("clean", acc & tw & ~lk), ("leaked", acc & tw & lk)):
            topo_scores[nm].append(t.topo_score.values[ti][sel])

        # non-time residue (every cut but the window)
        nt = acc & topo & muon & (t.pion_dif_score.values[ti] < 0.01)
        resid["all"] += int(nt.sum())
        resid["clean"] += int((nt & ~lk).sum())
        resid["leaked"] += int((nt & lk).sum())
        tsel = m.pred_positron_time_consensus_ns.values[mi][nt & ~lk]
        resid_t_clean += list(tsel)

    print(f"\nrows: all {tot[0]:,}  clean {tot[1]:,}  leaked {tot[2]:,} "
          f"({100 * tot[2] / max(tot[0], 1):.2f}%)", flush=True)
    print(f"{'stage':8s} {'all':>12s} {'clean':>12s} {'leaked':>10s}  "
          f"{'surv/acc clean':>15s} {'surv/acc leaked':>15s}")
    for s in ("acc", "time", "topo", "muon"):
        a, c, l = stages[s]
        ca, cl = stages["acc"][1], stages["acc"][2]
        print(f"{s:8s} {a:>12,} {c:>12,} {l:>10,}  "
              f"{c / max(ca, 1):>15.3e} {l / max(cl, 1):>15.3e}", flush=True)

    for nm in ("clean", "leaked"):
        v = np.concatenate(topo_scores[nm]) if topo_scores[nm] else np.array([])
        if len(v):
            print(f"in-window topo_score {nm:6s}: n={len(v):,}  mean {v.mean():.4f}  "
                  f">0.99: {100 * (v > 0.99).mean():.4f}%", flush=True)

    rt = np.sort(np.array(resid_t_clean))
    print(f"\nnon-time residue: all {resid['all']}  clean {resid['clean']}  "
          f"leaked {resid['leaked']}", flush=True)
    if len(rt):
        print(f"clean residue positron times: [{rt.min():.0f}, {rt.max():.0f}] ns; "
              f"earliest is the displayed fig9a event if 157 ns survives", flush=True)
    print("LEAK_AUDIT_DONE", flush=True)


def main(argv=None):
    return command('overlap_masks', 'pipeline', run, __doc__, argv=argv,
                   diagnostics=False)


if __name__ == "__main__":
    main()
