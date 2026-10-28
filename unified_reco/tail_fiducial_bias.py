"""Is the pion-stop fiducial radius a real handle on the invisible-muon residue, or a
two-event coincidence?

The feature scan found both remaining muDAR survivors stopping at r = 9.27 and 9.70 mm, at the
transverse edge of the ATAR. That is the geometric precondition for the muon to leave sideways
through dead material, so it has a mechanism behind it rather than being a bare correlation.

Two things decide whether it is usable.  First, whether the survivors the tighter working point
ALREADY removes also sit at large radius: if they do, the mechanism is confirmed on four events
instead of two; if they sit at small radius, the two hits are a coincidence.  Second, whether a
fiducial cut sculpts the tail, measured the same way as the gap study -- efficiency in the tail
over efficiency in the peak, on accepted pi -> e.
"""
import glob, json, os
import numpy as np, pandas as pd

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
ES, EM = 56.0, 90.0
KMUDIF, KRAD = 0x1000, 0x200

MCOLS = ["truth_live_E", "truth_event_type", "truth_is_pie", "pred_accepted",
         "pred_pion_stop_x", "pred_pion_stop_y", "pred_pion_stop_z",
         "pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns"]
TCOLS = ["topo_score", "muon_score", "pileup_score", "muon_dif_score", "pion_dif_score",
         "deposited_energy"]


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


def base(d):
    E = d.m_pred_positron_energy.to_numpy() + d.m_pred_dead_energy.to_numpy()
    acc = (d.m_pred_accepted.to_numpy() >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM)
    return E, acc


def topo_chain(d, loose):
    """loose = the ORIGINAL working point that left 4 distinct survivors;
    tight = the retuned one (muDIF < 0.40, pileup < 0.99) that leaves 2."""
    m = (d.topo_score.to_numpy() > 0.99) & (d.muon_score.to_numpy() < 0.02) \
        & (d.pion_dif_score.to_numpy() < 0.01)
    if loose:
        return m & (d.muon_dif_score.to_numpy() < 0.46)
    return m & (d.muon_dif_score.to_numpy() < 0.40) & (d.pileup_score.to_numpy() < 0.99)


def radius(d):
    return np.sqrt(d.m_pred_pion_stop_x.to_numpy() ** 2 + d.m_pred_pion_stop_y.to_numpy() ** 2)


# ------------------------------------------------------------------ michel survivors, both chains
for loose in (True, False):
    rows = []
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
        et = j.m_truth_event_type.to_numpy().astype(np.int64)
        ok = ((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0)
        E, acc = base(j)
        k = acc & ok & topo_chain(j, loose)
        if k.sum() == 0:
            continue
        r = radius(j)[k]
        tt = j.m_pred_positron_time_consensus_ns.to_numpy()[k]
        for a, b in zip(tt, r):
            rows.append((round(float(a), 1), float(b)))
    lbl = "LOOSE (muDIF<0.46, no pileup cut)" if loose else "TIGHT (muDIF<0.40, pileup<0.99)"
    seen = {}
    for a, b in rows:
        seen.setdefault(a, b)
    print(f"\n=== survivors of the {lbl} chain: {len(rows)} rows, {len(seen)} distinct ===",
          flush=True)
    for a in sorted(seen):
        print(f"   t = {a:7.1f} ns   pion-stop radius = {seen[a]:6.3f} mm", flush=True)

# ------------------------------------------------------------------ does a fiducial cut sculpt?
rows = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
    g = ev.split("shard_")[-1].split("/")[0]
    tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS + ["is_pie"])
    if len(m) != len(t):
        continue
    j = paired(t, m)
    if j is None:
        continue
    j = j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)].reset_index(drop=True)
    E, acc = base(j)
    if acc.sum() == 0:
        continue
    rows.append(pd.DataFrame({"E": E[acc], "r": radius(j)[acc]}))
    if sum(len(x) for x in rows) > 900000:
        break
d = pd.concat(rows, ignore_index=True)
tail = d.E.to_numpy() < ES
peak = ~tail
r = d.r.to_numpy()
print(f"\naccepted pi -> e: {len(d):,}  ({int(tail.sum()):,} tail, {int(peak.sum()):,} peak)",
      flush=True)
print(f"  radius  peak q50 {np.median(r[peak]):.3f}  q95 {np.quantile(r[peak],0.95):.3f}  "
      f"q99 {np.quantile(r[peak],0.99):.3f} mm", flush=True)
print(f"  radius  TAIL q50 {np.median(r[tail]):.3f}  q95 {np.quantile(r[tail],0.95):.3f}  "
      f"q99 {np.quantile(r[tail],0.99):.3f} mm", flush=True)

print("\n=== induced tail bias of a fiducial radius cut ===", flush=True)
print(f"{'r <':>7} {'eff peak':>9} {'eff tail':>9} {'bias':>17}", flush=True)
out = []
for cut in (9.7, 9.27, 9.0, 8.5, 8.0, 7.0):
    ep = float((r[peak] < cut).mean()); et = float((r[tail] < cut).mean())
    npk, ntl = int(peak.sum()), int(tail.sum())
    ratio = et / ep if ep > 0 else 0.0
    err = 100 * ratio * np.sqrt(abs(1 - et) / max(et * ntl, 1) + abs(1 - ep) / max(ep * npk, 1))
    print(f"{cut:7.2f} {100*ep:8.3f}% {100*et:8.3f}%   {100*(ratio-1):+7.2f} +/- {err:5.2f}%",
          flush=True)
    out.append(dict(cut=cut, eff_peak=100 * ep, eff_tail=100 * et,
                    bias=100 * (ratio - 1), bias_unc=float(err)))

json.dump(dict(fiducial_scan=out), open(f"{OUT}/tail_fiducial_bias.json", "w"), indent=1)
print("\nALL DONE", flush=True)
