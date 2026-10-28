#!/usr/bin/env python3
"""Find Bhabha-scatter candidates: events where delta/Bhabha electron hits
drag a naive line fit of the positron direction away from the truth.

For each event in a mixed-events parquet:
  - positron hits  = ATAR hits with the positron bit (0b100)
  - electron hits  = ATAR hits with the electron bit (0b1000) and NOT the
    positron bit (the knocked-out / pair partner, distinct from the track)
  - require a clean single visible chain (pileup contributes no ATAR hits),
    an attached electron branch (closest approach to the positron track
    < 0.7 mm), and enough hits to fit
  - fit the positron direction two ways with per-view least squares
    (x(z) from the XZ view, y(z) from the YZ view -> direction vector):
        fit_pos : positron hits only
        fit_all : positron + electron hits (what a naive fit would use)
  - candidate if the two fits disagree by more than --min_bias degrees:
    the Bhabha branch genuinely distorts the naive reconstruction

Output: tidy parquet tables (prefix_hits / prefix_meta), same convention as
attn_slice_probe. Runs on a compute node -- it loads a parquet.
"""
import argparse

import numpy as np
import pandas as pd

PDG_PION, PDG_MUON, PDG_POSITRON, PDG_ELECTRON = 1, 2, 4, 8


def fit_dir(zw, cw, vw, ew):
    """Direction from per-view weighted line fits; None if underconstrained.

    zw/cw: z and transverse coordinate per hit; vw: view flag (0 = XZ, 1 = YZ);
    ew: deposit, used as the fit weight the way a naive energy-weighted
    reconstruction would.
    """
    out = []
    for v in (0, 1):
        m = vw == v
        if m.sum() < 3 or np.ptp(zw[m]) < 0.2:
            return None
        b = np.polyfit(zw[m], cw[m], 1, w=np.sqrt(ew[m]))[0]
        out.append(b)
    d = np.array([out[0], out[1], 1.0])
    return d / np.linalg.norm(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="output parquet prefix")
    ap.add_argument("--scan_events", type=int, default=40000)
    ap.add_argument("--min_e_hits", type=int, default=3)
    ap.add_argument("--min_bias", type=float, default=8.0, help="degrees")
    ap.add_argument("--max_candidates", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    print(f"[scan] {len(df)} events; columns: {sorted(df.columns)[:40]}", flush=True)
    n = min(len(df), args.scan_events)

    hits_rows, meta_rows = [], []
    for i in range(n):
        row = df.iloc[i]
        pdg = np.asarray(row["atar_pdg"], dtype=np.int64)
        if pdg.size < 12:
            continue
        origin = np.asarray(row.get("atar_origin", np.zeros(pdg.size)))
        if (origin != 0).sum() > 0:      # keep it clean: no visible pileup
            continue
        x = np.asarray(row["atar_x"], float)
        y = np.asarray(row["atar_y"], float)
        z = np.asarray(row["atar_z"], float)
        E = np.asarray(row["atar_E"], float)
        v = np.asarray(row["atar_view"], float)
        sl = np.asarray(row["atar_slice"], np.int64)
        tt = np.asarray(row.get("atar_truth_t", np.zeros(pdg.size)), float)

        is_pos = (pdg & PDG_POSITRON) > 0
        is_ele = ((pdg & PDG_ELECTRON) > 0) & ~is_pos
        if is_ele.sum() < args.min_e_hits or is_pos.sum() < 6:
            continue

        c = np.where(v > 0.5, y, x)      # the measured transverse coordinate
        # attached branch: electron hits start on the positron track
        dmin = np.min(np.sqrt(
            (z[is_ele][:, None] - z[is_pos]) ** 2
            + (c[is_ele][:, None] - c[is_pos]) ** 2))
        if dmin > 0.7:
            continue

        d_pos = fit_dir(z[is_pos], c[is_pos], v[is_pos], E[is_pos])
        m_all = is_pos | is_ele
        d_all = fit_dir(z[m_all], c[m_all], v[m_all], E[m_all])
        if d_pos is None or d_all is None:
            continue
        bias = np.degrees(np.arccos(np.clip(np.dot(d_pos, d_all), -1, 1)))
        if bias < args.min_bias:
            continue

        for k in range(pdg.size):
            hits_rows.append((i, float(x[k]), float(y[k]), float(z[k]),
                              float(E[k]), int(v[k]), int(sl[k]),
                              int(pdg[k]), bool(is_pos[k]), bool(is_ele[k]),
                              float(tt[k])))
        meta_rows.append((i, float(bias), int(is_pos.sum()), int(is_ele.sum()),
                          float(d_pos[0]), float(d_pos[1]), float(d_pos[2]),
                          float(d_all[0]), float(d_all[1]), float(d_all[2])))
        print(f"[scan] event {i}: fit bias {bias:5.1f} deg   "
              f"{int(is_pos.sum())} e+ hits, {int(is_ele.sum())} e- hits",
              flush=True)
        if len(meta_rows) >= args.max_candidates:
            break

    if not meta_rows:
        raise SystemExit("[scan] no candidates -- relax --min_bias or scan more")
    pd.DataFrame(hits_rows, columns=[
        "event", "x", "y", "z", "E", "view", "slice", "pdg", "is_pos",
        "is_ele", "truth_t"
    ]).to_parquet(args.out + "_hits.parquet", index=False)
    pd.DataFrame(meta_rows, columns=[
        "event", "fit_bias_deg", "n_pos", "n_ele",
        "dpos_x", "dpos_y", "dpos_z", "dall_x", "dall_y", "dall_z"
    ]).to_parquet(args.out + "_meta.parquet", index=False)
    print(f"[scan] wrote {args.out}_{{hits,meta}}.parquet "
          f"({len(meta_rows)} candidates)", flush=True)


if __name__ == "__main__":
    main()
