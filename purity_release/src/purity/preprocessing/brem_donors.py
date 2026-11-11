"""Bank of real detached electron / photon hit clusters for transplanting into training events.

    python -m purity.preprocessing.brem_donors --input train.parquet --output donors.npz \\
        [--row-groups 36 37 38 39]

A donor cluster is the set of trigger-chain hits of one event that carry only electron or
photon bits (delta rays, Compton electrons, Bhabha electrons, brem conversions), lie within
2 ns of the positron's median hit time, and are farther than 0.5 mm (view coordinate and
z) from every positron hit of their view. Each hit is stored relative to the event's true
positron start, in the frame rotated about z so the positron's azimuth is 0, with its time
relative to the positron's median hit time, its energy and its PDG bits. purity.augment
places the clusters into other events relative to their positron.

Positions are the Geant4 deposit positions (atar_truth_x/y/z).
"""
import argparse

import numpy as np
import pyarrow.parquet as pq

from purity.preprocessing.hard_event_flags import E_DIST_MM, E_WIN_NS, _event, _flat, _group_median

COLS = ["atar_origin", "atar_pdg", "atar_view", "atar_t", "atar_E",
        "atar_truth_x", "atar_truth_y", "atar_truth_z",
        "truth_theta", "truth_phi",
        "truth_positron_start_x", "truth_positron_start_y", "truth_positron_start_z"]


def donors_from_table(table):
    n = table.num_rows
    origin, counts = _flat(table, "atar_origin")
    pdg = _flat(table, "atar_pdg")[0].astype(np.int64)
    view = _flat(table, "atar_view")[0].astype(np.int64)
    t = _flat(table, "atar_t")[0].astype(float)
    E = _flat(table, "atar_E")[0].astype(float)
    x, y, z = (_flat(table, c)[0].astype(float) for c in ("atar_truth_x", "atar_truth_y", "atar_truth_z"))
    ev = np.repeat(np.arange(n), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    u = np.where(view == 0, x, y)

    pos = (origin == 0) & ((pdg & 4) > 0)
    has_pos = np.bincount(ev[pos], minlength=n) > 0
    soft = (origin == 0) & ((pdg & 24) > 0) & ((pdg & 7) == 0)
    t_pos = _group_median(t[pos], ev[pos], n)
    cand = soft & has_pos[ev] & (np.abs(t - t_pos[ev]) < E_WIN_NS)

    theta = _event(table, "truth_theta")
    phi = _event(table, "truth_phi")
    ps = np.stack([_event(table, f"truth_positron_start_{c}") for c in "xyz"], 1)
    out = {k: [] for k in ("dx", "dy", "dz", "dt", "E", "pdg", "n", "theta")}
    for e in np.unique(ev[cand]):
        if not np.all(np.isfinite(ps[e])) or not np.isfinite(phi[e]):
            continue
        s = slice(starts[e], starts[e] + counts[e])
        pe, ce, ue, ze, ve = pos[s], cand[s], u[s], z[s], view[s]
        keep = []
        for i in np.flatnonzero(ce):
            m = pe & (ve == ve[i])
            d = np.min(np.hypot(ue[m] - ue[i], ze[m] - ze[i])) if m.any() else np.inf
            if d > E_DIST_MM:
                keep.append(i)
        if not keep:
            continue
        k = np.asarray(keep) + starts[e]
        rx, ry = x[k] - ps[e, 0], y[k] - ps[e, 1]
        c, sn = np.cos(-phi[e]), np.sin(-phi[e])
        out["dx"].append(c * rx - sn * ry)
        out["dy"].append(sn * rx + c * ry)
        out["dz"].append(z[k] - ps[e, 2])
        out["dt"].append(t[k] - t_pos[e])
        out["E"].append(E[k])
        out["pdg"].append(pdg[k])
        out["n"].append(len(k))
        out["theta"].append(theta[e])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--row-groups", type=int, nargs="*")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.input)
    acc = {k: [] for k in ("dx", "dy", "dz", "dt", "E", "pdg", "n", "theta")}
    for g in (a.row_groups or range(pf.num_row_groups)):
        part = donors_from_table(pf.read_row_group(g, columns=COLS))
        for k in acc:
            acc[k].extend(part[k])
        print(f"[donors] rg {g}: {len(part['n'])} clusters, {int(np.sum(part['n']))} hits", flush=True)
    n = np.asarray(acc["n"], np.int64)
    np.savez_compressed(
        a.output,
        offsets=np.concatenate([[0], np.cumsum(n)]),
        theta=np.asarray(acc["theta"], np.float64),
        **{k: np.concatenate(acc[k]) for k in ("dx", "dy", "dz", "dt", "E", "pdg")})
    print(f"[donors] {len(n)} clusters, {int(n.sum())} hits, hits per cluster: median "
          f"{np.median(n):.0f}, 90% {np.percentile(n, 90):.0f} -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
