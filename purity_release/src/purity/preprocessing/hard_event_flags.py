"""Truth flags for the hard event topologies of the main-model training pool.

    python -m purity.preprocessing.hard_event_flags --input train.parquet --output flags_dir \\
        [--row-groups 0 1 ...]

Writes flags_dir/rg_XXXX.parquet per row group: one row per input row, in input order, with
the truth topology of the triggering decay chain. The trainer reads them to oversample the
topologies the michel identification fails on (purity_retrain_handoff.md, change 4):

  near90       true positron polar angle within 10 deg of 90
  muon_near90  true muon direction (start to stop) within 10 deg of 90
  gap          the positron's first hit in its worse view lies > 1 mm from the positron start:
               a transverse positron runs inside planes of the other view
  e_window     trigger-chain hits carrying only the electron bit (delta ray, Compton electron,
               Bhabha) within 2 ns of the positron's median hit time
  e_detached   any of those farther than 0.5 mm (view coordinate and z) from every positron
               hit of the same view

Positions are the Geant4 deposit positions (atar_truth_x/y/z), so the flags describe the
event, not its measurement. The flags apply to every channel alike (pie and michel), so the
oversampling distorts both the same way. Categories follow rule_reco.dev.failure_breakdown.
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

E_WIN_NS = 2.0
E_DIST_MM = 0.5
GAP_MM = 1.0
NEAR90_DEG = 10.0
PIXEL_COLS = ["atar_origin", "atar_pdg", "atar_view", "atar_t",
              "atar_truth_x", "atar_truth_y", "atar_truth_z"]
EVENT_COLS = ["truth_theta", "truth_acceptance", "truth_is_pie", "truth_has_muon",
              "truth_positron_start_x", "truth_positron_start_y", "truth_positron_start_z",
              "truth_muon_start_x", "truth_muon_start_y", "truth_muon_start_z",
              "truth_muon_stop_x", "truth_muon_stop_y", "truth_muon_stop_z"]


def _flat(table, col):
    arr = table.column(col).combine_chunks()
    return np.asarray(arr.flatten().to_numpy(zero_copy_only=False)), np.diff(np.asarray(arr.offsets))


def _event(table, col):
    return np.asarray(table.column(col).to_numpy(zero_copy_only=False), dtype=float)


def _group_median(values, groups, n):
    """Median of values per group (groups need not be sorted); NaN for empty groups."""
    order = np.lexsort((values, groups))
    v, g = values[order], groups[order]
    counts = np.bincount(g, minlength=n)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.full(n, np.nan)
    has = counts > 0
    lo = start[has] + (counts[has] - 1) // 2
    hi = start[has] + counts[has] // 2
    out[has] = 0.5 * (v[lo] + v[hi])
    return out


def flags_for_table(table):
    n = table.num_rows
    origin, counts = _flat(table, "atar_origin")
    pdg = _flat(table, "atar_pdg")[0].astype(np.int64)
    view = _flat(table, "atar_view")[0].astype(np.int64)
    t = _flat(table, "atar_t")[0].astype(float)
    x, y, z = (_flat(table, c)[0].astype(float) for c in ("atar_truth_x", "atar_truth_y", "atar_truth_z"))
    ev = np.repeat(np.arange(n), counts)
    u = np.where(view == 0, x, y)

    pos = (origin == 0) & ((pdg & 4) > 0)
    ele = (origin == 0) & ((pdg & 8) > 0) & ((pdg & 7) == 0)
    n_pos = np.bincount(ev[pos], minlength=n)
    has_pos = n_pos > 0

    # Gap: per view, the distance from the positron start to the nearest positron hit.
    ps = np.stack([_event(table, f"truth_positron_start_{c}") for c in "xyz"], 1)
    gaps = np.full((n, 2), np.inf)
    for w in (0, 1):
        m = pos & (view == w)
        d = np.hypot(u[m] - ps[ev[m], w], z[m] - ps[ev[m], 2])
        np.minimum.at(gaps[:, w], ev[m], d)
    gap_worst = np.where(has_pos, gaps.max(1), np.nan)

    # Electron-only trigger hits in the positron's time window, and which are detached.
    t_pos = _group_median(t[pos], ev[pos], n)
    win = ele & has_pos[ev] & (np.abs(t - t_pos[ev]) < E_WIN_NS)
    n_e_window = np.bincount(ev[win], minlength=n)
    n_e_detached = np.zeros(n, np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for e in np.flatnonzero(n_e_window):
        s = slice(starts[e], starts[e] + counts[e])
        pe, we, ue, ze, ve = pos[s], win[s], u[s], z[s], view[s]
        for i in np.flatnonzero(we):
            m = pe & (ve == ve[i])
            d = np.min(np.hypot(ue[m] - ue[i], ze[m] - ze[i])) if m.any() else np.inf
            n_e_detached[e] += d > E_DIST_MM

    theta = np.degrees(_event(table, "truth_theta"))
    mu = np.stack([_event(table, f"truth_muon_stop_{c}") - _event(table, f"truth_muon_start_{c}")
                   for c in "xyz"], 1)
    mu_len = np.linalg.norm(mu, axis=1)
    has_mu = (_event(table, "truth_has_muon") > 0.5) & np.isfinite(mu_len) & (mu_len > 1e-3)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_theta = np.where(has_mu, np.degrees(np.arccos(np.clip(mu[:, 2] / mu_len, -1, 1))), np.nan)

    return pa.table({
        "n_pos_hits": n_pos.astype(np.int32),
        "theta_deg": theta.astype(np.float32),
        "muon_theta_deg": mu_theta.astype(np.float32),
        "gap_worst_mm": gap_worst.astype(np.float32),
        "n_e_window": n_e_window.astype(np.int32),
        "n_e_detached": n_e_detached.astype(np.int32),
        "near90": has_pos & (np.abs(theta - 90) < NEAR90_DEG),
        "muon_near90": has_mu & (np.abs(mu_theta - 90) < NEAR90_DEG),
        "gap": has_pos & (gap_worst > GAP_MM),
        "e_window": n_e_window > 0,
        "e_detached": n_e_detached > 0,
        "truth_acceptance": _event(table, "truth_acceptance").astype(np.int8),
        "truth_is_pie": _event(table, "truth_is_pie").astype(np.int8),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="directory for rg_XXXX.parquet")
    ap.add_argument("--row-groups", type=int, nargs="*", help="default: all")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.input)
    groups = a.row_groups if a.row_groups else range(pf.num_row_groups)
    os.makedirs(a.output, exist_ok=True)
    for g in groups:
        table = pf.read_row_group(g, columns=PIXEL_COLS + EVENT_COLS)
        out = flags_for_table(table)
        assert out.num_rows == pf.metadata.row_group(g).num_rows
        path = os.path.join(a.output, f"rg_{g:04d}.parquet")
        pq.write_table(out, path + ".tmp")
        os.replace(path + ".tmp", path)
        f = out.to_pandas()
        print(f"[flags] rg {g}: {len(f)} rows; " + ", ".join(
            f"{c} {100 * f[c].mean():.2f}%" for c in ("near90", "muon_near90", "gap", "e_window", "e_detached")),
            flush=True)


if __name__ == "__main__":
    main()
