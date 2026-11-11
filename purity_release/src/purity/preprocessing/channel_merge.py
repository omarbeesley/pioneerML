"""Merge hits that one detector channel would read as a single pulse, without ROOT.

    python -m purity.preprocessing.channel_merge --input mixed.parquet --output merged.parquet

Requires measured positions (atar_truth_x and lyso_truth_x present: atar_strip_positions and
lyso_crystal_positions have run), so that the same position is the same channel: an ATAR strip
(view, strip centre, plane) or a LYSO crystal. Two hits of one channel closer in time than the
window (ATAR 2 ns, LYSO 10 ns, root_to_parquet's) are merged, repeatedly, until no such pair is
left. Which hits may merge:

  ATAR   any two, whichever simulated event they come from (trigger, pileup donors).
  LYSO   two hits of the same simulated event (same atar/lyso_origin), and radioactivity
         (origin -1) with anything: a radioactivity hit joins the nearest-in-time hit of its
         crystal within the window. Trigger and pileup hits of one crystal are not merged with
         each other.

root_to_parquet merged each event's deposits in a single pass; a merged hit's energy-weighted
time could move within the window of another hit of the channel, which was never compared
again, so pairs remain in older parquets (about 18% of pie events have one in the LYSO). This
pass removes them and applies the cross-event rules above to mixed samples.

After merging, LYSO hits at or below the trigger threshold (root_to_parquet.LYSO_TRIG_E,
0.2 MeV) are dropped. Older parquets kept them (the converter's threshold never reached the
stored hits): about 20% of trigger and pileup LYSO hits, 0.2% of the LYSO energy. A fresh
conversion merges deposits before the threshold, so a small deposit next to a large pulse joins
it there but is dropped here; the two differ by about 1% of the LYSO hits.

A merged hit has the summed energy, energy-weighted times (t, truth_t) and truth positions, and
the OR of the PDG bits. It keeps the origin, slice and slice mean time of its higher-energy part,
preferring a non-radioactivity part. Row order and row groups are unchanged; within an event the
hits keep the order of their first part.
"""
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from purity.preprocessing._io import ensure_output, record_run
from purity.preprocessing.root_to_parquet import LYSO_TRIG_E

WINDOW = {"atar": 2.0, "lyso": 10.0}
THRESHOLD = {"lyso": LYSO_TRIG_E}                              # MeV, on the merged hit
RADIO = -1
MEAN = ("t", "truth_t", "truth_x", "truth_y", "truth_z")      # energy-weighted
LABEL = ("origin", "slice", "slice_mean_t")                    # from the label part


def _flat(table, col):
    a = table.column(col).combine_chunks()
    return a.flatten().to_numpy(zero_copy_only=False), np.diff(a.offsets.to_numpy())


def _attach_radio(ev, key, t, origin, window):
    """Merge group per hit: the origin, except that a radioactivity hit takes the origin of
    the nearest-in-time non-radioactivity hit of its crystal within the window."""
    group = origin.copy()
    radio = origin == RADIO
    if not radio.any() or radio.all():
        return group, 0
    o = np.lexsort((t, key, ev))
    ev_o, key_o, t_o, r_o = ev[o], key[o], t[o], radio[o]
    n = len(o)
    idx = np.arange(n)
    # within a channel sorted by time, the nearest other hit is the previous or the next one
    prev = np.maximum.accumulate(np.where(~r_o, idx, -1))
    nxt = np.minimum.accumulate(np.where(~r_o, idx, n)[::-1])[::-1]

    def distance(p):
        pc = np.clip(p, 0, n - 1)
        same = (p >= 0) & (p < n) & (ev_o[pc] == ev_o) & (key_o[pc] == key_o)
        return np.where(same, np.abs(t_o[pc] - t_o), np.inf), pc

    d_prev, p_prev = distance(prev)
    d_next, p_next = distance(nxt)
    best = np.where(d_next < d_prev, p_next, p_prev)
    sel = r_o & (np.minimum(d_prev, d_next) < window)
    group[o[sel]] = origin[o[best[sel]]]
    return group, int(sel.sum())


def merge_detector(cols, n, det, window, cross_event):
    """cols: flat per-hit arrays of one detector (keys without the detector prefix); n: hits per
    event. Returns (new flat arrays, new counts, merge statistics)."""
    ev = np.repeat(np.arange(len(n)), n)
    pos = np.round(np.c_[cols["x"], cols["y"], cols["z"]] * 1e4).astype(np.int64)
    if "view" in cols:
        pos = np.c_[pos, cols["view"].astype(np.int64)]
    _, key = np.unique(pos, axis=0, return_inverse=True)
    key = key.ravel().astype(np.int64)
    origin = cols["origin"].astype(np.int64)
    if cross_event:
        group, attached = np.zeros(len(ev), np.int64), 0
    else:
        group, attached = _attach_radio(ev, key, cols["t"], origin, window)
    c = {k: v.copy() for k, v in cols.items()}
    c["E"] = c["E"].astype(np.float64)
    first = np.arange(len(ev))
    stats = {"hits_in": int(len(ev)), "merges": 0, "merges_cross_origin": 0, "radio_attached": attached, "passes": 0}
    while True:
        o = np.lexsort((c["t"], group, key, ev))
        same = ((ev[o][1:] == ev[o][:-1]) & (key[o][1:] == key[o][:-1]) & (group[o][1:] == group[o][:-1])
                & (np.diff(c["t"][o]) < window))
        if not same.any():
            break
        stats["passes"] += 1
        # disjoint pairs: in each run of consecutive candidate pairs take every other one
        p = np.flatnonzero(same)
        run_start = np.r_[True, np.diff(p) > 1]
        rid = np.cumsum(run_start) - 1
        offset = np.arange(len(p)) - np.flatnonzero(run_start)[rid]
        p = p[offset % 2 == 0]
        i, j = o[p], o[p + 1]
        Ei, Ej = c["E"][i], c["E"][j]
        Et = Ei + Ej
        fi = np.where(Et > 0, Ei / np.where(Et > 0, Et, 1), 0.5)
        for k in MEAN:
            if k in c:
                c[k][i] = c[k][i] * fi + c[k][j] * (1 - fi)
        prio_i = Ei + 1e9 * (c["origin"][i] != RADIO)
        prio_j = Ej + 1e9 * (c["origin"][j] != RADIO)
        take_j = prio_j > prio_i
        stats["merges_cross_origin"] += int(np.sum(c["origin"][i] != c["origin"][j]))
        for k in LABEL:
            c[k][i] = np.where(take_j, c[k][j], c[k][i])
        c["pdg"][i] = c["pdg"][i] | c["pdg"][j]
        c["E"][i] = Et
        first[i] = np.minimum(first[i], first[j])
        keep = np.ones(len(ev), bool)
        keep[j] = False
        ev, key, group, first = ev[keep], key[keep], group[keep], first[keep]
        c = {k: v[keep] for k, v in c.items()}
        stats["merges"] += len(p)
    o = np.lexsort((first, ev))
    c = {k: v[o] for k, v in c.items()}
    stats["hits_out"] = int(len(o))
    return c, np.bincount(ev, minlength=len(n)), stats


def convert_table(table):
    out = table
    totals = {}
    for det in ("atar", "lyso"):
        names = [n for n in table.column_names if n.startswith(f"{det}_")
                 and str(table.schema.field(n).type).startswith("list")]
        flat, counts = {}, None
        for name in names:
            flat[name[len(det) + 1:]], counts = _flat(table, name)
        e_before = np.bincount(np.repeat(np.arange(len(counts)), counts), flat["E"], minlength=len(counts))
        merged, n_new, stats = merge_detector(flat, counts, det, WINDOW[det], cross_event=(det == "atar"))
        ev_new = np.repeat(np.arange(len(n_new)), n_new)
        e_after = np.bincount(ev_new, merged["E"], minlength=len(n_new))
        if not np.allclose(e_before, e_after, rtol=1e-9, atol=1e-9):
            raise SystemExit(f"{det}: energy per event not conserved by the merge")
        if det in THRESHOLD:
            keep = merged["E"] > THRESHOLD[det]
            stats["below_threshold"] = int((~keep).sum())
            stats["below_threshold_E"] = float(merged["E"][~keep].sum())
            merged = {k: v[keep] for k, v in merged.items()}
            n_new = np.bincount(ev_new[keep], minlength=len(n_new))
            stats["hits_out"] = int(keep.sum())
        offsets = pa.array(np.r_[0, np.cumsum(n_new)].astype(np.int32))
        for name in names:
            typ = table.schema.field(name).type.value_type
            arr = pa.ListArray.from_arrays(offsets, pa.array(merged[name[len(det) + 1:]]).cast(typ))
            out = out.set_column(out.schema.get_field_index(name), name, arr)
        totals[det] = stats
    return out, totals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--row-groups", type=int, nargs="*", help="Convert only these row groups (testing)")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.input)
    names = pf.schema_arrow.names
    for need in ("atar_truth_x", "lyso_truth_x", "atar_origin", "lyso_origin"):
        if need not in names:
            raise SystemExit(f"{a.input} lacks {need}: run atar_strip_positions and lyso_crystal_positions first")
    out = ensure_output(a.output)
    groups = a.row_groups if a.row_groups else list(range(pf.num_row_groups))
    totals = {d: {} for d in WINDOW}
    writer = None
    try:
        for g in groups:
            table, stats = convert_table(pf.read_row_group(g))
            for d, s in stats.items():
                for k, v in s.items():
                    totals[d][k] = totals[d].get(k, 0) + v if k != "passes" else max(totals[d].get(k, 0), v)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table, row_group_size=table.num_rows)
            print(f"  row group {g + 1}/{pf.num_row_groups}: {table.num_rows} events, {stats}", flush=True)
    except BaseException:
        if writer is not None:
            writer.close()
        out.unlink(missing_ok=True)
        raise
    writer.close()
    record_run(out, [a.input], __file__, {"row_groups": len(groups), "rows": pf.metadata.num_rows,
                                          "windows_ns": WINDOW, **{f"{d}_{k}": v for d in totals for k, v in totals[d].items()}})
    print(f"{out}: {totals}")


if __name__ == "__main__":
    main()
