"""Convert an older parquet to measured ATAR strip positions, without ROOT.

Parquets written before 2026-10-29 store the energy-weighted Geant4 deposit centroid in
`atar_x/y/z`. This command rewrites them to the current convention: those values move to
`atar_truth_x/y/z`, and `atar_x/y/z` become what the strips measure — the strip centre in
the measured coordinate (x for view 0, y for view 1), 0 in the unmeasured one, and the
plane centre in z. Each hit snaps to the nearest strip centre and plane of its own view
from the geometry table built by `build_atar_lut`. All other columns, the row order and
the row groups are unchanged, so unmixed pools and already mixed and resliced samples can
both be converted; positions play no part in mixing or reslicing.

Adjacent strips share their boundary, and the converter used each Geant4 step's end
point, so a deposit from a particle leaving a strip can have its centroid exactly on the
boundary (within 1e-5 mm; about 5% of hits). The older parquet does not record which
strip it was, but the event does: the particle crossed into the neighbouring strip and
left a hit there, and the converter merges deposits in one strip within 2 ns into a
single hit. So when exactly one of the two candidate strips already holds another hit
from the same simulated event (same view, plane and `atar_origin`) within 2 ns of
`atar_t`, the boundary hit goes to the other one. Against ROOT truth this decides 99.7%
of boundary hits, all correctly; the rest go to the even-numbered strip (unbiased on
average). Both counts are in the run record.

A hit farther than half a strip (0.1 mm) or half a plane (0.06 mm) from its snapped
position indicates a geometry mismatch; such hits are counted and, above
--max-outliers, abort the conversion.

    python -m purity.preprocessing.atar_strip_positions --input old.parquet --output new.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from purity.preprocessing._io import ensure_output, record_run

DEFAULT_LUT = Path(__file__).resolve().parents[3] / "assets" / "luts" / "atar_strips.npz"
HALF_PITCH, HALF_PLANE = 0.1, 0.06
TOL = 1e-4


BOUNDARY_TOL = 1e-5


def _snap(values, grid, tie_even=False):
    """Nearest grid value for each entry of `values` (`grid` sorted), and a mask of
    entries equidistant from two grid values; with `tie_even` those take the
    even-indexed one."""
    i = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    lo, hi = grid[i - 1], grid[i]
    d_lo, d_hi = np.abs(values - lo), np.abs(hi - values)
    tie = np.abs(d_lo - d_hi) < 2 * BOUNDARY_TOL
    pick_lo = d_lo <= d_hi
    if tie_even:
        pick_lo = np.where(tie, (i - 1) % 2 == 0, pick_lo)
    return np.where(pick_lo, lo, hi), tie


def _resolve_boundary(event, origin, view, plane, strip, t, tie, n_strips):
    """For boundary hits (`tie`), return +1 / -1 to choose the upper / lower candidate
    strip when exactly one candidate is occupied by a non-boundary hit of the same event,
    origin, view and plane within 2 ns, else 0. `strip` holds each hit's lower candidate
    index for boundary hits and its own index otherwise."""
    def key(ix, s):
        return ((((event[ix] * 1000 + origin[ix]) * 2 + view[ix]) * 64 + plane[ix]) * (n_strips + 2)) + s + 1
    occ = np.flatnonzero(~tie)
    k_occ = key(occ, strip[occ])
    order = np.argsort(k_occ, kind="stable")
    k_sorted, t_sorted = k_occ[order], t[occ][order]
    idx = np.flatnonzero(tie)

    def occupied(cand):
        k = key(idx, cand[idx])
        lo = np.searchsorted(k_sorted, k, "left")
        hi = np.searchsorted(k_sorted, k, "right")
        found = np.zeros(len(idx), bool)
        width = int((hi - lo).max()) if len(idx) else 0
        for j in range(width):
            pos = lo + j
            ok = pos < hi
            found |= ok & (np.abs(t_sorted[np.minimum(pos, len(t_sorted) - 1)] - t[idx]) < 2.0)
        return found

    occ_lo = occupied(strip)
    occ_hi = occupied(strip + 1)
    choice = np.zeros(len(t), np.int8)
    choice[idx] = np.where(occ_lo & ~occ_hi, 1, np.where(occ_hi & ~occ_lo, -1, 0))
    return choice


def load_lut(path):
    d = np.load(path)
    view, centre, z = d["view"], d["centre"].astype(np.float64), d["z"].astype(np.float64)
    return {v: (np.unique(centre[view == v]), np.unique(z[view == v])) for v in (0, 1)}


def convert_table(table, lut):
    """Return (new table, outlier counts) for one row group."""
    cols = {}
    flat = {}
    names = table.column_names
    for c in ("atar_x", "atar_y", "atar_z", "atar_view", "atar_t") + (("atar_origin",) if "atar_origin" in names else ()):
        arr = table.column(c).combine_chunks()
        flat[c] = arr.flatten().to_numpy(zero_copy_only=False)
        cols[c] = arr
    offsets = cols["atar_x"].offsets.to_numpy()
    offsets = offsets - offsets[0]
    x, y, z = (flat[c].astype(np.float64) for c in ("atar_x", "atar_y", "atar_z"))
    view = flat["atar_view"].astype(np.int64)
    t = flat["atar_t"].astype(np.float64)
    origin = (flat["atar_origin"].astype(np.int64) + 1 if "atar_origin" in flat
              else np.zeros(len(x), np.int64))
    event = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    mx, my, mz = np.zeros_like(x), np.zeros_like(y), np.empty_like(z)
    tie = np.zeros(len(x), bool)
    strip = np.zeros(len(x), np.int64)      # lower candidate index for boundary hits
    plane = np.zeros(len(x), np.int64)
    even = np.empty(len(x))
    for v in (0, 1):
        sel = view == v
        centres, planes = lut[v]
        coord = x if v == 0 else y
        snapped, tie_v = _snap(coord[sel], centres, tie_even=True)
        even[sel] = snapped
        tie[sel] = tie_v
        i = np.clip(np.searchsorted(centres, coord[sel]), 1, len(centres) - 1)
        own = np.searchsorted(centres, snapped)
        strip[sel] = np.where(tie_v, i - 1, own)
        mz[sel], _ = _snap(z[sel], planes)
        plane[sel] = np.searchsorted(planes, mz[sel])
    n_strips = max(len(lut[0][0]), len(lut[1][0]))
    choice = _resolve_boundary(event, np.clip(origin, 0, 999), np.clip(view, 0, 1), plane, strip, t, tie, n_strips)
    for v in (0, 1):
        sel = view == v
        centres = lut[v][0]
        resolved = sel & (choice != 0)
        picked = centres[np.clip(strip + (choice > 0), 0, len(centres) - 1)]
        final = np.where(resolved, picked, even)
        (mx if v == 0 else my)[sel] = final[sel]
    boundary = int(tie.sum())
    resolved_by_neighbour = int((choice != 0).sum())
    out_of_strip = int(np.sum(np.abs(np.where(view == 0, x - mx, y - my)) > HALF_PITCH + TOL))
    out_of_plane = int(np.sum(np.abs(z - mz) > HALF_PLANE + TOL))
    unknown_view = int(np.sum((view != 0) & (view != 1)))

    def lists(values, like):
        typ = like.type.value_type
        return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(values).cast(typ))

    for name, values in (("atar_x", mx), ("atar_y", my), ("atar_z", mz)):
        table = table.set_column(table.schema.get_field_index(name), name, lists(values, cols[name]))
    for name, values, like in (("atar_truth_x", x, cols["atar_x"]), ("atar_truth_y", y, cols["atar_y"]),
                               ("atar_truth_z", z, cols["atar_z"])):
        table = table.append_column(name, lists(values, like))
    return table, {"hits": int(len(x)), "boundary_hits": boundary,
                   "boundary_resolved_by_neighbour": resolved_by_neighbour,
                   "boundary_even_fallback": boundary - resolved_by_neighbour, "out_of_strip": out_of_strip,
                   "out_of_plane": out_of_plane, "unknown_view": unknown_view}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lut", default=str(DEFAULT_LUT), help="ATAR strip geometry from build_atar_lut")
    ap.add_argument("--max-outliers", type=float, default=1e-4,
                    help="Abort if more than this fraction of hits lie outside their snapped strip or plane")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.input)
    names = pf.schema_arrow.names
    if "atar_truth_x" in names:
        raise SystemExit(f"{a.input} already has atar_truth_x; it uses measured positions")
    missing = {"atar_x", "atar_y", "atar_z", "atar_view"} - set(names)
    if missing:
        raise SystemExit(f"{a.input} lacks {sorted(missing)}")
    out = ensure_output(a.output)
    lut = load_lut(a.lut)
    totals = {"hits": 0, "boundary_hits": 0, "boundary_resolved_by_neighbour": 0, "boundary_even_fallback": 0,
              "out_of_strip": 0, "out_of_plane": 0, "unknown_view": 0}
    writer = None
    try:
        for g in range(pf.num_row_groups):
            table, counts = convert_table(pf.read_row_group(g), lut)
            for k in totals:
                totals[k] += counts[k]
            bad = (counts["out_of_strip"] + counts["out_of_plane"] + counts["unknown_view"]) / max(counts["hits"], 1)
            if bad > a.max_outliers:
                raise SystemExit(f"row group {g}: outlier fraction {bad:.2e} > {a.max_outliers}: {counts}")
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table, row_group_size=table.num_rows)
            print(f"  row group {g + 1}/{pf.num_row_groups}: {table.num_rows} events, {counts['hits']} hits", flush=True)
    except BaseException:
        if writer is not None:
            writer.close()
        out.unlink(missing_ok=True)
        raise
    writer.close()
    record_run(out, [a.input, a.lut], __file__, {"row_groups": pf.num_row_groups,
                                                 "rows": pf.metadata.num_rows, **totals})
    print(f"{pf.metadata.num_rows} events, {totals['hits']} hits -> {out}; outliers {totals}")


if __name__ == "__main__":
    main()
