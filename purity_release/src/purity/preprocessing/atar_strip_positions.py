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
boundary. The older parquet does not record which strip it was; such boundary hits
(within 1e-5 mm of a boundary, about 4% of hits) go to the even-numbered of the two
strips, which is unbiased on average, and are counted in the run record.

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


def load_lut(path):
    d = np.load(path)
    view, centre, z = d["view"], d["centre"].astype(np.float64), d["z"].astype(np.float64)
    return {v: (np.unique(centre[view == v]), np.unique(z[view == v])) for v in (0, 1)}


def convert_table(table, lut):
    """Return (new table, outlier counts) for one row group."""
    cols = {}
    flat = {}
    for c in ("atar_x", "atar_y", "atar_z", "atar_view"):
        arr = table.column(c).combine_chunks()
        flat[c] = arr.flatten().to_numpy(zero_copy_only=False)
        cols[c] = arr
    offsets = cols["atar_x"].offsets.to_numpy()
    offsets = offsets - offsets[0]
    x, y, z = (flat[c].astype(np.float64) for c in ("atar_x", "atar_y", "atar_z"))
    view = flat["atar_view"].astype(np.int64)
    mx, my, mz = np.zeros_like(x), np.zeros_like(y), np.empty_like(z)
    boundary = 0
    for v in (0, 1):
        sel = view == v
        centres, planes = lut[v]
        coord = x if v == 0 else y
        snapped, tie = _snap(coord[sel], centres, tie_even=True)
        (mx if v == 0 else my)[sel] = snapped
        boundary += int(tie.sum())
        mz[sel], _ = _snap(z[sel], planes)
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
    return table, {"hits": int(len(x)), "boundary_hits": boundary, "out_of_strip": out_of_strip,
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
    totals = {"hits": 0, "boundary_hits": 0, "out_of_strip": 0, "out_of_plane": 0, "unknown_view": 0}
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
