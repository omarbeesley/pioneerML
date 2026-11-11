"""Convert a parquet to discrete LYSO crystal positions, without ROOT.

Parquets written before 2026-11-03 store, for a LYSO hit made by a tracked particle, the
energy-weighted Geant4 deposit centroid inside the crystal in `lyso_x/y/z`: a position the
calorimeter does not measure. Untracked deposits (mostly photons) already sit at the crystal
position (GeoHeader centre x 1.19, the converter's shower-maximum convention), as do
pileup-mixer hits, which are drawn from crystalLUT (the same positions). This command puts
every hit at its crystal's crystalLUT position and keeps the old values in
`lyso_truth_x/y/z`.
All other columns, the row order and the row groups are unchanged, so unmixed pools and
mixed, resliced samples convert alike.

The crystals point at the target, so a hit's crystal is the one whose front face, seen from
the origin, contains the hit's direction. The faces are the GeoHeader calorimeter layout
polygons in (theta, phi) (assets/luts/lyso_crystal_polys.npz, from
figures/purity_pedagogy/extract_calo_polys.py); the crystal positions are crystalLUT
(GeoHeader GetCentre x 1.19, row = volume id - LYSO_VID_BASE; identical to the position
root_to_parquet writes). Each hit is tested against the faces of its N_CAND crystals with the nearest centre directions, in order; a hit inside none of them
(a face edge, or a deposit leaking past the face projection) takes the nearest centre and
is counted, and above --max-fallback the conversion aborts.

    python -m purity.preprocessing.lyso_crystal_positions --input old.parquet --output new.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.path import Path as Polygon

from purity.preprocessing._io import ensure_output, record_run

DEFAULT_LUT_DIR = Path(__file__).resolve().parents[3] / "assets" / "luts"
N_CAND = 4
LYSO_VID_BASE = 300000
MAX_LUT_FACE_DEG = 0.5
CHUNK = 200_000


def _wrap(dphi):
    return (dphi + 180.0) % 360.0 - 180.0


class Crystals:
    def __init__(self, lut_dir=DEFAULT_LUT_DIR):
        lut_dir = Path(lut_dir)
        centres = np.load(lut_dir / "crystalLUT.npy")
        polys = np.load(lut_dir / "lyso_crystal_polys.npz", allow_pickle=True)
        self.vid = polys["vid"].astype(np.int64)
        # crystalLUT rows are volume id - LYSO_VID_BASE; checked below against the face centres
        self.centre = centres[self.vid - LYSO_VID_BASE].astype(np.float64)
        self.unit = self.centre / np.linalg.norm(self.centre, axis=1, keepdims=True)
        self.cph = polys["cph"].astype(np.float64)
        # each face unwrapped around its own centre phi, so faces on the +-180 seam stay whole
        self.faces = [Polygon(np.c_[th, c + _wrap(ph - c)])
                      for th, ph, c in zip(polys["theta"], polys["phi"], self.cph)]
        face_dir = np.c_[np.sin(np.radians(polys["cth"])) * np.cos(np.radians(self.cph)),
                         np.sin(np.radians(polys["cth"])) * np.sin(np.radians(self.cph)),
                         np.cos(np.radians(polys["cth"]))]
        self.lut_face_agreement_deg = float(np.degrees(np.arccos(np.clip(
            np.sum(face_dir * self.unit, 1), -1, 1))).max())
        if self.lut_face_agreement_deg > MAX_LUT_FACE_DEG:
            raise SystemExit(f"crystalLUT and the face polygons disagree by {self.lut_face_agreement_deg:.3f} deg")

    def assign(self, xyz):
        """Index into self.vid of each hit's crystal, and a mask of nearest-centre fallbacks."""
        n = len(xyz)
        out = np.empty(n, np.int64)
        fallback = np.zeros(n, bool)
        for s in range(0, n, CHUNK):
            p = xyz[s:s + CHUNK]
            r = np.linalg.norm(p, axis=1)
            u = p / np.maximum(r, 1e-9)[:, None]
            th = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1)))
            ph = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
            dots = u @ self.unit.T
            cand = np.argpartition(-dots, N_CAND, axis=1)[:, :N_CAND]
            cand = np.take_along_axis(cand, np.argsort(-np.take_along_axis(dots, cand, 1), 1), 1)
            got = np.full(len(p), -1, np.int64)
            for rank in range(N_CAND):
                todo = np.flatnonzero(got < 0)
                if len(todo) == 0:
                    break
                c_of = cand[todo, rank]
                for c in np.unique(c_of):
                    h = todo[c_of == c]
                    inside = self.faces[c].contains_points(np.c_[th[h], self.cph[c] + _wrap(ph[h] - self.cph[c])])
                    got[h[inside]] = c
            miss = got < 0
            got[miss] = cand[miss, 0]
            out[s:s + CHUNK] = got
            fallback[s:s + CHUNK] = miss
        return out, fallback


def convert_table(table, crystals):
    """Return (new table, counts) for one row group."""
    cols = {c: table.column(c).combine_chunks() for c in ("lyso_x", "lyso_y", "lyso_z")}
    offsets = cols["lyso_x"].offsets.to_numpy()
    offsets = offsets - offsets[0]
    xyz = np.stack([cols[c].flatten().to_numpy(zero_copy_only=False).astype(np.float64)
                    for c in ("lyso_x", "lyso_y", "lyso_z")], 1)
    idx, fallback = crystals.assign(xyz)
    centre = crystals.centre[idx]
    new = centre
    moved = np.linalg.norm(xyz - new, axis=1)

    def lists(values, like):
        return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(values).cast(like.type.value_type))

    for j, name in enumerate(("lyso_x", "lyso_y", "lyso_z")):
        table = table.set_column(table.schema.get_field_index(name), name, lists(new[:, j], cols[name]))
    for j, name in enumerate(("lyso_truth_x", "lyso_truth_y", "lyso_truth_z")):
        table = table.append_column(name, lists(xyz[:, j], cols[f"lyso_{'xyz'[j]}"]))
    return table, {"hits": int(len(xyz)), "fallback": int(fallback.sum()),
                   "already_at_crystal": int(np.sum(np.linalg.norm(xyz - new, axis=1) < 1e-3)),
                   "moved_over_30mm": int(np.sum(moved > 30.0)),
                   "moved_sum_mm": float(moved.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lut-dir", default=str(DEFAULT_LUT_DIR), help="crystalLUT.npy + lyso_crystal_polys.npz")
    ap.add_argument("--max-fallback", type=float, default=1e-3,
                    help="Abort if more than this fraction of hits lie inside none of the candidate faces")
    ap.add_argument("--row-groups", type=int, nargs="*", help="Convert only these row groups (testing)")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.input)
    names = pf.schema_arrow.names
    if "lyso_truth_x" in names:
        raise SystemExit(f"{a.input} already has lyso_truth_x; it uses crystal positions")
    missing = {"lyso_x", "lyso_y", "lyso_z"} - set(names)
    if missing:
        raise SystemExit(f"{a.input} lacks {sorted(missing)}")
    out = ensure_output(a.output)
    crystals = Crystals(a.lut_dir)
    print(f"{len(crystals.vid)} crystals; LUT centre vs face-centre direction, max {crystals.lut_face_agreement_deg:.4f} deg",
          flush=True)
    totals = {"hits": 0, "fallback": 0, "already_at_crystal": 0, "moved_over_30mm": 0,
              "moved_sum_mm": 0.0}
    groups = a.row_groups if a.row_groups else range(pf.num_row_groups)
    writer = None
    try:
        for g in groups:
            table, counts = convert_table(pf.read_row_group(g), crystals)
            for k in totals:
                totals[k] += counts[k]
            if counts["fallback"] / max(counts["hits"], 1) > a.max_fallback:
                raise SystemExit(f"row group {g}: fallback fraction {counts['fallback'] / counts['hits']:.2e} "
                                 f"> {a.max_fallback}: {counts}")
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table, row_group_size=table.num_rows)
            print(f"  row group {g + 1}/{pf.num_row_groups}: {table.num_rows} events, {counts}", flush=True)
    except BaseException:
        if writer is not None:
            writer.close()
        out.unlink(missing_ok=True)
        raise
    writer.close()
    totals["moved_mean_mm"] = totals.pop("moved_sum_mm") / max(totals["hits"], 1)
    record_run(out, [a.input, Path(a.lut_dir) / "crystalLUT.npy", Path(a.lut_dir) / "lyso_crystal_polys.npz"],
               __file__, {"row_groups": len(list(groups)), "rows": pf.metadata.num_rows,
                          **totals})
    print(f"{totals['hits']} hits -> {out}; {totals}")


if __name__ == "__main__":
    main()
