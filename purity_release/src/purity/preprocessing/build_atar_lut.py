"""Extract the ATAR strip geometry from a simulation file's GeoHeader.

Writes an NPZ with one entry per ATAR strip volume: `volume_id`, `view` (0 measures x,
1 measures y), `centre` (the strip centre in its measured coordinate, mm) and `z` (its
plane centre, mm), as float32 values read from the geometry header exactly as
root_to_parquet stores them. `atar_strip_positions` uses it to convert older parquets
without ROOT. Needs PyROOT and the PIONEER simulation runtime.

    python -m purity.preprocessing.build_atar_lut --input sim.root --output assets/luts/atar_strips.npz
"""
import argparse

import numpy as np

from purity.preprocessing._io import ensure_output, record_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Simulation ROOT file containing GeoHeader")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    import ROOT
    out = ensure_output(a.output)
    geo = ROOT.TFile(a.input).Get("GeoHeader")
    if not geo:
        raise SystemExit(f"no GeoHeader in {a.input}")
    atar = ROOT.PIDetectorType.kAtar
    # ATAR strip volume ids are 100000-104799 in the PIONEER geometry.
    vids = [v for v in range(100000, 105000) if geo.GetDetectorType(v) == atar]
    view = np.array([0 if abs(geo.GetPsi(v)) < 0.1 else 1 for v in vids], dtype=np.int8)
    centre = np.array([geo.GetX(v) if w == 0 else geo.GetY(v) for v, w in zip(vids, view)], dtype=np.float32)
    z = np.array([geo.GetZ(v) for v in vids], dtype=np.float32)
    np.savez(out, volume_id=np.array(vids, dtype=np.int32), view=view, centre=centre, z=z)
    record_run(out, [a.input], __file__, {
        "strips": len(vids), "planes": int(len(np.unique(z))),
        "strips_per_view": [int((view == 0).sum()), int((view == 1).sum())]})
    print(f"{len(vids)} strips, {len(np.unique(z))} planes -> {out}")


if __name__ == "__main__":
    main()
