"""
Apply the FAST3 (Channel 5, AC-coupled) response function to ATAR hit
energies in a mixed parquet file. Writes a new parquet with the same schema
but `atar_E` replaced by reconstructed (saturation-affected) energies.

Calibration: 30 keV ↔ 15 mV FAST3 output. The Channel 5 AC measured curve
is used as-is in the linear / sub-saturation regime, so this anchor pins
the input scale at V_in ≈ 16.46 mV / 30 keV ≈ 549 mV/MeV.

Coupling rationale: AC coupling matches production readout (DC-blocking
capacitor in series with the LGAD bias). Channel 5 is the LGAD-coupled
channel; Channel 1 was the bare test-pulse reference.

`--stretch X` (X > 1.0) builds a chip with X times the dynamic range by
scaling the entire 1x curve in BOTH V_in and V_out by X:

    V_out_X(V_in) = X * V_out_1x(V_in / X)

This is the natural "X times dynamic range, same small-signal gain" model:
    * Linear-regime gain dV_out/dV_in is unchanged at V_in -> 0, so the
      30 keV <-> 15 mV calibration anchor is preserved at the SAME V_in
      across all stretch values.
    * Saturation V_out scales by X (~752 mV -> 1504 mV at X=2.0).
    * Saturation V_in scales by X (~700 mV -> 1400 mV at X=2.0).
    * The curve is a smooth scaled copy of the measured 1x curve --
      no piecewise blending of measured + linearly-extrapolated regions
      and therefore no slope discontinuity at the original knee.

Usage:
    python apply_fast3_response.py --input  training_5_02/data.parquet \\
                                   --output training_5_02/data_fast3.parquet
    python apply_fast3_response.py --input  training_5_02/data.parquet \\
                                   --output training_5_02/data_fast3_2x.parquet \\
                                   --stretch 2.0
"""
import argparse
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# --- FAST3 Linearity_AC table ---
# Columns: [Ch1 mean (V), Ch1 std (V), Ch5 mean (V), Ch5 std (V)]
# Keys (V_in) are in mV; values are output amplitudes in V.
Linearity_AC = {
    10:[16.8e-3, 1.36e-3, 9.07e-3, 1.21e-3],
    15:[25.8e-3, 1.27e-3, 13.8e-3, 1.36e-3],
    20:[36.2e-3, 2.41e-3, 17.9e-3, 1.55e-3],
    30:[55.4e-3, 2.05e-3, 26.7e-3, 1.69e-3],
    50:[95.8e-3, 2.07e-3, 45.2e-3, 1.53e-3],
    75:[149e-3,  2.13e-3, 69.3e-3, 7.30e-3],
    100:[189e-3, 3.18e-3, 89.9e-3, 3.38e-3],
    150:[293e-3, 3.65e-3, 144e-3,  3.01e-3],
    200:[392e-3, 4.30e-3, 201e-3,  4.55e-3],
    250:[492e-3, 3.97e-3, 257e-3,  4.67e-3],
    300:[608e-3, 3.53e-3, 336e-3,  4.64e-3],
    350:[662e-3, 3.78e-3, 394e-3,  4.77e-3],
    400:[701e-3, 3.91e-3, 449e-3,  5.35e-3],
    500:[726e-3, 2.49e-3, 547e-3,  4.81e-3],
    600:[754e-3, 4.79e-3, 626e-3,  6.82e-3],
    700:[761e-3, 4.84e-3, 679e-3,  6.06e-3],
    800:[759e-3, 4.75e-3, 710e-3,  4.47e-3],
    900:[760e-3, 5.20e-3, 731e-3,  4.62e-3],
    1000:[758e-3, 4.95e-3, 743e-3, 5.73e-3],
    1500:[761e-3, 5.13e-3, 758e-3, 5.42e-3],
    2000:[762e-3, 4.40e-3, 752e-3, 3.89e-3],
}

# --- Calibration anchor ---
# Specified: 30 keV at FAST3 input ↔ 15 mV at FAST3 output.
TARGET_E_MEV    = 0.030   # 30 keV
TARGET_V_OUT_MV = 15.0    # 15 mV output

# Module-level lookup arrays — populated by configure_response() at startup.
INPUT_V_MV         = None
CH5_MEAN_MV        = None
CH5_STD_MV         = None
INPUT_V_MV_EXT     = None
CH5_MEAN_MV_EXT    = None
CH5_STD_MV_EXT     = None
ANCHOR_V_IN_MV     = None
INPUT_SCALE_MV_PER_MEV  = None
OUTPUT_SCALE_MEV_PER_MV = None
STRETCH            = 1.0


def configure_response(stretch=1.0):
    """Build the response lookup arrays and recompute the calibration anchor.

    `stretch` builds a chip with `stretch` times the dynamic range by scaling
    the measured 1x curve in both V_in and V_out by `stretch`:

        V_out_stretch(V_in) = stretch * V_out_1x(V_in / stretch)

    Equivalent operation on the lookup table: multiply both the V_in keys and
    the V_out values by `stretch`. The shape of the curve is unchanged --
    just stretched along both axes. Consequences:

      * Small-signal gain dV_out/dV_in at V_in -> 0 is unchanged, so the
        30 keV <-> 15 mV anchor resolves to the SAME V_in for every stretch
        (~16.46 mV) and INPUT_SCALE_MV_PER_MEV is stretch-independent.
      * Saturation V_out scales by `stretch`  (~752 mV  -> 1504 mV at 2x).
      * Saturation V_in scales by `stretch`  (~700 mV  -> 1400 mV at 2x).
      * The curve is everywhere smooth -- no piecewise blend of measured and
        linearly-extrapolated regions, so no slope discontinuity at the knee.

    Sigma is kept in absolute mV (electronics noise floor doesn't scale with
    dynamic range). It IS evaluated at V_in / stretch so the noise at a given
    V_in is what the 1x chip would see at the same fraction of full scale.
    """
    global INPUT_V_MV, CH5_MEAN_MV, CH5_STD_MV
    global INPUT_V_MV_EXT, CH5_MEAN_MV_EXT, CH5_STD_MV_EXT
    global ANCHOR_V_IN_MV, INPUT_SCALE_MV_PER_MEV, OUTPUT_SCALE_MEV_PER_MV
    global STRETCH

    STRETCH = float(stretch)
    _keys   = sorted(Linearity_AC.keys())

    v_in_orig    = np.asarray(_keys, dtype=np.float64)
    v_out_orig   = np.asarray([Linearity_AC[v][2] * 1000.0 for v in _keys], dtype=np.float64)
    v_std_orig   = np.asarray([Linearity_AC[v][3] * 1000.0 for v in _keys], dtype=np.float64)

    INPUT_V_MV   = STRETCH * v_in_orig
    CH5_MEAN_MV  = STRETCH * v_out_orig
    CH5_STD_MV   = v_std_orig.copy()        # noise floor unchanged in absolute mV

    # Origin extension so np.interp linearly extrapolates from (0, 0) for V_in below the table minimum.
    INPUT_V_MV_EXT  = np.concatenate([[0.0], INPUT_V_MV])
    CH5_MEAN_MV_EXT = np.concatenate([[0.0], CH5_MEAN_MV])
    CH5_STD_MV_EXT  = np.concatenate([[0.0], CH5_STD_MV])

    # Anchor calibration. Because both axes scale together, the linear-regime
    # gain is preserved and the anchor V_in is stretch-independent (~16.46 mV).
    ANCHOR_V_IN_MV          = float(np.interp(TARGET_V_OUT_MV, CH5_MEAN_MV_EXT, INPUT_V_MV_EXT))
    INPUT_SCALE_MV_PER_MEV  = ANCHOR_V_IN_MV / TARGET_E_MEV
    OUTPUT_SCALE_MEV_PER_MV = TARGET_E_MEV / TARGET_V_OUT_MV


# Initialize with default stretch=1.0 so the module's helpers work on import.
configure_response(stretch=1.0)


def apply_fast3_response_array(energies_mev, add_noise, rng):
    """Vectorized: per-hit energy (MeV) → reconstructed energy (MeV) via
    FAST3 Channel 5 AC-coupled response.

    Steps:
      1. E (MeV) → V_in (mV) via the linear input scale.
      2. V_in → V_out via piecewise-linear interpolation of the curve. With
         stretch > 1, the curve has a higher asymptote but is identical to
         the 1x curve below the natural knee.
      3. Optional Gaussian noise from the per-bin σ column (absolute mV;
         clamped to the table edges via np.interp's default behavior).
      4. V_out (mV) → E_reco (MeV) via the calibration scale.
    """
    e = np.asarray(energies_mev, dtype=np.float64)
    if e.size == 0:
        return e.astype(np.float32, copy=False)

    v_in_mv  = e * INPUT_SCALE_MV_PER_MEV
    v_out_mv = np.interp(v_in_mv, INPUT_V_MV_EXT, CH5_MEAN_MV_EXT)

    if add_noise:
        sigma = np.interp(v_in_mv, INPUT_V_MV_EXT, CH5_STD_MV_EXT)
        v_out_mv = v_out_mv + rng.normal(0.0, sigma)
        np.maximum(v_out_mv, 0.0, out=v_out_mv)   # no negative output

    e_reco_mev = v_out_mv * OUTPUT_SCALE_MEV_PER_MV
    return e_reco_mev.astype(np.float32, copy=False)


def transform_atar_E_column(atar_E_col, add_noise, rng):
    """Apply response per-event over a ragged-list column."""
    out = []
    for arr in atar_E_col:
        if arr is None:
            out.append(arr)
            continue
        np_arr = np.asarray(arr)
        if np_arr.size == 0:
            out.append(np_arr.astype(np.float32, copy=False))
            continue
        out.append(apply_fast3_response_array(np_arr, add_noise, rng))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to mixed input parquet.")
    parser.add_argument("--output", required=True, help="Path to write the modified parquet.")
    parser.add_argument("--chunk_size", type=int, default=10_000,
                        help="Events per processing chunk (default 10000).")
    parser.add_argument("--no_noise", action="store_true",
                        help="Disable Gaussian noise from per-bin σ (use mean curve only).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for noise.")
    parser.add_argument("--stretch", type=float, default=1.0,
                        help="Extend the dynamic range by this factor without "
                             "altering the sub-saturation curve. stretch=2.0 "
                             "doubles the saturation V_out (~752 → 1504 mV) "
                             "while keeping E_reco identical at energies that "
                             "didn't saturate at 1x. Calibration anchor V_in "
                             "is preserved across stretch values.")
    args = parser.parse_args()

    add_noise = not args.no_noise
    rng = np.random.default_rng(args.seed)

    configure_response(stretch=args.stretch)

    print(f"FAST3 Channel 5 AC-coupled response")
    print(f"  Calibration:    {TARGET_E_MEV*1000:.1f} keV ↔ {TARGET_V_OUT_MV:.2f} mV output")
    print(f"  Stretch factor: {STRETCH:.3f}x  (dynamic-range-only; sub-saturation curve unchanged)")
    print(f"  V_in scale:     {INPUT_SCALE_MV_PER_MEV:.4f} mV/MeV  (anchor V_in = {ANCHOR_V_IN_MV:.4f} mV)")
    print(f"  Reco scale:     {OUTPUT_SCALE_MEV_PER_MV*1000:.2f} keV/mV")
    print(f"  Noise:          {'ON (per-bin σ from table)' if add_noise else 'OFF (mean curve only)'}")
    print(f"  Saturation V_out (asymptote): {CH5_MEAN_MV[-1]:.1f} mV → E_reco_max = {CH5_MEAN_MV[-1] * OUTPUT_SCALE_MEV_PER_MV * 1000:.1f} keV")
    print()
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    pf = pq.ParquetFile(args.input)
    n_total = pf.metadata.num_rows
    print(f"Rows:   {n_total}")
    print(f"Chunk:  {args.chunk_size}")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    writer = None
    schema = None
    n_done = 0

    for batch in pf.iter_batches(batch_size=args.chunk_size):
        df = batch.to_pandas()
        df['atar_E'] = transform_atar_E_column(df['atar_E'], add_noise, rng)

        table = pa.Table.from_pandas(df)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(args.output, schema)
        else:
            table = table.cast(schema, safe=False)
        writer.write_table(table)
        n_done += len(df)
        print(f"  {n_done}/{n_total}", flush=True)

    if writer is not None:
        writer.close()
    print(f"\nDone. Wrote {n_done} rows to {args.output}.")


if __name__ == "__main__":
    main()