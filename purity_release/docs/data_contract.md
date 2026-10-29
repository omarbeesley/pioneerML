# Data contract and reconstruction flow

The research flow is PIONEER simulation ROOT files → `root_to_parquet.py` →
mixing/detector-response processing → time reslicing → mixed Parquet events →
`PURITYDataset` → model inference → event/slice/calorimeter or tail-score tables →
physics selection, efficiencies, fits, and plots. Conversion and production
mixing commands are included in `purity.preprocessing` and documented in
[data preparation](data_preparation.md). They require the source simulation
assets and are not silently performed by the inference runner.

The examples are the first 16 rows of each documented final mixed/resliced
simulation input. The source paths, original row counts, and example hashes are
in `provenance/examples.json`. They are too small for efficiency, resolution,
tail-flatness, or R_e/mu claims.

`src/purity/dataset.py` is the authoritative loader for the mixed-event schema.
ATAR arrays include strip coordinates, view, energy, time, time-slice membership,
and simulation labels. LYSO arrays contain positions, deposited energy, time, and
simulation-origin/association fields. Truth scalars are carried for evaluation.
Coordinates and normalizations are defined in `src/purity/constants.py`; the
loader normalizes physical quantities for the models and the benchmark restores
units for its output tables. Do not replace this loader with the older notebook
or graph-record pipeline.

ATAR positions are what the strips measure. Strips are 0.2 mm wide on a 0.2 mm
pitch, in 48 planes 0.12 mm thick. For a view-0 strip `atar_x` is the strip centre
and `atar_y` is 0; for a view-1 strip `atar_y` is the strip centre and `atar_x` is 0;
`atar_z` is the plane centre. The energy-weighted Geant4 deposit centroid in the
strip is kept as truth in `atar_truth_x`, `atar_truth_y` and `atar_truth_z`, which the
loader uses for the slice-endpoint training targets. The model inputs use only the
measured positions.

**Parquets converted before 2026-10-29 store the truth centroid in `atar_x/y/z`.**
The measured coordinate there is continuous within the strip and z is continuous
within the plane, so those samples give the model sub-strip position information a
real detector does not provide. Both selected checkpoints were trained and evaluated
on such samples; their ATAR position resolutions are optimistic, and running them on
newly converted samples is out of their training distribution. New results need
reconverted samples and retrained models.

Older parquets do not need the ROOT files to be reconverted:

```bash
python -m purity.preprocessing.atar_strip_positions --input old.parquet --output new.parquet
```

moves `atar_x/y/z` to `atar_truth_x/y/z` and snaps each hit to the nearest strip centre
and plane of its own view, using the geometry table `assets/luts/atar_strips.npz`
(built from a simulation file by `purity.preprocessing.build_atar_lut`). It keeps every
other column, the row order and the row groups, so it applies equally to unmixed pools
and to mixed, resliced samples; positions play no part in mixing or reslicing. About
5% of hits have their deposit centroid exactly on the boundary between two strips,
because the converter used Geant4 step end points: the particle left the strip there and
crossed into its neighbour. The neighbour therefore holds its own hit at the same time,
and one strip cannot hold two hits within the converter's 2 ns merge window, so a
boundary hit goes to the candidate strip that has no other hit from the same simulated
event (same view, plane and `atar_origin`) within 2 ns. Against fresh ROOT conversions
this decides 99.7% of boundary hits, all correctly; the rest go to the even-numbered
strip. Overall 3 of 59,927 hits in 1,000 events differ from a ROOT conversion. A 250,000-event
row group takes about 60 s.

The pinned profiles apply the historical **250 total hit** limit; rows with zero
hits or more than 250 ATAR+LYSO hits are removed. That selection affects reported
efficiencies. Main outputs carry diagnostics for retained rows. With
`--max-events`, main inference takes the first N rows after hit filtering, while
tail inference randomly samples rows before filtering. These capped runs do not
select the same events and must not be joined by row position. The main limit
does not bound input-loading memory; use a small input parquet for a quick check.

The production reslicer uses simulation-origin information. This release
reproduces that simulation-based input convention; it does not establish that
the same segmentation is achievable in real detector data. Do not substitute
unresliced or differently resliced inputs and assume checkpoint performance is
unchanged.

Main inference names each sample deterministically as
`sample_NNN_<input-stem>` and writes `<tag>_events.parquet`,
`<tag>_slices.parquet`, and `<tag>_lyso_clusters.parquet`. It includes reconstructed
pion stop, direction, energy, dead-material correction, positron timing,
acceptance, and corresponding truth/diagnostic fields. Tail inference concatenates
inputs in the supplied order into `predictions.parquet`, containing classifier
scores and evaluation truth quantities.

Historical main/tail joins were positional, sometimes resynchronized using
float32 deposited-energy equality. An energy value is not a unique event ID.
Do not join predictions from arbitrary runs or different filtering/batch/sample
settings by row position and assume event identity. Preserve original shard
membership and ordering when reproducing a recorded analysis.

New outputs carry a stable event key. Main `<tag>_events.parquet` and tail
`predictions.parquet` both contain `source_file`, the input path as given to the run,
and `source_row`, the event's row in that input file before the hit filter. Join main
and tail outputs of the same input on these two columns. The key is defined whenever
the input is read in file order: whole-file reads, tail `--shard_size` streaming and
`--n_chunks` slices. A tail run whose `--max_events` is below the input size subsamples
row groups at random and writes `source_row` = −1; a main run's `--max-events` keeps the
first events and their keys. `event_id` remains the event's position after the hit filter,
which the slice, calorimeter and hit tables reference. Archived prediction tables
predate the key; their joins still need the historical procedure.

The general measurement code uses reconstructed live energy plus the predicted
dead-energy correction, consensus timing, generator weights, an acceptance
selection and a 56 MeV split. Its defaults are not a substitute for the
specific final thesis figure/fit configuration. In particular, the 495 ns upper
trim belongs to final acceptance/performance tables, while the recorded final
time fit uses a 500 ns boundary. The reproduction document gives exact recipes.
