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
