# Preparing and freezing pipeline inputs

The release now includes the preprocessing stages needed between PIONEER ROOT
simulation and the model/analysis inputs. Run commands from the release directory
after installation, or set `PYTHONPATH=$PWD/src` in a compatible environment.
NumPy, pandas, PyArrow, and tqdm suffice for the non-ROOT stages. PyROOT and the
PIONEER simulation dictionaries are needed only when actually converting ROOT.
Every module imports safely and supports `--help` without starting work.

```text
ROOT simulation files grouped by channel and source split
  -> preprocessing.root_to_parquet / generate_unmixed
  -> unmixed source pools / preprocessing.assemble_pool
  -> preprocessing.generate_tail_reveal
  -> preprocessing.origin_reslice
  -> main and tail inference
  -> final_analysis.combine_events
  -> final tables, spectra, time fit, and tail analysis
```

All input, output, and LUT paths are explicit. Existing output files and their
manifests are refused. Successful commands write `OUTPUT.manifest.json`, recording
input filenames/sizes, effective arguments, and the executing module's hash. Large
input contents are not hashed automatically; freeze and archive the selected
files alongside these manifests for a publication-quality dataset snapshot.
Generated files belong under a new output directory for each production.

## Small mixing and reslicing example

`examples/unmixed/` contains the first 32 stored rows of each archived pion and
Michel evaluation source pool. `assets/luts/` contains the actual small crystal
geometry and valid-ID lookup arrays used by the mixer. Their provenance is in
`provenance/preprocessing_fixtures.json` and `preprocessing_export.json`.
These fixtures demonstrate the pipeline; they do not establish performance.

```bash
python -m purity.preprocessing.generate_tail_reveal \
  --only pie_eval --num_eval 12 --chunk_size 6 \
  --data_dir examples/unmixed --lut_dir assets/luts \
  --out_dir outputs/preparation/pie --seed 4101 --pileup_lam 0.24

python -m purity.preprocessing.generate_tail_reveal \
  --only michel_eval --num_eval 12 --chunk_size 6 \
  --data_dir examples/unmixed --lut_dir assets/luts \
  --out_dir outputs/preparation/michel --seed 9001 --pileup_lam 0.24 \
  --no_replace

python -m purity.preprocessing.origin_reslice \
  --input outputs/preparation/pie/tail_reveal_pie_eval.parquet \
  --output outputs/preparation/pie_resliced.parquet

python -m purity.preprocessing.origin_reslice \
  --input outputs/preparation/michel/tail_reveal_michel_eval.parquet \
  --output outputs/preparation/michel_resliced.parquet
```

The resliced files can replace `examples/pie.parquet` and `examples/michel.parquet`
in the main/tail inference commands in the README. Origin reslicing uses
simulation `atar_origin`: hits are grouped by their original time-slice and
mixing origin, then the slice's energy-weighted mean time is recomputed. This is
part of the simulated benchmark preparation, not an origin estimate made by a
real detector. The separate historical spatial-reslicing experiment is not used.

## ROOT conversion and source splits

For one explicit source pool:

```bash
python -m purity.preprocessing.root_to_parquet \
  --input '/path/to/pie/eval/*.root' \
  --output outputs/unmixed/unmixed_pie_eval.parquet \
  --seed 42 --shuffle_files --nprocs 1 --shard_size 100000
```

The converter reads the `sim` tree and `GeoHeader`, walks particle lineages,
merges deposits, applies the archived smearing/response rules, and writes ragged
ATAR/LYSO arrays and truth scalars. It does not run Geant4 or create ROOT files.
Provide the same PIONEER detector geometry, physics/generator setup, ROOT event
classes, and dictionaries as the intended production. These large simulation
assets and the `G4Pioneer` executable remain external.

For all channel-by-split directories, use the wrapper:

```bash
python -m purity.preprocessing.generate_unmixed \
  --root_base /path/to/root_pools --output_dir outputs/unmixed \
  --max_train 10000000 --max_val 500000 --max_eval 1000000 \
  --seed 42 --nprocs 1
```

The expected tree is `{pie,michel,mudif,pidif}/{train,val,eval}/*.root`.
`--only pie_eval`, for example, selects one job. This wrapper consumes an existing
split; it does not randomly repartition a shared pool. Freeze the source-file
assignment yourself and ensure that all pileup donors as well as trigger primaries
come from the intended split. The historical documentation states that train,
validation, and evaluation are disjoint at source-file level, but a complete
historical ROOT split manifest was not recovered.

The default ROOT skim drops pion-DIF; the wrapper supplies `--keep_pidif` only for
the dedicated `pidif` channel. Muon-DIF is retained and labelled. Direct conversion
supports `--keep_untriggered`; the wrapper exposes `--keep_untriggered_michel` for
Michel donors whose beam misses the DTAR trigger. This changes the donor pool and
must match the intended recipe. It is not automatically enabled by the export.

The conversion seed controls file shuffling and per-hit smearing. The wrapper
adds the sum of character codes of the job name to the CLI seed. Parallel workers
receive seed offsets and event-ID offsets; changing worker count changes the
conversion realization. With several workers, `--max_events` is an approximate
limit because each worker gets `ceil(max_events/nprocs)`. Use one worker for an
exact single-stream limit. Source row/event IDs must still be combined with source
file and mixing-realization identity; the converter's local integer IDs do not
establish globally unique downstream join keys by themselves.

## Final fl2 mixing recipe

The final launchers used `generate_tail_reveal`, despite the name also covering
the main-model evaluation. Its supported jobs are `training`, `validation`,
`pie_eval`, `michel_eval`, `michel_eval_in_win`, `mudif_eval`, and `pidif_eval`.
The final main comparison used the **open** Michel job and the window-enforced
pion job. The in-window Michel job is a differently conditioned tail-study
population, with different rate accounting.

The final main launchers used `--pileup_lam 0.24`, `--chunk_size 100000`, no biased
or calorimeter-only training injection, 20 MHz LYSO radioactivity, a 2 ns trigger
gap, and default removal of radiative-muon donors. The revised mixer samples
old-muon and new-pion accidentals consistently with equilibrium and excludes
direct pion-to-positron events from delayed accidental donor streams.
`--keep_rmd` deliberately restores radiative-muon donors and changes the recipe.

For production shard index `k`, the historical launcher passed seed `k+101` for
pion jobs and `k+9001` for Michel jobs. The generator then adds
`sum(ord(c) for c in job_name)` before setting the NumPy RNG. Both stages are
recorded in the output manifest; the launcher seed alone is not the effective
NumPy seed.

The source implementation of `--no_replace` initializes a new permutation inside
each `generate_batch` call. It prevents primary reuse **within a generated chunk**;
chunked runs can reuse a primary between chunks. This export preserves that
behavior. The historical comments claiming uniqueness over an entire production
were too broad. Preserve chunk size and job order when reproducing a realization.
The direct `pileup_mixer` CLI generates one batch; the final wrapper is preferred
because it also records the production job settings and removes radiative donors
by default.

The final Michel shard helper grouped 14 sorted unmixed parquet files into each
local pool. The portable equivalent is:

```bash
python -m purity.preprocessing.assemble_pool \
  --input '/path/to/parquet/michel/*.parquet' --index 0 --files-per-pool 14 \
  --output outputs/pool_000/unmixed_michel_eval.parquet
```

Put the corresponding pion and optional DIF evaluation pools beside it, named
`unmixed_pie_eval.parquet`, `unmixed_mudif_eval.parquet`, and
`unmixed_pidif_eval.parquet`, then use that directory as `--data_dir`.
The assembler records the exact selected source files. Expanding a mutable glob
later can change grouping and seed-to-pool assignments; freeze the inventory first.

`generate_benchmarks` is also included for the older standard four-job recipe.
It takes explicit `--data_dir`, `--out_dir`, and `--lut_dir`, but its augmentation,
window, and radiative-donor defaults differ from the final fl2 wrapper. Use it to
reproduce that historical configuration, not as a substitute for the final recipe.

## Optional FAST3 response

The archived FAST3 Channel 5 AC response transform is available independently:

```bash
python -m purity.preprocessing.apply_fast3_response \
  --input outputs/preparation/pie_resliced.parquet \
  --output outputs/preparation/pie_fast3.parquet \
  --seed 42 --stretch 1.0
```

It replaces `atar_E` using the embedded measured response table and optional
Gaussian electronics noise (`--no_noise` disables noise). It does not recompute
truth energies or the existing slice-mean times. Record its ordering explicitly.
The final fl2 launchers did not insert this transform, so do not add it when
reproducing the released main-model evaluation profile.

## Freeze combined benchmark events

The final time fit reads one event parquet per channel. Combine a fixed list or a
quoted glob of benchmark outputs as follows:

```bash
python -m purity.final_analysis.combine_events \
  --input '/path/to/selected/pie/shard_*/pie_eval_events.parquet' \
  --output outputs/snapshots/pie_combined.parquet

python -m purity.final_analysis.combine_events \
  --input '/path/to/selected/michel/mix_*/pimu_eval_events.parquet' \
  --output outputs/snapshots/michel_combined.parquet
```

The combiner streams batches, preserves sorted source order, and requires matching
column sets and types. It normalizes column order to the first input. Schema
mismatches fail before writing output instead of silently dropping a shard as the
old builder could. `--min-shard-bytes 1000001` reproduces the old size screen; the
default zero threshold accepts tiny demonstration outputs. The output manifest
records the exact included list. Do not reconstitute the historical time-fit
snapshots from every current shard: that run used 34 pion and 377 Michel shards,
while later table productions had more. See `reproduction.md` for each population.

## Validation and limits

`provenance/validation_preprocessing.json` records exact original/exported
Parquet-value agreement for seeded physical-pileup mixing on 12 real source events
in two chunks, exact origin-reslicing agreement across batch boundaries, exact
concatenation and grouped-pool selection, and FAST3 array parity with noise enabled
and disabled at stretches 1 and 2. Empty slices, zero-energy means, schema rejection,
existing-output protection, safe imports, and CLI help were also checked.

No full ROOT conversion, Geant4 run, or full-statistics input regeneration was
performed in the release audit. Historical source-split membership, missing mixed
input shards, unrecorded dropout seeds, and stable main/tail join identity remain
separate provenance requirements. These portable preprocessing commands provide
the executable stages; they do not manufacture absent historical data or certify
that newly generated samples reproduce every thesis number.
