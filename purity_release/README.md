# PURITY for PIONEER

The selected PURITY reconstruction and tail-reveal models, with everything needed to
run them, benchmark them, perform the R_e/μ analysis and retrain them: data
preparation, inference, benchmark diagnostics, the measurement, the tail cut chain and
both trainers. Copy this whole directory, including `checkpoints/`. Large simulation
and prediction tables live elsewhere and are passed to each command explicitly.

New results should identify their source revision, checkpoint, inputs and settings.

## Models

| Model | Definition | Selected checkpoint |
| --- | --- | --- |
| PURITY reconstruction | [models/main.py](src/purity/models/main.py): `PURITYModel` | `main_hybridv2_filmg2_epoch19.pth` |
| PURITY tail reveal | [models/tail.py](src/purity/models/tail.py): `PURITYTailModel` | `PURITY_TAIL10_epoch10.pth` |

[models/layers.py](src/purity/models/layers.py) contains shared graph and attention
components. The files in `checkpoints/` are weights-only copies of the original
training checkpoints: the optimizer and scheduler state is removed, and every model
tensor and all metadata are unchanged, as are both models' outputs
([validation](provenance/validation_weights_only.json)). The full originals, needed
only to resume training exactly, are in
`/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/model_weights/`.

The main model reconstructs the triggering decay chain, pion stop and positron
kinematics from ATAR and LYSO hits. The tail model uses ATAR information to select
signal and reject Michel, pileup and decay-in-flight backgrounds.

Main epoch 20 was the latest trained checkpoint; **epoch 19 is included because the
final reconstruction analysis used it**. TAIL10 epoch 10 is the selected tail
checkpoint. [Main model details](docs/main_model.md) and
[tail model details](docs/tail_model.md) record their settings and history. The
profiles in `configs/` pin the inference settings; use them.

**ATAR positions.** Since 2026-10-29 the converter stores the measured strip
position: strip centre, plane centre in z, 0 in the coordinate a strip does not
measure. The Geant4 deposit centroid moved to `atar_truth_x/y/z`. Both selected
checkpoints were trained on earlier samples whose `atar_x/y/z` held that continuous
truth centroid, so their ATAR position resolutions are optimistic. Applying them to
newly converted samples is out of their training distribution; retrain on reconverted
samples for new results ([data contract](docs/data_contract.md)).

## Run

The validated environment is the production `pytorch.sif` container; versions are
recorded in [environment.json](provenance/environment.json). From this directory:

```bash
export PURITY_IMAGE=/path/to/pytorch.sif
apptainer exec --cleanenv --contain --pwd /tmp \
  --bind "$PWD":/release --env PYTHONPATH=/release/src \
  --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
  "$PURITY_IMAGE" python3 -m purity \
  --config /release/configs/main_filmg2_ep19.json \
  --data /release/examples/pie.parquet /release/examples/michel.parquet \
  --output /release/outputs/main
```

For tail inference, use `configs/tail10_ep10.json` and `outputs/tail`. Add `--nv`
for GPU use. The runner checks checkpoint hashes, sets the profile and seed, and
records inputs, settings, source hashes and completion in `run.json`. Main inference
writes event, slice and calorimeter tables; tail inference writes
`predictions.parquet`. Main `--dump-atar-hits` adds per-hit tables. Main events and
tail predictions both carry `source_file` and `source_row`, which join the two outputs
of the same input exactly ([data contract](docs/data_contract.md)).

With the recorded dependencies installed natively, `python -m pip install -e .`
provides the same `python -m purity` interface. Native installation has not been
validated from scratch.

In either runtime, from the release root:

```bash
python scripts/run_example_analysis.py --output-dir outputs/example-analysis
python -I tests/check_standalone.py --output outputs/standalone-check.json
python scripts/verify_release.py
```

The example workflow mixes a few events, runs both models and draws the benchmark
plots. The standalone check imports local code, strictly loads both checkpoints and
runs the examples. The integrity check verifies every packaged file against
`release_manifest.json`.

## Contents

| Location | Contents | Guide |
| --- | --- | --- |
| `src/purity/preprocessing/` | ROOT conversion, pool assembly, pileup mixing, reslicing, training-data assembly | [data preparation](docs/data_preparation.md) |
| `src/purity/benchmark.py`, `eval_tail.py` | Main and tail inference (run through `python -m purity`) | [data contract](docs/data_contract.md) |
| `src/purity/diagnostics/` | Benchmark plots, tail-score ROCs, per-head tables, cut scans, counting checks, event displays | [analysis](docs/analysis.md) |
| `src/purity/measurement/` | R_e/μ measurement | [analysis](docs/analysis.md) |
| `src/purity/final_analysis/` | Performance and acceptance-bias tables, time fit, combined event tables | [final analysis](docs/reproduction.md) |
| `src/purity/final_analysis/tail/` | Tail cut chain: overlap masks, cutflow cache and plots | [tail analysis](docs/tail_analysis.md) |
| `src/purity/training/`, `scripts/slurm/` | Trainers of both checkpoints and their SLURM launchers | [training](docs/training.md) |
| `examples/`, `assets/` | Small event samples and detector lookup tables | |
| `references/final/` | Reference outputs of the performance and acceptance-bias tables | [final analysis](docs/reproduction.md) |
| `provenance/` | Checkpoint and source hashes, training and run evidence, validation reports | |

Both models match their pre-cleanup outputs exactly on the small CPU fixtures,
including initialization and checkpoint compatibility. These examples check
integration; they do not measure performance.

The thesis figure recipes and their bundled caches were moved out of this release on
2026-10-28; they are kept in the sibling `thesis_reproduction/` folder. Historical tail
inference used dropout with unrecorded seeds; new runs use seed 0 and cannot replay
those predictions bit for bit.
