# Training the two models

The release includes the trainers that produced both selected checkpoints, the
recipes that produced their training data, and SLURM launchers for each step. The
trainers run from the release package with no research checkout. Model details and
the selection of each checkpoint are in [main_model.md](main_model.md) and
[tail_model.md](tail_model.md).

| Step | Command | Launcher |
| --- | --- | --- |
| Mix reconstruction training data | `purity.preprocessing.generate_benchmarks`, `origin_reslice` | `scripts/slurm/mix_train_main.sbatch` |
| Join and clean those shards | `purity.preprocessing.concat_clean` | `scripts/slurm/concat_train_main.sbatch` |
| Train the reconstruction model | `purity.training.main` | `scripts/slurm/train_main.sbatch` |
| Mix tail training data | `purity.preprocessing.mix_tail_shard`, `origin_reslice` | `scripts/slurm/mix_train_tail.sbatch` |
| Train the tail model | `purity.training.tail` | `scripts/slurm/train_tail.sbatch` |

Submit each launcher from the release root. They run `pytorch.sif` with the release
mounted at `/release`, as the inference recipe in the [README](../README.md) does, and
take their data and output directories from environment variables documented in each
file's header. Set `SIF` to use a different container. A job that dies with `Illegal
instruction` hit a compute node whose CPU does not support the container's build; the
historical launchers excluded such nodes, and `sbatch --exclude=<node>` does the same.
Both trainers exit non-zero when training fails, so `--dependency=afterok` chains stop.

## Reconstruction model

`purity.training.main` is the trainer of `main_hybridv2_filmg2`. Its source matches the
MD5 the filmg2 training log recorded for `train_purity.py`; the release changes only
imports, the class name, required paths, one optional weight override and the
provenance written into checkpoints. The model reads `PURITY_GATE_MODE`,
`PURITY_BEST_XY_BETA`, `PURITY_ROLE_FILM` and `PURITY_ENERGY_TIME_GATE_NS` when it is
imported. The trainer sets them to the training profile (`best_xy`, 0.5, 1, 0) unless
they are already exported, and records the values used.

The selected epoch-19 checkpoint came from two runs on the same data and seed:

| Phase | Run | Epochs | Differences |
| --- | --- | --- | --- |
| 1 | `main_hybridv2_filmg` | 1–4 | `--w_angle_decorr 0.5` |
| 2 | `main_hybridv2_filmg2` | 5–20 | resumes phase-1 epoch 4; `w_angle_decorr` 0.05; LRs pinned to 2.5e-4 / 2.5e-4 / 5e-4 |

Both phases use `--dropout 0.0 --weight_decay 1e-3 --epoch_mode chunked --chunk_seed
12345 --val_max_batches 1000` with the default 1,000,000-event epochs, stage 2 from
epoch 2, batch size 100 and the 250-hit cap. Chunked epochs walk the 40 row groups of
`train.parquet` in ten disjoint four-group chunks, so 20 epochs are two full passes.
Resuming restarts the step scheduler and overrides the optimizer learning rates from
the command line; that is why phase 2 pins them. The recorded base rate was 5e-4 for
epochs 1–4 and 2.5e-4 for epochs 5–9, then halved at epochs 10, 15 and 20. The
positron-recall weight ramps from 1 to 5 over epochs 5–9 inside the trainer.

```bash
J=$(sbatch --parsable --export=ALL,PHASE=1,DATA=/data/train10M_fixed,OUT=$PWD/outputs/weights \
    scripts/slurm/train_main.sbatch)
sbatch --dependency=afterok:$J --export=ALL,PHASE=2,DATA=/data/train10M_fixed,OUT=$PWD/outputs/weights \
    scripts/slurm/train_main.sbatch
```

Phase 1 was launched for 20 epochs and stopped during epoch 5; the launcher runs it for
4, which gives the same first four epochs because the schedule does not depend on the
total. Phase 1 ran a slightly earlier `train_purity.py` (log MD5 `d90331b4…`) that was not
preserved. The epoch-4 checkpoint stores its task weights, and they equal the current
defaults except `w_angle_decorr` = 0.5 (its role weight of 1.0 is the ramp's value at
epoch 4). `--w_angle_decorr 0.5` therefore reproduces the recorded phase-1 loss
configuration; it cannot certify that no other line of that source differed. The
model source has one change since training, the positron-energy time gate added for
inference, which is off when `PURITY_ENERGY_TIME_GATE_NS` is 0.

On the g3200 GPU a training epoch took 64–73 minutes; phase 2 ran in 20.0 hours.

## Tail model

`purity.training.tail` is the trainer of `PURITY_TAIL10`, ported from the external
`train_tail10.py`. TAIL10 used 10 epochs, the shard template
`train_{:03d}_osl.parquet` with five shards per epoch, `val_0_osl.parquet`, and default
arguments otherwise: end-to-end training, dropout 0.05, weight decay 1e-4 and no
backbone reconstruction losses. The learning rate was 5e-4 for epochs 1–2 and halved at
epochs 3, 6 and 9.

**TAIL10 trained on its first five shards only.** Its trainer rebuilt the dataset at
each shard rotation but not the DataLoader, so all ten epochs used shards 000–004
(997,380 events after the hit filter). The training log shows the rotation announcing
a new group every epoch while printing `train events:` once. The release trainer
rebuilds the loader, so by default it trains on all 45 shards and will produce a
different model. `--historical_tail10_loader` (launcher `HISTORICAL=1`) restores the
TAIL10 behaviour.

```bash
sbatch --export=ALL,DATA=/data/tail10,OUT=$PWD/outputs/weights scripts/slurm/train_tail.sbatch
sbatch --export=ALL,DATA=/data/tail10,OUT=$PWD/outputs/weights,HISTORICAL=1,NAME=PURITY_TAIL10_replay \
    scripts/slurm/train_tail.sbatch
```

The run took 11.5 hours on the g3200 GPU. The trainer sets no random seed, as in the
original, so even a historical-loader replay is a new realization.

## Training data

### Archived pools

The exact inputs of both checkpoints still exist under
`/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/`. Use them for a faithful
retrain; their sizes and row counts are in
[training_artifacts.json](../provenance/training_artifacts.json).

| Pool | Files | Events |
| --- | --- | --- |
| Reconstruction training | `train10M_fixed/train.parquet`, 40 row groups | 10,000,000 |
| Reconstruction validation | `train10M_fixed/val.parquet`, 4 row groups | 1,000,000 |
| Tail training | `tail10/train_000_osl.parquet` … `train_044_osl.parquet` | 45 × 200,000 |
| Tail validation | `tail10/val_0_osl.parquet`, `val_1_osl.parquet` | 2 × 200,000 |
| Tail pure-class evaluation | `tail10/eval_{pie,michel,mudif,pidif}_osl.parquet` | 4 × 150,000 |

### Regenerating them

The reconstruction pool is 40 training and 4 validation shards of 250,000 events. Each
is mixed by `generate_benchmarks --only training` (or `validation`) with
`TRAIN_VAL_OPTS` — 20% pion-to-positron triggers, 50% calorimeter-only and 10% biased
injection — from the training or validation unmixed split, with seed 9000+task
(9500+task for validation) and 50,000-event chunks, then origin-resliced.
`concat_clean` joins them and drops events with any non-finite hit coordinate, time or
energy; the archived pool lost none.

```bash
J=$(sbatch --parsable --export=ALL,UNMIXED=/data/unmixed_split,OUT=$PWD/outputs/train10M \
    scripts/slurm/mix_train_main.sbatch)
sbatch --dependency=afterok:$J --export=ALL,OUT=$PWD/outputs/train10M scripts/slurm/concat_train_main.sbatch
```

Each tail shard is mixed by `mix_tail_shard` from one generation-task group per
channel: 30% pion-to-positron, 25% muon DIF, 15% pion DIF and 30% Michel triggers, no
biased or calorimeter-only injection, 20 MHz LYSO radioactivity, with radiative muon
decays removed from the Michel and muon-DIF pools. The launcher reproduces the
historical group assignment (training task *i* uses Michel group *i*, pion group
*i* mod 23, muon-DIF group *i* mod 36, pion-DIF group *i* mod 16; validation and
evaluation use held-out groups) and seeds 7000+task.

```bash
sbatch --export=ALL,POOLS=/data/parquet,OUT=$PWD/outputs/tail10 scripts/slurm/mix_train_tail.sbatch
```

**A regenerated pool follows the recipe but is not the archived realization.** Both
archived pools were mixed in July 2026 by a pileup mixer whose accidental sampling was
revised afterwards (equilibrium old-muon/new-pion accidentals, no direct
pion-to-positron accidental donors; see [data_preparation.md](data_preparation.md)).
No commit preserves the July mixer. The release mixer is the revised one, so
regenerated shards differ in accidental composition as well as in random realization.

## Checkpoint contents

The two shipped checkpoints are weights-only copies. Both trainers resume them with
a fresh optimizer at the command-line learning rates. For an exact resume, which
needs the AdamW state, use the full originals in
`scratch_pipeline/model_weights/` (for phase 2, `main_hybridv2_filmg_epoch4.pth`).

Every checkpoint the release trainers write contains `model`, `optimizer`,
`scheduler`, `epoch`, the losses and (main model) `stage`, `global_step` and
`task_weights`, as before, plus `provenance`: the full argument set, the
import-time environment (main model) and the SHA-256 of the trainer source. Strict
inference loading reads only `model`, so these checkpoints load with the existing
profiles. The two selected checkpoints predate this field.

## Validation

Bounded CPU runs in `pytorch.sif` exercised every command above
([validation_training.json](../provenance/validation_training.json)). Each trainer
trained from scratch and resumed its released checkpoint for one more epoch (all keys
loaded), and every checkpoint it wrote has the same parameter keys and shapes as the
released one. The phase-1 override and the historical tail loader behaved as recorded.
Each data recipe produced a shard that loads in the dataset. These runs check the code
paths and compatibility, not training quality; no full-length retrain has been run.

Two defects of the source trainers surfaced and are fixed in the release. The tail
trainer built no DataLoader when given a single training file. Both trainers exited
with status 0 after a crash.
