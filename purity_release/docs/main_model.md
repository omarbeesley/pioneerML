# Main PURITY model: the pinned reconstruction release

This release preserves the current implementation, including changes newer than
the thesis. The thesis comparison, kept with the thesis figure recipes in the sibling
`thesis_reproduction/` folder, records historical differences; it does not prescribe
changes to make the code match the Chapter 5 prose.

The final reconstruction analyses examined in this audit use **`main_hybridv2_filmg2_epoch19.pth`**, trained on 29 July 2026. Its architecture is `PURITYModel` in [models/main.py](../src/purity/models/main.py), with the Δz FiLM role head enabled. The research class was called `PURITYHybridModelV2`; the release contains just this selected main architecture. The portable configuration is [main_filmg2_ep19.json](../configs/main_filmg2_ep19.json); exact artifact metadata and SHA-256 fingerprints are in [main_artifacts.json](../provenance/main_artifacts.json).

PURITY reconstructs the pion stop position, positron direction, positron time, positron energy and event acceptance jointly from ATAR and calorimeter hits. Its detector graph backbones feed particle and slice classifiers, endpoint regression, calorimeter object condensation, and transformer heads that match the trigger decay chain and its shower. The tail model is a separate classifier used with these reconstructed quantities in the final rejection and energy-spectrum analysis.

## Which version is latest?

| Meaning | Model artifact | Evidence |
| --- | --- | --- |
| Pinned final reconstruction analysis | `main_hybridv2_filmg2_epoch19.pth` | Executable commands in `reco_ep19_pie_rest.sbatch:22` and `reco_ep19_michel_inwin.sbatch:31`; consumed by `tail_reveal_cache.py:116` and `:133`. |
| Last epoch of the latest located main training run | `main_hybridv2_filmg2_epoch20.pth` | Checkpoint metadata identifies epoch 20, global step 199754. |
| Best validation loss in that run | `main_hybridv2_filmg2_best.pth` | Checkpoint metadata also identifies epoch 20; validation loss −0.0598678188. |
| Older repository-local research snapshot | `PURITY_v3a_best.pth` | Metadata identifies epoch 16, from the May 2026 research freeze. `PURITY_v3a_epoch17.pth` is the last local v3a epoch. |

The pinned epoch 19 has global step 189767 and validation loss −0.0518728629. A later or lower-loss checkpoint would change the model being reproduced. This release selects epoch 19 because the final analysis launchers selected it. The earlier `ra5gr` and `nodrop` runs are not additional model implementations in this release. Some archived analysis inputs were produced with those earlier checkpoints; their provenance remains documented.

## Required inference settings

Set the environment **before importing the model module**, because these values are read at import time.

| Setting | Value |
| --- | --- |
| Model class | `purity.models.PURITYModel` |
| Constructor dropout | `0.0` |
| `PURITY_ROLE_FILM` | `1` |
| `PURITY_GATE_MODE` | `best_xy` |
| `PURITY_BEST_XY_BETA` | `0.5` |
| `PURITY_ENERGY_TIME_GATE_NS` | `5.0` |
| `PURITY_MAX_HITS` | `250` |
| Inference mode | `train`, under inference/no-gradient context |
| MC passes | `1` |
| Legacy benchmark CLI batch size | `200` |

The role FiLM setting changes parameter structure, so omitting it prevents a strict load. Dropout zero is also essential: the original benchmark's default is 0.1. The model retains `train()` mode to preserve the historical attention-mask path. The current source also disables PyTorch's fused attention fastpath. Every dropout module is disabled for this checkpoint.

The hit limit **drops events** with zero hits or more than 250 total ATAR plus calorimeter hits; it does not truncate them. Retain that selection when comparing stored performance. The release sets seed 0 explicitly. The audited historical launchers did not record an inference seed; the release seed does not claim to recover one.

## Dataset and inference suffixes

`filmg2_ep19tg` means epoch 19 inference with a 5 ns positron-energy time gate. The gate restricts the energy sum to hits near the consensus positron time. It is an inference setting, not another trained checkpoint.

`filmg2_ep19fl` and `filmg2_ep19fl2` use the **same epoch 19 checkpoint and the same 5 ns gate** on regenerated mixed datasets. In the audited `fl2` launchers, pion-to-positron donors are excluded from accidental streams. These tags therefore also encode data-production changes. They cannot be reproduced merely by choosing a different checkpoint. Some launcher comments still mention epoch 10 or 11, while their executable commands load epoch 19; the executable command is the evidence used here.

The repo's final tail reconstruction cache reads `filmg2_ep19tg_pie` and `filmg2_ep19_michel_inwin`. Its introductory docstring still names `nodrop_ep12`; the actual file-reading code selects `filmg2` epoch 19. Those tail cutflow samples must be kept distinct from the newer `fl2` samples used by other reconstruction analyses.

## Training lineage and source provenance

`filmg2` resumes `main_hybridv2_filmg_epoch4.pth`. Epochs 1–4 used the match-gated angle loss with an angle-decorrelation weight of 0.5. Epochs 5–20 reduced that weight to 0.05. The resume launcher explicitly supplied learning rates 2.5e-4 for the base and event groups and 5e-4 for the bias group, because its resume handling resets the scheduler and overrides learning rates.

The training recipe used `train10M_fixed/train.parquet` and `val.parquet`, dropout 0, weight decay 1e-3, chunked epochs of approximately one million events, row-group shuffle seed 12345, a 250-hit cap, and at most 1000 validation batches. The launcher ran through epoch 20. A faithful training reproduction must preserve both training phases, the source revision and the dataset provenance; starting today's script with a single set of default losses does not encode this history.

The original checkpoint contains model parameters, optimizer and scheduler states, epoch, stage, step counts, training and validation losses, and task weights. The release ships a weights-only copy without the optimizer and scheduler states (39 MB instead of 113 MB); its tensors, metadata and model outputs are identical to the original, which remains in `scratch_pipeline/model_weights/` ([validation](../provenance/validation_weights_only.json)). It does **not** contain the full invocation or import-time environment. Checkpoints written by the release trainer, `purity.training.main`, add both; [training.md](training.md) gives the two-phase recipe and its launcher. The external training log recorded source MD5 hashes. Current original `models.py`, `dataset.py` and `train_purity.py` match those hashes. Current `models_v2.py` differs, so its fingerprint is recorded as the inspected inference source, separately from the historical training hash. Strict parameter compatibility alone does not establish identical forward behavior across source revisions.

The original run artifacts and launchers are under:

```text
/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/
```

The training evidence is `train_filmg.sbatch`, `train_filmg2.sbatch`, `logs/train_purity_filmg2_1706750.log` and the corresponding checkpoint loss histories. The final inference evidence is `reco_ep19_pie_rest.sbatch`, `reco_ep19_michel_inwin.sbatch`, `eval_filmg2_ep19tg_pie.sbatch`, `michel_flat_big2.sbatch` and `pie_flat_chain2.sbatch`. Their paths, hashes and relevant line numbers are recorded in the artifact manifest.

## Verification performed

A CPU-only load in the existing `pytorch.sif` container, with PyTorch `2.7.0+cu128`, strictly loaded the epoch 19 checkpoint into the current original model source: **all parameter keys matched**, the model has **9,806,509 parameters**, and every dropout module has probability **0.0**. The audited archive metadata was also decoded without loading tensor storage, and the checkpoint files were fingerprinted.

A separate [export parity check](../provenance/validation_main.json) compared the original and exported dataset/model code on two pion-to-positron and two Michel fixture events, using the same checkpoint and seed on CPU with two threads. All graph inputs and all 55 output tensors per sample batch matched exactly, including the NaN-aware comparison policy (the tested tensors contained no NaNs). That migration comparison has already been performed; its report is retained. The shipped [standalone check](../tests/check_standalone.py) loads the local classes and checkpoints and runs the bundled examples without a research checkout.

The [model consolidation check](../provenance/validation_model_cleanup.json) additionally verifies that the new direct `PURITYModel` definition preserves checkpoint tensors, seeded initialization, module registration, constructor RNG state, and all outputs on the same small fixtures. The obsolete base-class forward was removed. Some unused registered parameters and overwritten initialization statements remain to preserve the original checkpoint and random-number sequence.

These checks establish checkpoint compatibility and exact export parity on the small fixtures. They do not reproduce the full benchmark, training run, or published numerical analysis. Preserving the mixed inputs, split and overlap masks, analysis selections, and cached numerical references is necessary for those comparisons.
