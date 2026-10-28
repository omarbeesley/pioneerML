# PURITY tail reveal: selected model and provenance

This release preserves the current implementation, including changes newer than
the thesis. The thesis comparison, kept with the thesis figure recipes in the sibling
`thesis_reproduction/` folder, records historical differences; it does not prescribe
changes to make the code match the Chapter 5 prose.

The selected tail model is **`PURITY_TAIL10_epoch10.pth`**, completed on July 6,
2026. It is newer than the `PURITY_TAIL_PIDIF_recon` checkpoints in the original
repository's `model_weights/` folder. The final tail inference launch scripts
explicitly select TAIL10 epoch 10, and a production log confirms all checkpoint
keys loaded successfully. Epoch 10 also has the lowest validation loss in that
run: 0.728120, with training loss 0.698593.

The release ships a weights-only copy of this checkpoint (30 MB instead of 87 MB): the
optimizer and scheduler state is removed, and its tensors, metadata and model outputs
are identical to the original in `scratch_pipeline/model_weights/`
([validation](../provenance/validation_weights_only.json)).

Use [the model configuration](../configs/tail10_ep10.json). Its checkpoint hash,
source locations, and supporting file hashes are in
[the provenance manifest](../provenance/tail_artifacts.json).

## Purpose and model structure

PURITY reconstructs detector events. Its tail branch uses only ATAR information
to identify pion-to-positron signal and reject Michel decays, pileup, and pion or
muon decays in flight. The main reconstruction supplies energy, acceptance, and
timing for the final measurement. Removing calorimeter inputs from the tail
classifier reduces the direct dependence of the selection on calorimeter energy;
residual selection bias still needs to be measured.

The architecture is `PURITYTailModel` in [models/tail.py](../src/purity/models/tail.py):

- An ATAR backbone uses view-aware hit features, attention blocks, and pooled
  slice/event representations. It also produces particle labels, endpoints,
  pion stopping position, positron direction, and timing quantities.
- `PieTaggerHead` provides a holistic pion-to-positron score.
- `PieTopoBranch` independently builds a graph over time groups using geometry,
  energy, and time-free embeddings to provide a topology score.
- `MuonVetoHead`, `MuonDIFVetoHead`, `PionDIFVetoHead`, and `PileupVetoHead` provide
  background scores. Some heads also produce per-hit auxiliary predictions;
  the muon-DIF branch includes a muon travel-distance estimate.

The release exposes this single tail architecture. Its six selection heads are
listed above; the main reconstruction is defined separately in `models/main.py`.

## Inference convention

The final thesis tail predictions use **training mode with dropout 0.05 active**.
The ordinary `eval_tail.py` defaults instead disable dropout, so reproducing the
thesis convention requires the explicit mode and dropout settings in the release
configuration. This is established by the external `tail_drop_pie.sbatch`,
`tail_drop_michel_inwin.sbatch`, and `central_cache_drop.sbatch` launch recipes,
and by `logs/taildroppie_1712222_0.log`.

The constructor receives dropout 0.05, but the original `PieTaggerHead`
transformer hardcodes dropout 0.1. Both probabilities remain active in training
mode; this export preserves that historical behavior rather than making all
modules use 0.05.

The release config uses batch size 128, a 250-hit cap, one pass, and seed 0.
Historically, signal and Michel inference used batch size 128; the central DIF
cache used batch size 400. The historical runs did not record a random seed.
The release's seed 0 is a new choice for repeatable runs, **not a replay of the
original dropout realization**. Device, batch size, chunk boundaries, and software
versions can also affect a stochastic inference pass. Exact regeneration of the
published figures requires retaining the original prediction/cache artifacts.

## Training history that must remain explicit

The external tail training launch used 45 shards of 200,000 mixed events, intended
to rotate five shards per epoch, for ten epochs. The mixture recipe is 30% signal,
25% muon DIF, 15% pion DIF, and 30% Michel, before any generation or dataset
filtering. The validation launch selects `val_0_osl.parquet`. The source group
assignment and RNG seeds are recorded in `tail10_chain.sbatch`; source generation
jobs and mixture settings are recorded in `mix_shard_tail.py`.

**The selected checkpoint did not train on all nine million prepared events.**
The historical trainer replaced the dataset variable each epoch but failed to
rebuild its DataLoader. It reused the first five shards for all ten epochs. The
training log reports 997,380 loaded events after dataset filtering. The current
external `train_tail10.py`, modified after the checkpoint was trained, documents
and fixes the mistake at lines 524–532. Running that corrected trainer changes
the training population and will produce a new model. The selected checkpoint
and its original training population must be preserved as a historical result.
The release port, `purity.training.tail`, keeps the fix and offers
`--historical_tail10_loader` to replay the original behaviour; see
[training.md](training.md).
No exact retraining claim is made by this extraction.

The historical `*_osl.parquet` samples also underwent `origin_reslice.py`, which
splits ATAR slices using `(original time slice, simulation origin)` and recomputes
energy-weighted slice times. This uses simulation origin bookkeeping. It is part
of the historical benchmark definition and should not be silently replaced by a
different slicing procedure or described as independent of simulation truth.

The older local families are useful historical artifacts, but are not the final
selected model:

| Family in original repository | Last local artifacts | Interpretation |
| --- | --- | --- |
| `PURITY_TAIL` | June 2, 2026 | Early tail branch |
| `PURITY_TAIL_MUDIF` | June 3, 2026 | Added muon-DIF branch |
| `PURITY_TAIL_MUDIF_recon` | June 4, 2026 | Reconstruction-supervised experiments |
| `PURITY_TAIL_PIDIF_recon` | June 5, 2026 | Latest local family; best epoch 9, final epoch 10 |
| External `PURITY_TAIL10` | July 6, 2026 | Selected final tail model; best and final epoch 10 |

Some older prefixes were reused, leaving epoch files from different executions
under the same name. A numerically larger epoch suffix alone does not identify
the newest or best model.

## Final analysis inputs and selection

The current `unified_reco/tail_reveal_cache.py` is the strongest local record of
the final reconstructed-energy tail analysis. It joins these external artifacts:

| Sample | Tail prediction family | Main reconstruction family |
| --- | --- | --- |
| Signal | `norad_eval2_drop` | `filmg2_ep19tg_pie` |
| Michel in window | `michel_eval_inwin_drop` | `filmg2_ep19_michel_inwin` |
| Pion DIF | `tail_reveal_eval10/predictions_drop.parquet` | `tail_osl/pidif_osl_events.parquet` |
| Muon DIF | `tail_reveal_eval10/predictions_drop.parquet` | `tail_osl/mudif_osl_events.parquet` |

The final analysis therefore combines multiple main reconstruction generations.
The `tail_osl` launch recipe selects `main_hybridv2_nodrop_epoch12.pth`, with
`PURITY_GATE_MODE=best_xy`, beta 0.5, and dropout 0.0. It must not be silently
substituted with a later main checkpoint while claiming exact thesis replay.
Some analysis docstrings still describe older signal and Michel input paths;
the executable path expressions above are the audited inputs.

The cache builder defines reconstructed energy as predicted positron energy plus
predicted dead-material energy. Its implemented acceptance requires finite energy
in [0, 90] MeV, despite a nearby comment saying there is no upper cap. The prompt
window begins 2 ns after the 0.5th percentile onset of the accepted reconstructed
signal time and ends 32 ns after that onset. The final score selection is:

- Topology score greater than 0.99.
- Muon score less than 0.02.
- Muon-DIF score less than 0.46.
- Pion-DIF score less than 0.01.

The final chain does not apply the separate pileup score. The low/high energy
boundary is 56 MeV. `tail_rates.py` records the analysis's channel rate conventions
and the 0.006 physical signal-tail fraction used to rescale the simulated low bin;
these are analysis assumptions, not classifier outputs.

## Sample overlap and remaining reproduction work

The final cache builder removes Michel triggers identified by external
`michel_inwin_leakmask/*.parquet` masks. The included `purity.final_analysis.tail.overlap_masks` command (exported from
`michel_leak_audit.py`) traces
the overlap to generation job 1595422, source groups t000–t044, included in later
Michel pools. It constructs masks from four float32 truth values. Because the
historical loader actually reused only the first five training shards, the mask's
full intended-training-source exclusion is conservative relative to the documented
loaded population. Retain the masks and source-group definition for thesis replay;
do not replace them by a newly estimated overlap percentage. Comments in different
scripts quote inconsistent overlap percentages.

For Michel normalization, the cache builder records the generator-weight sum after
this overlap exclusion but before removing pion-to-positron, muon-DIF, and radiative
muon events. Removed channels lose their rate rather than having that rate
redistributed to the survivors.

Historical tail/main joins are positional, resynchronized by equality of float32
live-energy values. They can skip pieces on missing or mismatched inputs. A future
benchmark release should carry immutable event/source IDs, an explicit complete
input manifest, and checks for full join coverage. These properties cannot be
inferred from checkpoint weights.

This extraction provides a selected checkpoint, model code, inference settings,
and audit provenance. Full thesis regeneration still needs the historical data,
predictions, overlap masks, cached statistics, and a pinned runtime. The manifest
records their locations; the large analysis inputs are not bundled here.
