# Final analysis: tables, time fit and data production

This release includes model inference, data preparation, the final table and
time-fit recipes, and the tail cut chain. See [the analysis workflow](analysis.md)
for the complete command map. A successful small inference example checks that the
released model runs. It does not reproduce the full-statistics physics results.

The final main-model analysis used `main_hybridv2_filmg2_epoch19.pth`. The
`filmg2_ep19fl2` evaluation names identify the revised input production and
inference settings, not a different set of weights. Use the supplied
`configs/main_filmg2_ep19.json`: FiLM enabled, `best_xy` gate, beta 0.5,
5 ns energy time gate, zero dropout, and the 250-hit dataset cap. See
[main_model.md](main_model.md) and [tail_model.md](tail_model.md) for model details.

New results are reproducible by fixing the released source, selected checkpoint,
input population and analysis settings. The historical recipes below identify the
versions used for archived results; differences from those results do not by
themselves show a fault in the current implementation.

## Archived performance and acceptance commands

The `performance` command summarises reconstruction performance on the `fl2`
samples. The `acceptance_bias` command gives the acceptance-bias decomposition; the
495 ns edge is the quoted result (thesis Table 21).

After installing the package, give each command the event tables produced by the
main benchmark. These commands require NumPy and PyArrow, with no ROOT or GPU.
Quote globs so the program can record exactly which files it selected.

```bash
python -m purity.final_analysis.performance \
  --pie '/path/to/filmg2_ep19fl2_pie/shard_*/pie_eval_events.parquet' \
  --michel '/path/to/filmg2_ep19fl2_michel/mix_*/pimu_eval_events.parquet' \
  --output outputs/performance.json

python -m purity.final_analysis.acceptance_bias \
  --pie '/path/to/filmg2_ep19fl2_pie/shard_*/pie_eval_events.parquet' \
  --michel '/path/to/filmg2_ep19fl2_michel/mix_*/pimu_eval_events.parquet' \
  --output outputs/acceptance_bias.json
```

Inputs may also be explicit filenames or several quoted globs. Missing inputs,
missing required columns, and existing output files are errors; choose a new
output path for every run. The default minimum size of 1,000,001
bytes preserves the original scripts' `size > 1,000,000` filter. Use
`--min-shard-bytes 0` for small inference examples. Each command writes a second
`*.inputs.json` with selected and excluded filenames, sizes, and implementation
hashes. File size is a historical screening convention, not an integrity check;
PyArrow must also successfully read every selected table.

Both extractors preserve the original Michel purification: exclude event-type
bits `0x1` (direct pion-to-positron decays) and `0x200` (radiative muon decays),
and require true positron energy at most 55 MeV. This defines the simulated
population behind the results; these truth selections are not data selections.

The performance table uses a true fiducial pion stop, an in-window true trigger
positron, and `truth_htp > 0.5` for its denominator. Fiducial bounds are
`1.2 < z < 4.8 mm`, `|x| < 8 mm`, and `|y| < 8 mm`; the time window is
`[-300, 495] ns`. A chain is found when `pred_htp > 0.5`. Stop-distance
percentiles and angular residuals use found chains, without an IoU gate.
Angular widths are half the 84th-minus-16th percentile in 6-degree true-angle
bins with more than 200 entries. The legacy key `hw_at120_deg` uses the first
qualifying bin centred between 117 and 123 degrees, normally the 117-degree
bin. Its name is retained to compare with the archived table. Small examples
with too few angular entries return JSON `null` for unavailable widths.

The acceptance table uses generator weights. Reconstructed acceptance is
`pred_accepted >= 0.5`, reconstructed time in the selected window, and
`abs(t_reco - t_truth) < 5 ns`. Truth acceptance is `truth_acceptance > 0.5`
with true time at most the upper window edge. It reports both 500 and 495 ns,
including false-positive and false-negative window migrations. The direct
ratio of channel efficiencies and the first-order sum of the listed bias terms
are separate results. Preserve both when quoting the table.

Archived outputs are in [references/final](../references/final). Their numeric
formulas were compared against the original Python functions on 8,000 controlled
events with exact agreement, including both channels and both time windows.
That validation also checked missing-column rejection and sparse angular bins;
see [final_analysis_validation.json](../provenance/final_analysis_validation.json).
The release does not claim a new full-pool numerical reproduction.

## Historical populations differ by result

The external directories continued to acquire shards during analysis. A shared
prefix therefore does not identify a fixed statistical sample. The execution
logs give these actual populations, overriding stale counts in script comments:

| Product | Recorded run | Pion-to-positron shards | Michel shards |
|---|---|---:|---:|
| Separate fl2 performance summary | `perffl2_1777443.log` | 40 | 394 |
| Acceptance-bias table | `accfinal_1777323.log` | 40 | 390 |
| Revised time fit | `fl2timefit_1776963.log` | 34 | 377 |
| Directory inventory during release audit | 2026-09-10 | 40 | 394 |

The separate fl2 performance summary records median stop errors of 22.5123/22.6640 micrometres
and chain-failure rates of 0.05070/0.21943 percent for pion-to-positron/Michel.
The archived acceptance bias is +0.127327 percent for the 495 ns edge and
+0.085681 percent for 500 ns. These are archived reference values, not values
newly measured by this release build.

The time-fit combined files remain available externally as
`ep19fl2_pie_combined.parquet` (8,791,255 rows) and
`ep19fl2_michel_combined.parquet` (97,613,456 rows). The original concatenation
and fit log records those counts. Rebuilding them from all current shards would
change the sample. The exact historical 390-file acceptance subset has not been
recovered. Current filenames and sizes, source hashes, log hashes, and small
reference hashes are recorded in
[pipeline_artifacts.json](../provenance/pipeline_artifacts.json).

## Final spectrum and time-fit recipe

The final revised figure scripts live outside the original Git repository under
`/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/final_plots/`:

- `fig5_energy_spectrum_fixed.py` reads the `filmg2_ep19fl2` event shards.
- `fig8_timefit_fixed.py` reads the two combined snapshot files and imports the
  original `unified_reco.purity_analysis.io` and `measurement` modules.
- `build_fl2_combined.py` and `fl2_timefit.sbatch`, in the parent directory,
  concatenate shards and launch the fit. The launcher requested 220 GB RAM.

The final time-fit recipe is exported as an explicit command using the release's
`purity.measurement` modules:

```bash
python -m purity.final_analysis.time_fit \
  --pie /path/to/ep19fl2_pie_combined.parquet \
  --michel /path/to/ep19fl2_michel_combined.parquet \
  --output-dir outputs/final-time-fit
```

The output directory must be new or empty. It receives
`fig8_time_fit_fixed.pdf`, `.png`, and `.json`; the JSON records the central fits,
weighted histograms, closure metrics, and historical uncertainty recipe. The
inputs are **combined benchmark event summaries**, not raw hit fixtures. Use the
frozen 34-pion-shard / 377-Michel-shard combined snapshots to target the archived
figure; current shard globs describe a different population. The full fit loads
both tables into memory, and its original launcher requested 220 GB RAM.

The final energy-spectrum figure recipe is kept with the thesis figures in
`thesis_reproduction/`. The generic `purity.measurement.run_analysis` entry point
does not encode all the final time-fit figure's choices. The exported time-fit command preserves:

- `E = max(pred_positron_energy, 0) + max(pred_dead_energy, 0)`, split at
  56 MeV, with no upper energy cap; generator weights and pion-pool luminosity
  scaling by the post-purification Michel/pion row-count ratio.
- `pred_accepted >= 0.5` and consensus time within `[-300, 500] ns`.
  Selected wrong matches remain in the spectrum; truth time agreement within
  5 ns labels the clean component for closure and efficiency studies.
- 200 bins, a 4 ns pion-stop shift, and a fit using bins whose shifted centres
  are at least 5 ns. The high bin contains pion and muon templates; the low bin
  contains the muon template. Lifetimes are fixed in the analytic decay laws.
- A low-bin accidental fractional slope measured in its negative-time sideband;
  the high-bin accidental shape is flat. Each bin has its own sideband level.
  The actual `measure(..., pileup_slope_frac={'low': FRAC, 'high': 0.0})` call
  establishes this, despite older comments describing a shared slope.
- Weighted display uncertainties `sqrt(sum(w**2))`, 1,500 independent Gaussian
  histogram-fluctuation toys with seed 11, and a separately reported selected pion low-energy fraction.

The archived fit log reports `R = 1.182595e-4`, pool truth `1.183748e-4`,
raw relative closure `-0.097%`, and selected pion low-energy fraction `0.4004%`.
The comparison is against the non-radiative simulated pool, not the nominal
Standard Model branching ratio. The figure also computes a tail-adjusted
closure; the raw and adjusted quantities must not be interchanged. Its toy-fit
helper uses flat accidental subtraction while the low-bin central fit uses the
fitted slope. The historical plotted errors also divide extrapolated-amplitude
errors by fit-region truth integrals, and use full-selected-window truth errors
beside fit-region yields. The tail fraction is treated as exact; the anchor
uncertainty is the fixed historical 0.04 percentage points. These choices are
recorded explicitly and preserved; the toys are not an event bootstrap.

[Time-fit export provenance](../provenance/time_fit_export.json) identifies the
original script and modifications. [Synthetic validation](../provenance/validation_time_fit.json)
compares the original and exported calculations on 12,000 synthetic rows per
pool, including every central fit histogram, fitted yield, efficiency, weighted
error, Gaussian-toy error, and closure metric. This establishes numerical parity
of the port on that sample, not reproduction of the archived 100-million-event
fit. Tiny input sets can produce undefined statistics; the JSON represents them
as `null` and lists their paths.

## Full data-production chain

The source pipeline is:

```text
PIONEER simulation ROOT files, channel and train/validation/evaluation pools
  -> root_to_parquet.py: unmixed events and simulation truth
  -> generate_tail_reveal.py + pileup_mixer.py: pileup and radioactivity
  -> origin_reslice.py: slices partitioned by original slice and mixing origin
  -> pinned PURITY benchmark and tail evaluation
  -> event tables, matched model outputs, final tables, spectra and fits
```

The final main input launchers were `michel_flat_big2.sbatch` and
`pie_flat_chain2.sbatch` in the external pipeline directory. Both call
`generate_tail_reveal.py` with `--pileup_lam 0.24`, chunk size 100,000, and
the revised equilibrium pileup mixer. The mixer excludes direct pion-to-positron
donors from delayed accidental streams; generation drops radiative muon donors
by default. Michel uses open-window generation, `--no_replace`, and seed
`task_id + 9001`; pion-to-positron uses window enforcement and seed
`task_id + 101`. Each is origin-resliced before inference with the main model's
pinned settings. The Michel launcher retained mixed inputs for only its first
20 tasks. The complete evaluated raw population is therefore not supplied as a
ready-made release dataset.

Origin reslicing groups by `(atar_slice, atar_origin)` and recomputes
energy-weighted slice time. It depends on simulation mixing-origin bookkeeping;
this preparation is part of the reported simulated benchmark and must not be
described as reconstruction directly from detector measurements. The separate
`reslice_lib.py` spatial-clustering experiment is not this production operation.

The generation wrappers use separate channel-by-split source pools. Their
documentation states source-file separation between training, validation, and
evaluation. A complete ROOT-file split manifest was not recovered in this audit,
so independence has not been re-established by inspecting every source event.
Do not remix training pools into an evaluation dataset or transfer biased
training augmentation into the unbiased benchmarks. `generate_benchmarks.py`
is an older four-job wrapper with different source/output defaults; its defaults
do not recreate the final fl2 samples.

ROOT conversion needs PyROOT and the PIONEER simulation's dictionaries/runtime.
Mixing additionally needs `crystalLUT.npy` and `validLUT.npy`; the small production
lookup tables are bundled under `assets/luts`. The original ROOT pools and the
PIONEER simulation runtime remain external assets. Historical jobs sourced
`/home/obeesley/main/setenv.sh` and used the external `pytorch.sif` container.
The old `research_env_backup_11_19_25.yml` is a workstation environment export,
with an unrelated absolute prefix, not a tested lockfile for the final pipeline.

## Tail-analysis joins and remaining work

The August `tail_final_chain.sbatch` invokes `tail_final_reco.py`,
`tail_final_mudif.py`, and `tail_final_figs.py` after preparing train-mode tail
predictions. It sets `TAIL_MICHEL_NMAX=107`. These results combine datasets,
main-model predictions, and tail-model scores; the supplied classifier by itself
does not regenerate their cutflow or normalization. `tail_rates.py` supplies
channel-rate assumptions and the physical-tail rescaling convention.

Some historical joins resynchronize rows by float32 equality between tail
`deposited_energy` and main `truth_live_E`, or join positionally. This is not a
unique event key. New inference outputs carry `source_file` and `source_row`, which
join main and tail outputs of the same input exactly
([data contract](data_contract.md)); the archived prediction tables predate them.
Replacing the historical matching procedure should be validated against frozen
products before claiming the same cutflow.

[Tail analysis](tail_analysis.md) gives the cut-chain execution order;
[data preparation](data_preparation.md) covers the source-pool, mixing, reslicing and
combined-table commands.

Full archived-number reproduction still requires the intended per-result input
manifests and historical prediction realizations. Preserve or regenerate missing
raw inputs with their recorded seeds, then compare each output against the
appropriate reference product. Mutable directory names alone do not define the
historical statistical sample.
