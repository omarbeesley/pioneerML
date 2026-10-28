# Tail cut chain

The tail cut chain combines main-model reconstruction (energy, acceptance, consensus
time) with tail-model scores to remove Michel, pion-DIF and muon-DIF backgrounds from
the pion-to-positron low-energy tail. The commands below build its cutflow cache from
saved prediction tables and draw the cutflows. They need NumPy, Pandas, PyArrow and
Matplotlib, but no GPU, ROOT or model. Run from the release root with `PYTHONPATH=src`,
or in the container with the release and input data mounted.

| Command | Reads | Writes |
| --- | --- | --- |
| `purity.final_analysis.tail.overlap_masks` | Michel predictions, main events, raw Michel pieces | `<piece>.parquet` overlap masks |
| `purity.final_analysis.tail.cache` | Prepared input directory (below) | `tail_reveal_cache.npz` |
| `purity.final_analysis.tail.cutflows` | `tail_reveal_cache.npz` | `tail_reveal_{michel,pidif,mudif}` cutflows |
| `purity.final_analysis.tail.fig_cutflow_spectra` | `tail_reveal_cache.npz` | four-panel cutflow rows `cutflow_spectra_{michel,pidif,mudif}` |
| `purity.final_analysis.tail.mudif_tradeoff` | signal and muon-DIF prediction tables | muon-DIF contamination versus tail bias |
| `purity.final_analysis.tail.bias_vs_theta` | a tail prediction table | tail bias versus true positron angle |

## The cut chain

Energy is `pred_positron_energy + pred_dead_energy` from the main model, with the
original inclusive 90 MeV cap and a 56 MeV tail/peak split. The stages are:

1. main-model acceptance;
2. prompt time window: onset plus [2, 32] ns, where onset is the 0.5th percentile of
   finite accepted signal consensus times;
3. pion-to-positron topology score greater than 0.99;
4. muon veto: muon score below 0.02 and muon-DIF score below 0.46, with the pion-DIF
   veto (score below 0.01) for the full chain.

The pileup score is not part of this chain. `rates.py` holds the channel-rate
assumptions, the 56 MeV split and the 0.006 physical signal-tail rescaling; these are
analysis inputs, not classifier outputs.

The tail model was trained on Michel decays that also appear in the Michel evaluation
pieces. `overlap_masks` marks those overlapping trigger decays, and `cache` removes
them. Its source-group exclusion is the intended TAIL10 training pool, which is
conservative because TAIL10 loaded only its first five shards
([tail_model.md](tail_model.md)).

## Build the cache

Prepare a frozen, complete input directory with these relative paths; a production
directory still acquiring shards is not a fixed sample:

| Asset | Used by |
| --- | --- |
| `norad_eval2_drop/ev_*/predictions.parquet` | `cache` |
| `purity_eval/filmg2_ep19tg_pie/shard_*/pie_eval_events.parquet` | `cache` |
| `michel_eval_inwin_drop/*/predictions.parquet` | `cache`, `overlap_masks` |
| `purity_eval/filmg2_ep19_michel_inwin/*/mic_events.parquet` | `cache`, `overlap_masks` |
| `michel_inwin/*.parquet` | `overlap_masks` |
| `michel_inwin_leakmask/*.parquet` | `cache` (written by `overlap_masks`) |
| `tail_reveal_eval10/predictions_drop.parquet` | `cache` |
| `purity_eval/tail_osl/{pidif,mudif}_osl_events.parquet` | `cache` |
| `tail10/eval_mudif_osl.parquet` | `cache` |
| `parquet/michel/michel_1595422_t*_c*.parquet`, groups t000–t044 | `overlap_masks` |

The paths are those of the final analysis under
`/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/`; for new predictions,
reproduce the layout. Each signal shard and Michel piece must have its paired
predictions and overlap mask; missing pairs or failed alignment raise an error.

```bash
python -m purity.final_analysis.tail.overlap_masks \
  --pipeline-dir /path/to/inputs --output-dir /path/to/inputs/michel_inwin_leakmask
python -m purity.final_analysis.tail.cache \
  --pipeline-dir /path/to/inputs --output-dir outputs/tail_numeric
python -m purity.final_analysis.tail.cutflows \
  --cache-dir outputs/tail_numeric --output-dir outputs/tail_figures
python -m purity.final_analysis.tail.fig_cutflow_spectra \
  --cache-dir outputs/tail_numeric --output-dir outputs/talk-cut-panels
```

Run on the final-analysis inputs on 2026-10-28, `cache` gave signal survival 100%,
63.2%, 31.2% and 26.7% and Michel survival 100%, 3.5%, 0.0061% and none over the
four stages, equal to the archived four-panel figure. The cache took about one minute
on a compute node.

Historical tail/main joins inside `cache` are positional, resynchronized by float32
energy equality, because the archived predictions predate the stable event key. New
inference outputs carry `source_file` and `source_row` ([data contract](data_contract.md)).

## Contamination versus tail bias

```bash
python -m purity.final_analysis.tail.mudif_tradeoff \
  --pie-pred /path/to/pie5M_pie_full.parquet \
  --mudif-pred /path/to/tail_reveal_eval10/predictions.parquet \
  --output-dir outputs/contamination-vs-tail-bias
```

It scans seven signal working points and plots signed tail bias against muon-DIF
contamination, with the operating points labelled. It keeps the original 20:1
signal/muon-DIF normalization at acceptance and the 0.6% physical signal tail. Signal
energy is `deposited_energy + dead_E + Epure`; muon-DIF energy is
`deposited_energy + dead_E`.

## Bias versus angle

```bash
python -m purity.final_analysis.tail.bias_vs_theta \
  --pred /path/to/predictions.parquet --pie-eff 0.5 --output-dir outputs/tail-bias-vs-angle
```

It plots `efficiency(tail) / efficiency(peak)` against true polar angle for the
muon-DIF and pion-DIF vetoes, with a threshold per angle bin that keeps 50% of signal
by default. It needs `is_pie`, `acceptance`, `positron_theta`, `deposited_energy`,
`dead_E` and both DIF scores. Energy is deposited `live + dead`, split at 56 MeV and
capped at 75 MeV. A ratio of one is unbiased.

## Run records

Every command writes `<recipe>.run.json` listing actual reads, selected columns, row
counts, source hashes and output hashes. Large inputs are identified by path, size and
modification time; `--hash-inputs` also hashes their contents. Existing outputs are
rejected, so use a fresh output directory for each run.
