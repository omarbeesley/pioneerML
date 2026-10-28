# Analysis workflow

Start from saved model outputs when you want to repeat plots or change cuts; inference
is not needed for every analysis run.

| Task | Code and instructions |
| --- | --- |
| ROOT conversion, source-pool assembly, mixing and reslicing | `purity.preprocessing`; [data preparation](data_preparation.md) |
| Main-model inference | `python -m purity --config configs/main_filmg2_ep19.json ...` |
| Tail-model inference | `python -m purity --config configs/tail10_ep10.json ...` |
| Reconstruction benchmark plots | `purity.diagnostics.benchmark_plots` |
| Reconstruction-versus-truth plots of the R_e/μ inputs (time, HTP, time spread, dead energy, spectra) | `purity.diagnostics.part1` |
| ROC, score, DIF, flatness and fusion diagnostics from tail scores | `purity.diagnostics.tail_scores` |
| Per-head cut tables and cut scans | `purity.diagnostics.per_head_tables`, `cut_sweep` |
| R_e/μ measurement and counting cross-checks | `purity.measurement.run_analysis`, `purity.diagnostics.counting`, `counting_bias` |
| Performance and acceptance-bias tables, time fit | [final analysis](reproduction.md) |
| Tail cut chain and cutflow plots | `purity.final_analysis.tail`; [tail analysis](tail_analysis.md) |
| Event displays from mixed inputs | `purity.diagnostics.event_display` |

Run these commands in the recorded environment with the package installed, or set
`PYTHONPATH="$PWD/src"` from the release root. The README shows the validated
Apptainer invocation; additional data directories must be bound into the container
explicitly. Every command provides `--help`.

## One-command small workflow

```bash
python scripts/run_example_analysis.py --output-dir outputs/example-analysis
```

This uses the bundled unmixed events and LUTs to mix 12 events per channel, reslices
them, runs both selected models, and produces the benchmark and tail-score plots. It
writes a log per step and `workflow.json`. In Apptainer, use
`python3 /release/scripts/run_example_analysis.py` with
`--output-dir /release/outputs/example-analysis`. Each run needs a new output
directory. These are integration checks, not physics results.

## From model outputs to diagnostic plots

Run both model examples from the README first, producing `outputs/main` and
`outputs/tail`. Then:

```bash
python -m purity.diagnostics.benchmark_plots \
  --results-dir outputs/main --output-dir outputs/benchmark-plots

python -m purity.diagnostics.tail_scores \
  --pred outputs/tail/predictions.parquet --output-dir outputs/tail-score-plots

python -m purity.diagnostics.per_head_tables \
  --pred outputs/tail/predictions.parquet

python -m purity.diagnostics.cut_sweep \
  --pred outputs/tail/predictions.parquet --output-dir outputs/cut-scan \
  --grid_points 3

python -m purity.diagnostics.event_display \
  --input examples/pie.parquet --event 0 --mode all \
  --num_events 2 --output outputs/example-events.pdf
```

The benchmark suite writes acceptance, matching and classification, pion-stop,
endpoint, direction, energy, time and hit-level plots. The tail suite reads saved
scores and labels, computes ROC and energy/time dependence, and fits a small
logistic-fusion cross-check on CPU. Missing populations are skipped explicitly. The
bundled examples have no DIF events and too few events to measure rejection or
spectrum bias; use held-out signal, Michel, muon-DIF and pion-DIF samples for that.

The cut scan uses four independent signal/Michel heads and rejects inputs containing
DIF events, so that a mixed background is not described as Michel. Its tuned cuts are
diagnostic working points, not the final analysis cuts. `per_head_tables` supports the
DIF heads and prints the acceptance, energy definition and cuts it uses.

`benchmark_plots` keeps its own diagnostic defaults, including a configurable Michel
display scale. It does not apply the generator-weight normalization of the final
analysis.

## Final analysis

The performance, acceptance-bias and time-fit commands take explicit event-table
inputs; [reproduction.md](reproduction.md) gives their recipes and the reference
outputs to compare against. `purity.final_analysis.combine_events` assembles combined
tables from a fixed list of shards. Start each analysis in a new output directory;
source hashes and validation records are under `provenance/`.
