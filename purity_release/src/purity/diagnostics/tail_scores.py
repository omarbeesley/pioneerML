"""Plot tail ROC, score, DIF, flatness and fusion diagnostics from saved predictions.

No checkpoint or model forward pass is needed. The logistic fusion cross-check
fits a small CPU PyTorch model, as in the original evaluation script. Diagnostic
operating points are distinct from the final reconstructed-energy tail cutflow.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ._runtime import prepare_output, record_run
from ._tail_score_functions import roc_auc
from ._tail_score_render import render

COLUMN_MAP = {
    'pie': 'pie_score', 'topo': 'topo_score', 'muon': 'muon_score',
    'pileup': 'pileup_score', 'muon_dif': 'muon_dif_score',
    'pion_dif': 'pion_dif_score', 'energy': 'positron_energy',
    'dep_energy': 'deposited_energy', 'ptime': 'positron_t',
    'theta': 'positron_theta',
}
for name in ['pie_logit', 'topo_logit', 'muon_logit', 'pileup_logit',
             'muon_dif_logit', 'pion_dif_logit', 'is_pie', 'muon_present',
             'pileup_present', 'is_mudif', 'is_pidif', 'muon_dif_present',
             'pion_dif_present', 'dead_E', 'atar_posE', 'acceptance',
             'pion_stop_x', 'pion_stop_y', 'pion_stop_z', 'muon_decay_ke', 'pion_decay_ke']:
    COLUMN_MAP[name] = name


def load_predictions(paths):
    frames = []
    columns = sorted(set(COLUMN_MAP.values()))
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(pf.schema_arrow.names))
        if missing:
            raise ValueError(f'{path}: missing prediction columns: {missing}')
        frames.append(pf.read(columns=columns).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    if not len(df) or not (df.is_pie == 1).any() or not (df.is_pie == 0).any():
        raise ValueError('Tail diagnostics require signal and non-signal events')
    return {key: df[column].to_numpy() for key, column in COLUMN_MAP.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-bins', type=int, default=12)
    parser.add_argument('--signal-efficiency', type=float, default=0.90,
                        help='Per-head signal efficiency for fusion diagnostics')
    parser.add_argument('--bias-signal-efficiency', type=float, default=0.50)
    parser.add_argument('--energy-split', type=float, default=56.0)
    parser.add_argument('--bias-energy-max', type=float, default=75.0,
                        help='Historical DIF-bias study cap; not the final tail cutflow cap')
    parser.add_argument('--no-dead-energy', action='store_true')
    args = parser.parse_args()
    if args.n_bins < 2 or not 0 < args.signal_efficiency < 1 or not 0 < args.bias_signal_efficiency < 1:
        parser.error('Use at least two bins and efficiencies strictly between 0 and 1')
    data = load_predictions(args.pred)
    output = prepare_output(args.output_dir)
    historical = SimpleNamespace(output_dir=str(output), n_bins=args.n_bins,
        leak_sig_eff=args.signal_efficiency, bias_pie_eff=args.bias_signal_efficiency,
        bias_e_split=args.energy_split, bias_e_max=args.bias_energy_max,
        bias_no_dead_E=args.no_dead_energy)
    render(data, historical)
    labels = data['is_pie'].astype(int)
    aucs = {key: roc_auc(data[key], labels)[2] for key in ['pie', 'topo']}
    for score, label in [('muon', 'muon_present'), ('pileup', 'pileup_present')]:
        aucs[score] = (roc_auc(data[score], data[label].astype(int))[2]
                       if len(np.unique(data[label])) == 2 else None)
    summary = {'events':len(labels),'signal_events':int((labels==1).sum()),
               'non_signal_events':int((labels==0).sum()), 'auc':aucs,
               'background_definition':'All rows with is_pie == 0; use signal/Michel-only inputs for a Michel-only comparison.',
               'scope':'Unweighted diagnostic curves and sample-derived operating points; not final thesis selection metrics.'}
    (output/'score_metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    record_run(output,args.pred,vars(args),'purity.diagnostics.tail_scores')


if __name__ == '__main__':
    main()
