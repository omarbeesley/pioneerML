"""Run a small preparation -> inference -> analysis demonstration.

Use the production Python environment. This creates new output files only;
fixture-sized results demonstrate plumbing, not detector performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--events', type=int, default=12,
                        help='Small number per channel (1..32), before hit filtering')
    args = parser.parse_args()
    if not 1 <= args.events <= 32:
        parser.error('--events must be between 1 and 32 for these fixtures')
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output directory must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root / 'src')
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MKL_NUM_THREADS', '2')
    steps = []
    for channel in ['pie','michel']:
        mix = output/'preparation'/channel
        steps.append((f'mix_{channel}', 'purity.preprocessing.generate_tail_reveal', [
            '--only',f'{channel}_eval','--num_eval',str(args.events),'--chunk_size','6',
            '--data_dir',str(root/'examples/unmixed'),'--lut_dir',str(root/'assets/luts'),
            '--out_dir',str(mix),'--seed','4101' if channel=='pie' else '9001',
            '--pileup_lam','0.24'] + (['--no_replace'] if channel=='michel' else [])))
        steps.append((f'reslice_{channel}', 'purity.preprocessing.origin_reslice', [
            '--input',str(mix/f'tail_reveal_{channel}_eval.parquet'),
            '--output',str(output/'preparation'/f'{channel}_resliced.parquet')] ))
    inputs = [str(output/'preparation'/f'{c}_resliced.parquet') for c in ['pie','michel']]
    for family,config in [('main','main_filmg2_ep19.json'),('tail','tail10_ep10.json')]:
        steps.append((f'infer_{family}','purity',[
            '--config',str(root/'configs'/config),'--data',*inputs,
            '--output',str(output/family)]))
    steps += [
        ('benchmark_plots','purity.diagnostics.benchmark_plots',[
            '--results-dir',str(output/'main'),'--output-dir',str(output/'benchmark_plots')]),
        ('tail_score_plots','purity.diagnostics.tail_scores',[
            '--pred',str(output/'tail/predictions.parquet'),'--output-dir',str(output/'tail_score_plots')]),
    ]
    record = {'status':'running','scope':'Small end-to-end pipeline demonstration, not performance reproduction.',
              'started_utc':datetime.now(timezone.utc).isoformat(),'events_requested_per_channel':args.events,'steps':[]}
    report = output/'workflow.json'
    try:
        for name,module,arguments in steps:
            command = [sys.executable,'-m',module,*arguments]
            print(f'[{name}]',flush=True)
            entry={'name':name,'command':command,'status':'running'}
            record['steps'].append(entry)
            report.write_text(json.dumps(record,indent=2)+'\n')
            with (output/f'{name}.log').open('w') as log:
                run=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            entry['returncode']=run.returncode
            entry['status']='complete' if run.returncode==0 else 'failed'
            if run.returncode:
                raise RuntimeError(f'{name} failed; see {output/name}.log')
        record['status']='complete'
        print(f'Completed example workflow: {output}',flush=True)
    except BaseException as error:
        record['status']='failed';record['error']=str(error)
        raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat()
        report.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
