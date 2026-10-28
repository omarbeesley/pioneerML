"""Shared command and provenance handling; numerical recipes live in each module."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class AnalysisRun:
    """Record actual file reads, selected columns and output hashes for one recipe."""

    def __init__(self, name, output_dir, settings, hash_inputs=False):
        self.name = name
        self.output_dir = output_dir
        self.path = output_dir / f'{name}.run.json'
        if self.path.exists():
            raise FileExistsError(f'Existing run record: {self.path}; use a new output directory')
        stems = {
            'cache': ['tail_reveal_cache.npz'],
            'cutflows': ['tail_reveal_michel', 'tail_reveal_pidif', 'tail_reveal_mudif'],
            'fig_cutflow_spectra': ['cutflow_spectra_michel', 'cutflow_spectra_pidif',
                                    'cutflow_spectra_mudif'],
        }
        protected = [output_dir / (stem if '.' in stem else stem + ext)
                     for stem in stems.get(name, []) for ext in ('.pdf', '.png')]
        if name == 'overlap_masks':
            protected += list(output_dir.glob('*.parquet'))
        existing = [str(p) for p in protected if p.exists()]
        if existing:
            raise FileExistsError('Existing recipe output; use a new output directory: ' + ', '.join(existing))
        self.hash_inputs = hash_inputs
        self.data = {'schema_version': 1, 'recipe': name, 'status': 'running',
                     'started_utc': datetime.now(timezone.utc).isoformat(),
                     'settings': settings, 'python': sys.version, 'inputs': [],
                     'source_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                     'hash_input_policy': 'sha256' if hash_inputs else 'path, byte size, mtime; small cache inputs also sha256',
                     'outputs': []}
        self.before = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                       for p in output_dir.iterdir() if p.is_file()}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2) + '\n')

    def note(self, path, operation, columns=None, rows=None, small=False):
        path = Path(path).resolve()
        stat = path.stat()
        entry = {'path': str(path), 'operation': operation,
                 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if columns is not None:
            entry['columns'] = list(columns)
        if rows is not None:
            entry['rows_read'] = int(rows)
        if self.hash_inputs or small:
            entry['sha256'] = sha256(path)
        self.data['inputs'].append(entry)

    def read_parquet(self, path, *args, **kwargs):
        import pandas as pd
        data = pd.read_parquet(path, *args, **kwargs)
        self.note(path, 'pandas.read_parquet', kwargs.get('columns'), len(data))
        return data

    def read_table(self, path, *args, **kwargs):
        import pyarrow.parquet as pq
        data = pq.read_table(path, *args, **kwargs)
        self.note(path, 'pyarrow.parquet.read_table', kwargs.get('columns'), data.num_rows)
        return data

    def read_json(self, path):
        with Path(path).open() as stream:
            data = json.load(stream)
        self.note(path, 'json.load', small=True)
        return data

    def load_npz(self, path):
        import numpy as np
        data = np.load(path, allow_pickle=False)
        self.note(path, 'numpy.load', small=True)
        return data

    def glob(self, pattern):
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f'No inputs match required pattern: {pattern}')
        return matches

    def finish(self, error=None):
        self.data['status'] = 'failed' if error is not None else 'complete'
        if error is not None:
            self.data['error'] = f'{type(error).__name__}: {error}'
        for path in sorted(self.output_dir.iterdir()):
            if not path.is_file() or path == self.path:
                continue
            stat = path.stat()
            if self.before.get(path) != (stat.st_size, stat.st_mtime_ns):
                self.data['outputs'].append({'file': path.name, 'bytes': stat.st_size,
                                             'sha256': sha256(path)})
        if error is None and not self.data['outputs']:
            self.data['status'] = 'failed'
            self.data['error'] = 'Recipe returned without producing any output files'
        self.data['finished_utc'] = datetime.now(timezone.utc).isoformat()
        self.save()
        if error is None and self.data['status'] != 'complete':
            raise RuntimeError(self.data['error'])


def command(name, kind, run, description, argv=None, diagnostics=False):
    parser = argparse.ArgumentParser(description=description)
    if kind == 'either':
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--pipeline-dir', type=Path,
                           help='Prepared input directory with the historical pipeline-relative layout')
        group.add_argument('--cache-dir', type=Path,
                           help='Directory containing angular_predictions.parquet')
    elif kind == 'pipeline':
        parser.add_argument('--pipeline-dir', type=Path, required=True,
                            help='Prepared input directory with the historical pipeline-relative layout')
    else:
        parser.add_argument('--cache-dir', type=Path, required=True,
                            help='Directory containing tail_reveal_cache.npz and/or tail_stats.json')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory; different recipes may share it to merge tail_stats.json')
    parser.add_argument('--hash-inputs', action='store_true',
                        help='Also SHA256 every parquet read (adds a full read of large input files)')
    if diagnostics:
        parser.add_argument('--skip-cutflows', action='store_true',
                            help='Skip the historical deposited-energy Michel/pion-DIF cutflows')
        parser.add_argument('--michel-nmax', type=int, default=107,
                            help='Historical maximum Michel pieces; 107 matches tail_final_chain.sbatch')
    args = parser.parse_args(argv)
    for label in ('pipeline_dir', 'cache_dir'):
        value = getattr(args, label, None)
        if value is not None and not value.is_dir():
            parser.error(f'Input directory does not exist: {value}')
    if diagnostics and args.michel_nmax <= 0:
        parser.error('--michel-nmax must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = getattr(args, 'pipeline_dir', None)
    cache = getattr(args, 'cache_dir', None)
    pipeline = pipeline.resolve() if pipeline is not None else None
    cache = cache.resolve() if cache is not None else None
    settings = vars(args).copy()
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in settings.items()}
    record = AnalysisRun(name, output, settings, args.hash_inputs)
    try:
        run(pipeline_dir=pipeline, cache_dir=cache, output_dir=output, record=record,
            skip_cutflows=getattr(args, 'skip_cutflows', False),
            michel_nmax=getattr(args, 'michel_nmax', 107))
    except BaseException as exc:
        record.finish(error=exc)
        raise
    else:
        record.finish()
    return record.data
