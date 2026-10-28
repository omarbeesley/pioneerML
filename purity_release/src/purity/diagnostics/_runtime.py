"""Small shared helpers for explicit, recorded diagnostic runs."""
import math
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import sys


def prepare_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_run(output, inputs, parameters, module):
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    inputs = sorted({Path(p).resolve() for p in inputs})
    def json_ready(value):
        if isinstance(value, dict):
            return {k:json_ready(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):
            return None
        return value
    data = {'completed_utc': datetime.now(timezone.utc).isoformat(),
            'module': module, 'parameters': parameters,
            'python': sys.version,
            'input_files': [{'path': str(p), 'bytes': p.stat().st_size,
                             'sha256': digest(p)} for p in inputs],
            'scope': 'Diagnostic output for the supplied sample, not a claim of archived thesis reproduction.'}
    (Path(output) / 'analysis_run.json').write_text(json.dumps(json_ready(data), indent=2, default=str, allow_nan=False) + '\n')
