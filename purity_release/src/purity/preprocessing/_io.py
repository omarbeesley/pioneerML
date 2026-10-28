"""Output preservation and small provenance records for preprocessing commands."""

import hashlib
import json
import sys
from pathlib import Path


def ensure_output(output):
    path = Path(output)
    for candidate in (path, Path(str(path) + ".manifest.json")):
        if candidate.exists():
            raise FileExistsError(f"Output exists; choose a new path: {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size}


def record_run(output, inputs, source, settings=None):
    output = Path(output)
    manifest = Path(str(output) + ".manifest.json")
    result = {
        "command": sys.argv,
        "source": file_record(source),
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "inputs": [file_record(path) for path in inputs],
        "output": file_record(output),
        "settings": settings or {},
        "data_contents_hashed": False,
    }
    with manifest.open("x") as handle:
        json.dump(result, handle, indent=2, default=str)
        handle.write("\n")
    return manifest
