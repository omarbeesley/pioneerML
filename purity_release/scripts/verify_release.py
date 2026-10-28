"""Verify a copied release using only the Python standard library.

Run from any directory: python /path/to/purity_release/scripts/verify_release.py
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'release_manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = root / entry['file']
        if not path.is_file():
            failures.append(f"missing: {entry['file']}")
            continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != entry['sha256']:
            failures.append(f"checksum mismatch: {entry['file']}")
    # Files the manifest does not list, outside the locations .gitignore excludes.
    listed = {entry['file'] for entry in manifest['files']} | {'release_manifest.json'}
    ignored_dirs = {'outputs', '__pycache__', 'build', 'dist', '.venv', '.pytest_cache'}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if (not path.is_file() or ignored_dirs & set(rel.parts) or path.suffix in {'.pyc', '.pyo', '.pyd'}
                or any(part.endswith('.egg-info') for part in rel.parts)):
            continue
        if rel.as_posix() not in listed:
            failures.append(f"not in manifest: {rel.as_posix()}")
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} release files, including checkpoints and examples.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
