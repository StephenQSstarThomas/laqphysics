#!/usr/bin/env python3
"""Verify original package members; new build and run outputs are allowed."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'bundle_manifest.json').read_text());errors=[]
for name,record in manifest['files'].items():
    file=root/name
    if not file.is_file():errors.append(name+': missing');continue
    if file.stat().st_size!=record['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:
        errors.append(name+': changed')
if errors:raise SystemExit('\n'.join(errors))
print(f"PASS: {len(manifest['files'])} package files; source commit {manifest['source_commit']}")
print('The package contains completed small-result evidence; full production convergence remains pending.')
