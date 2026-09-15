#!/usr/bin/env python3
"""Losslessly compress completed spectrum artifacts before version-control delivery."""
from pathlib import Path
import zipfile,os,json,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1]/'results/followup';rows=[]
for path in sorted(root.rglob('spectrum*.npz')):
    if '.tmp.' in path.name:continue
    try:
        with zipfile.ZipFile(path) as archive:
            if all(info.compress_type==zipfile.ZIP_DEFLATED for info in archive.infolist()):continue
        before=path.stat();data={k:v for k,v in np.load(path).items()}
    except (ValueError,OSError,zipfile.BadZipFile):continue  # a producer may still be writing
    if path.stat().st_mtime_ns!=before.st_mtime_ns:continue
    temporary=path.with_name(path.stem+'.pack.tmp.npz');np.savez_compressed(temporary,**data)
    with np.load(temporary) as check:
        for key,value in data.items():np.testing.assert_array_equal(check[key],value)
    hashes={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in data.items()}
    if path.stat().st_mtime_ns!=before.st_mtime_ns:temporary.unlink();continue
    os.replace(temporary,path)
    rows.append({'file':str(path.relative_to(root)),'bytes_before':before.st_size,'bytes_after':path.stat().st_size,'array_sha256':hashes})
log=root/'lossless_packing.json';previous=json.loads(log.read_text()) if log.exists() else []
log.write_text(json.dumps(previous+rows,indent=2)+'\n')
print('Losslessly packed spectra:',len(rows))
