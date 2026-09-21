#!/usr/bin/env python3
"""Audit first; optionally remove only completed-run raw flux, retaining results.

Without --apply this is read-only. A final-spectrum-only choice is explicit
because it gives up new ionic projections and quadrature re-integration.
"""
import argparse,fcntl,json,os,sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from surface_storage import atomic_json,load_flux,ShardedArray
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--apply',action='store_true')
p.add_argument('--final-spectrum-only',action='store_true',help='accept loss of re-projection/re-integration when no projected cache exists')
a=p.parse_args();out=Path(a.out).resolve()
with ExitStack() as stack:
    for name in ['.case.lock','.propagation.lock','.projection.lock','.extraction.lock']:
        lock=stack.enter_context((out/name).open('a'))
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('an active writer owns this case; raw data must remain')
    meta=json.loads((out/'run.json').read_text())
    if not meta.get('complete'):raise ValueError('incomplete run: raw history is needed by its legacy restart path')
    with np.load(out/'spectrum.npz') as d:
        if str(d['source_signature'])!=meta['signature'] or not np.isfinite(d['angle_integrated']).all():raise ValueError('missing or invalid final spectrum')
    cache=False;record=out/'channel_projection.json'
    if record.exists():
        m=json.loads(record.read_text())
        cache=(m['recipe']['configuration']==meta['signature'] and ((out/m['cache_path']/'array.json').exists() if m.get('cache_path') else (out/'projected_channels.npy').exists()))
    if (out/'surface_online/complete.json').exists() and (out/'surface_online/projected/array.json').exists():cache=True
    files=[]
    if (out/'flux.npy').exists():files.append(out/'flux.npy')
    if (out/'flux_shards').exists():files.extend(sorted((out/'flux_shards').glob('part_*.npy')))
    if files:
        raw=load_flux(out);expected=(meta['nsteps']//meta['config'].get('surface_stride',1)+1,meta['nrad'],len(meta['surface_indices']),meta['channels'])
        if raw.shape!=expected or raw.dtype!=np.dtype(np.complex128):raise ValueError('raw array does not match this simulation')
        if (out/'flux.npy').exists() and (out/'flux.npy').resolve().name!='flux.npy':raise ValueError('unexpected physical target name')
    links=[];targets={p.resolve() for p in files}
    # Known project aliases must not silently lose a shared history.
    for base in [root/'results',out.parent]:
        for candidate in base.rglob('flux.npy'):
            if candidate.resolve() in targets and candidate.resolve()!=candidate and candidate.absolute() not in [p.absolute() for p in files]:links.append(str(candidate))
    report={'case':str(out),'complete_spectrum_validated':True,'projected_cache_present':cache,
            'raw_files':[{'path':str(f),'physical_target':str(f.resolve()),'logical_bytes':f.stat().st_size,'allocated_bytes':f.stat().st_blocks*512} for f in files],
            'other_known_aliases':sorted(set(links)),'would_free_allocated_bytes':sum(f.stat().st_blocks*512 for f in files),
            'retained':'spectra, figures, wave checkpoint, and any projected cache',
            'lost':'new ionic final-channel/gauge projections from this raw history; legacy continuation using the old flux stream'}
    print(json.dumps(report,indent=2))
    if a.apply and files:
        if links:raise ValueError('other known case aliases reference these raw files; resolve them before deletion')
        if not cache and not a.final_spectrum_only:raise ValueError('no projected cache: use --final-spectrum-only only if saved final spectra are sufficient')
        atomic_json(out/'raw_cleanup_manifest.json',report)
        for file in files:
            actual=file.resolve()
            if file.is_symlink():file.unlink()
            actual.unlink()
        if (out/'flux_shards/array.json').exists():
            (out/'flux_shards/array.json').unlink();(out/'flux_shards').rmdir()
        print('Removed only the listed raw flux files. Final spectra and other data retained.')
