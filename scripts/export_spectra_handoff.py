#!/usr/bin/env python3
"""Export finished SI energy spectra and provenance, leaving restart data on scratch."""
import argparse,hashlib,json,shutil,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from surface_storage import atomic_json
from spectrum_report import update_index

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def export(source,dest,fallback_snapshot=None):
    source=source.resolve();dest=dest.resolve()
    if source==dest or source in dest.parents:raise ValueError('handoff must be separate from source runs')
    dest.mkdir(parents=True,exist_ok=True);snapshots=set();exports=[]
    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or not (folder/'STATUS.json').exists():continue
        status=json.loads((folder/'STATUS.json').read_text(encoding='utf-8'))
        if status['state']!='complete':continue
        meta=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        if not meta['complete']:raise ValueError('complete status without finished propagation')
        obs=json.loads((folder/'observables.json').read_text(encoding='utf-8'));target=dest/folder.name;target.mkdir(exist_ok=True)
        names=['input.json','input_template.json','config.json','run.json','STATUS.json','RUN.md','attempts.json','observables.json',
               'ionic_transfer.json','history.json','history.csv','resource_estimate.json','rendering_provenance.json',
               'surface_online/layout.json','surface_online/preparation.json','surface_online/complete.json']
        names+=list(v for k,v in obs['artifacts'].items() if k!='full_angular_amplitudes')
        for name in dict.fromkeys(names):
            file=folder/name
            if file.exists():
                if file.stat().st_size>50_000_000:raise ValueError('unexpected large handoff artifact: '+str(file))
                (target/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target/name)
        if (folder/'output_manifest.json').exists():shutil.copy2(folder/'output_manifest.json',target/'source_output_manifest.json')
        files=[{'path':str(f.relative_to(folder)),'bytes':f.stat().st_size} for f in sorted(folder.rglob('*')) if f.is_file()]
        primary=folder/obs['artifacts']['spectrum'];full=folder/'spectrum.npz'
        record={'source_run_directory':str(folder),'configuration_signature':meta['signature'],'files':files,
                'largest_source_file_bytes':max(f['bytes'] for f in files),
                'primary_spectrum_sha256':sha256(primary),'full_angular_spectrum_sha256':sha256(full) if full.exists() else None,
                'scope':'Lightweight completed-result handoff; no wavefunction, ionic endpoints, projected history or restart accumulators copied.'}
        atomic_json(target/'export.json',record)
        with (target/'RUN.md').open('a',encoding='utf-8') as stream:
            stream.write('\n## 轻量交接范围\n\n此目录保留可重画总谱与符合能谱的主 NPZ/CSV、图和完整输入。大数组、全角复振幅与恢复检查点仍在原计算目录：\n\n`'+str(folder)+'`\n\n原始文件清单、大小和数据摘要见 export.json；本目录不能直接恢复传播。\n')
        local=[{'path':str(f.relative_to(target)),'bytes':f.stat().st_size} for f in sorted(target.rglob('*')) if f.is_file() and f.name!='output_manifest.json']
        atomic_json(target/'output_manifest.json',{'scope':'Only exported files; full scratch inventory is in export.json and source_output_manifest.json',
                    'files':local,'largest_actual_file_bytes':max(f['bytes'] for f in local)})
        attempts=json.loads((folder/'attempts.json').read_text(encoding='utf-8')) if (folder/'attempts.json').exists() else []
        if meta.get('numerical_snapshot'):
            snapshots.add(meta['numerical_snapshot'][:16])
        elif not attempts and fallback_snapshot:
            snapshots.add(fallback_snapshot.name)
            atomic_json(target/'snapshot_provenance.json',{'snapshot':fallback_snapshot.name,'evidence':'Campaign calibration_snapshot.txt; early runner did not yet write attempts.json.'})
        for attempt in attempts:
            if attempt.get('numerical_snapshot'):snapshots.add(attempt['numerical_snapshot'][:16])
        exports.append({'run_id':folder.name,'source':str(folder),'source_largest_file_bytes':record['largest_source_file_bytes']})
    snapshot_dir=dest/'numerical_snapshots';snapshot_dir.mkdir(exist_ok=True)
    for key in sorted(snapshots):
        frozen=source.parent/'code'/key
        if not frozen.is_dir():raise FileNotFoundError('missing numerical snapshot '+str(frozen))
        # Portable source only. Original binary hashes remain in the manifest;
        # make all rebuilds the library on the receiving machine.
        with tarfile.open(snapshot_dir/(key+'.tar.gz'),'w:gz') as archive:
            for file in sorted(frozen.rglob('*')):
                if file.is_file() and (file.suffix in ('.py','.f90') or file.name in ('Makefile','numerical_snapshot.json')):
                    archive.add(file,arcname=str(Path(key)/file.relative_to(frozen)),recursive=False)
    atomic_json(dest/'export_manifest.json',{'source_root':str(source),'completed_cases':exports,'numerical_source_snapshots':sorted(snapshots),
                'pending_runs_omitted':True,'scope':'Completed physical results only; no production convergence certification.'})
    update_index(dest)
    print('Exported',len(exports),'complete spectra to',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--fallback-snapshot',type=Path,help='Recorded calibration code path for early runs without attempts.json');a=p.parse_args()
    export(Path(a.source_root),Path(a.destination),a.fallback_snapshot)
