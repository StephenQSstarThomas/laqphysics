#!/usr/bin/env python3
"""Rebuild the named figures/index from finished data; never rerun propagation."""
import argparse,hashlib,json,sys,fcntl
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from spectrum_report import publish,update_index
from surface_storage import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out-root',required=True);a=p.parse_args();folder=Path(a.out_root)
for out in sorted(folder.iterdir()):
    if not out.is_dir() or not (out/'run.json').exists():continue
    with (out/'.case.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('SKIPPED active writer',out.name,flush=True);continue
        meta=json.loads((out/'run.json').read_text(encoding='utf-8'))
        if not meta.get('complete'):continue
        old=json.loads((out/'observables.json').read_text(encoding='utf-8')) if (out/'observables.json').exists() else {}
        if not (out/'spectrum.npz').exists() and not (old.get('artifacts') and (out/old['artifacts']['spectrum']).exists()):continue
        result=publish(out,out.name)
        # Replace only known, generated presentation files from the first renderer.
        for file in [out/f'{out.name}__single_ionization_spectrum.npz',out/f'{out.name}__single_ionization_spectrum.csv',
                     out/'figures'/f'{out.name}__single_ionization_spectrum.png',out/'figures'/f'{out.name}__single_ionization_spectrum.pdf']:
            if file.exists():file.unlink()
        atomic_json(out/'STATUS.json',{'state':'complete','complete_spectrum':True,'run_id':out.name,'input':'input.json',
                    'spectrum':result['artifacts']['spectrum'],'figure':result['artifacts']['figure'],
                    'conditional_figure':result['artifacts']['conditional_figure'],'observables':'observables.json',
                    'design_targets_passed':result['target_passed'],'numerical_convergence_certified':False})
        atomic_json(out/'rendering_provenance.json',{'reporter_sha256':hashlib.sha256((root/'python/spectrum_report.py').read_bytes()).hexdigest(),
                    'configuration_signature':meta['signature'],'propagation_rerun':False})
        files=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file() and f.name!='output_manifest.json']
        largest=max(f['bytes'] for f in files);limit=meta.get('max_file_bytes',4_000_000_000)
        if largest>limit:raise RuntimeError('output exceeds configured per-file cap')
        atomic_json(out/'output_manifest.json',{'run_id':out.name,'files':files,'largest_actual_file_bytes':largest,'max_file_bytes':limit})
        print('RENDERED',out.name,'peak ratio',result['peak_height_ratio_03_over_06'],'prepared ion',result['prepared_ion_P_2p_plus1'],flush=True)
update_index(folder)
