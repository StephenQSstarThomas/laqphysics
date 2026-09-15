#!/usr/bin/env python3
"""Fail closed on missing minimum-handoff evidence; keep full convergence separate."""
import argparse,hashlib,json,re,sys,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from convergence import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-root',default='results/convergence_complete');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path(a.out_root);rows=[]
    def check(name,passed,details):rows.append({'check':name,'passed':bool(passed),'details':details})
    for filename,expected in [('tests_release.log',79),('tests_debug.log',79),('tests_cuda_final.log',6)]:
        text=(root/filename).read_text() if (root/filename).exists() else ''
        counts=re.findall(r'(\d+) passed',text);count=int(counts[-1]) if counts else 0
        check(filename,count>=expected and not re.search(r'\d+ failed|ERROR collecting',text),{'passed_tests':count,'required':expected})
    for filename,key,limit in [
        ('bipolar_end_to_end.json','relative_complex_amplitude_L2',1e-10),
        ('ionic_gpu_regression.json','relative_amplitude_L2',1e-7)]:
        d=json.loads((root/filename).read_text()) if (root/filename).exists() else {}
        check(filename,key in d and d[key]<limit,{'value':d.get(key),'limit':limit})
    def pure_gauge_pass(d):
        rows=d['comparisons'];errors=[x['wavefunction_L2_difference_without_phase_fit'] for x in rows]
        return (len(errors)>=3 and all(a>b for a,b in zip(errors,errors[1:])) and errors[-1]<1e-5
                and all(abs(r['norm']-1)<1e-9 and r['maximum_true_linear_residual']<=1.01e-12 for x in rows for r in x['runs']))
    required=[
        ('precision_regression.json',lambda d:d['amplitude_L2_relative_after_initial_sign_convention']<1e-7
         and max(d['normalized_shape_L1'])<1e-7 and max(abs(np.array(d['relative_yield_change'])))<1e-7),
        ('projection_physical_regression.json',lambda d:d['maximum_relative_amplitude_difference']<1e-11),
        ('pure_gauge_wavefunctions.json',pure_gauge_pass),
        ('gauge_smoke_comparison.json',lambda d:max(d['normalized_shape_L1'])<.02 and max(abs(np.array(d['relative_yield_change'])))<.02),
        ('bipolar_cuda_check.json',lambda d:d['bipolar_CUDA_vs_Fortran_relative_H_error']<1e-12 and d['mixed_CUDA_vs_CPU_relative_H_error']<1e-12)]
    for filename,predicate in required:
        path=root/filename;d=json.loads(path.read_text()) if path.exists() else {}
        try:passed=predicate(d)
        except (KeyError,ValueError,TypeError):passed=False
        check(filename,passed,d if d else 'missing')
    spectra={}
    for name in ['reference','inner14']:
        folder=root/name;path=folder/'run.json';meta=json.loads(path.read_text()) if path.exists() else {}
        valid=meta.get('complete',False) and (folder/'spectrum.npz').exists() and (folder/'channel_projection.json').exists()
        details={'complete_propagation':meta.get('complete',False),'completed_steps':meta.get('completed_steps'),
                 'spectrum_present':(folder/'spectrum.npz').exists()}
        if valid:
            config=json.loads((repo/'configs/convergence_complete'/(name+'.json')).read_text())
            signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
            segments=[meta]+[json.loads(p.read_text()) for p in folder.glob('run_segment*.json')]
            history=json.loads((folder/'history.json').read_text());data=np.load(folder/'spectrum.npz')
            projection=json.loads((folder/'channel_projection.json').read_text())
            residual=max(s.get('maximum_linear_residual_this_invocation',0.) for s in segments)
            details.update(maximum_linear_residual=residual,maximum_exchange_error=max(h['exchange_error'] for h in history),
                           ground_residual=meta['ground_residual'],final_inner_norm=history[-1]['real_region_norm'],
                           maximum_ionic_linear_residual=projection['ionic_maximum_true_linear_residual'])
            valid=all(s['signature']==signature for s in segments) and str(data['source_signature'])==signature
            valid=valid and projection['recipe']['configuration']==signature and projection['nframes']==meta['nsteps']//config.get('surface_stride',1)+1
            valid=valid and projection['ionic_maximum_true_linear_residual']<=1.01e-12
            valid=valid and residual<=1.01*config['linear_tolerance'] and meta['ground_residual']<1e-8
            valid=valid and all(0<=h['real_region_norm']<=1+1e-6 and h['exchange_error']<1e-7 for h in history)
            valid=valid and np.isfinite(data['amplitudes']).all() and np.isfinite(data['angle_integrated']).all()
            if valid:spectra[name]=data
        check(name+' full production path',valid,details)
    if len(spectra)==2:
        limits=json.loads((repo/'configs/convergence_complete/production_plan.json').read_text())['acceptance']
        comparison=compare(spectra['reference'],spectra['inner14'],limits)
        check('near-nuclear radial 12 -> 14',comparison['status']=='passed',comparison)
    else:check('near-nuclear radial 12 -> 14',False,'pending full spectra')
    protected=json.loads((repo/'migration.json').read_text())['sha256']
    integrity={name:hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest for name,digest in protected.items()}
    check('original input integrity',all(integrity.values()),integrity)
    result={'scope':'minimum software validation handoff, not full production convergence',
            'minimum_handoff_passed':all(x['passed'] for x in rows),'full_convergence_certified':False,'checks':rows,
            'remaining_plan':'configs/convergence_complete/production_plan.json',
            'statement':'No implementation error was found in the documented checks. This does not certify every parameter regime or full convergence.'}
    temporary=root/'handoff_acceptance.tmp.json';temporary.write_text(json.dumps(result,indent=2)+'\n');os.replace(temporary,root/'handoff_acceptance.json')
    for row in rows:print('PASS' if row['passed'] else 'PENDING/FAIL',row['check'])
    return 0 if result['minimum_handoff_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
