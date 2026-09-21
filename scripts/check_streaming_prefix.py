#!/usr/bin/env python3
"""Compare actual committed states before continuing the accelerated calculation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
old=Path(a.reference);new=Path(a.candidate)
c1=json.loads((old/'input.json').read_text());c2=json.loads((new/'input.json').read_text())
c1.pop('storage',None);c2.pop('storage',None)
if c1!=c2:raise ValueError('physical/observation inputs differ, beyond storage execution settings')
x=np.load(old/'checkpoint.npz');y=np.load(new/'checkpoint.npz')
if int(x['step'])!=int(y['step']):raise ValueError('checkpoint times differ')
overlap=np.vdot(x['ground'],y['ground']);phase=overlap/abs(overlap)
u=np.load(old/'surface_online'/f"accumulator_{int(x['surface_generation'])}.npz")
v=np.load(new/'surface_online'/f"accumulator_{int(y['surface_generation'])}.npz")
if int(u['sample_index'])!=int(v['sample_index']):raise ValueError('surface checkpoint times differ')
result={'step':int(x['step']),'time_au':float(u['time']),'ground_overlap_abs':float(abs(overlap)),
        'initial_phase_real':float(phase.real),'initial_phase_imag':float(phase.imag),
        'wavefunction_relative_L2':float(np.linalg.norm(y['psi']-phase*x['psi'])/np.linalg.norm(x['psi'])),
        'partial_complex_spectrum_relative_L2':float(np.linalg.norm(v['amplitude']-phase*u['amplitude'])/np.linalg.norm(u['amplitude'])),
        'scope':'same actual committed TDSE time; serial LU/CPU accumulation versus CPU iterative parallel replay/GPU accumulation. Initial-state phase only, no spectral fit.'}
result['reference_run_metadata']=json.loads((old/'run.json').read_text())
result['candidate_prefix_run_metadata']=json.loads((new/'run.json').read_text())
result['passed']=result['wavefunction_relative_L2']<1e-7 and result['partial_complex_spectrum_relative_L2']<1e-7 and abs(result['ground_overlap_abs']-1)<1e-9
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(2)
