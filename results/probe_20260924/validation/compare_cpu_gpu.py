#!/usr/bin/env python3
"""Same three-pulse input on CPU (uninterrupted) and GPU (interrupted at step 1500,
resumed): compare final wavefunction, complex SI amplitudes, spectra and ionic transfers.
Usage: compare_cpu_gpu.py CPU_CASE_DIR GPU_CASE_DIR"""
import json,sys
from pathlib import Path
import numpy as np
cpu,gpu=map(Path,sys.argv[1:3])
meta={k:json.loads((d/'run.json').read_text(encoding='utf-8')) for k,d in (('cpu',cpu),('gpu',gpu))}
assert meta['cpu']['signature']==meta['gpu']['signature'],'different inputs'
with np.load(cpu/'checkpoint.npz') as a,np.load(gpu/'checkpoint.npz') as b:
    assert int(a['step'])==int(b['step']);psi=float(np.linalg.norm(a['psi']-b['psi'])/np.linalg.norm(a['psi']))
with np.load(cpu/'spectrum.npz') as a,np.load(gpu/'spectrum.npz') as b:
    amp=float(np.linalg.norm(a['amplitudes']-b['amplitudes'])/np.linalg.norm(a['amplitudes']))
    spec=float(np.max(abs(a['angle_integrated']-b['angle_integrated']))/np.max(abs(a['angle_integrated'])))
ta,tb=[json.loads((d/'ionic_transfer.json').read_text(encoding='utf-8'))['transfers_from_1s'][0] for d in (cpu,gpu)]
za=np.array(ta['amplitude_real'])+1j*np.array(ta['amplitude_imag']);zb=np.array(tb['amplitude_real'])+1j*np.array(tb['amplitude_imag'])
report={'input_signature':meta['cpu']['signature'],'steps':meta['cpu']['nsteps'],
        'cpu':{'device':meta['cpu']['device'],'invocations':'one, uninterrupted','maximum_linear_residual':meta['cpu']['maximum_linear_residual_all_segments']},
        'gpu':{'device':meta['gpu']['device'],'invocations':'interrupted after 1500 steps, resumed','maximum_linear_residual':meta['gpu']['maximum_linear_residual_all_segments']},
        'final_wavefunction_relative_L2':psi,'complex_SI_amplitudes_relative_L2':amp,'angle_integrated_spectrum_max_relative':spec,
        'ionic_transfer_amplitudes_max_abs':float(np.max(abs(za-zb))),
        'scope':'Implementation consistency (backend, replay workers, checkpoint/resume); both use the same numerics and linear tolerance 1e-10.'}
(Path(__file__).with_name('cpu_vs_gpu.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
