#!/usr/bin/env python3
"""Calibrate 3D ESSS source on a weak TDSE run; compare to a DISTINCT strong run.

This is a model-calibrated matrix element, not an independent exact scattering integral.
"""
from pathlib import Path
import sys,json,numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from pulses import Pulse
from esss import driven
from run3d import setup

root=Path(__file__).resolve().parents[1];out=root/'results/3d_calibration';out.mkdir(exist_ok=True)
short=root/'results/3d_dt_refined'
if not (short/'spectrum.npz').exists():short=root/'results/3d_absorber_refined'
meta=json.loads((short/'run.json').read_text());data=np.load(short/'spectrum.npz')
hist=json.loads((short/'history.json').read_text());pg=hist[-1]['ground_population']
p=Pulse(**meta['config']['pulses'][0]['pulse']);h,_=setup(meta['config'])
T=h.grid.kinetic.toarray();r=h.r.real
e1,u=eigh(T+np.diag(-2*h.cut/r));e2,v=eigh(T+np.diag(-2*h.cut/r+1/r**2))
d12=float(abs(np.dot(v[:,0]*r,u[:,0])/np.sqrt(3)))
dg=float(np.sqrt(-2*np.log(pg)/(np.pi*p.field**2*p.area_squared(p.duration))))
e=data['energy'];P=data['angle_integrated'][0];threshold=meta['ground_energy']-e1[0]+p.omega
t=np.linspace(0,p.duration,4001)
fourier=simpson(p.envelope(t)[None,:]*np.exp(1j*(e[:,None]-threshold)*t[None,:]),x=t,axis=1)
valid=abs(e-threshold)<.12
dge=np.sqrt(4*P[valid]/(p.field**2*abs(fourier[valid])**2))
result={'weak_calibration_run':short.name,'weak_ground_population':pg,'dg_flat_from_depletion':dg,
        'd12_from_same_radial_grid':d12,'ionic_energies':[float(e1[0]),float(e2[0])],
        'numerical_ground_energy':meta['ground_energy'],'weak_source_center':float(threshold),
        'energy':e[valid].tolist(),'dg_magnitude_from_weak_spectrum':dge.tolist(),
        'limitations':'weak-pulse calibration on a truncated 3D model; absolute physical He cross section not converged'}
np.savez(out/'dipole_source.npz',energy=e[valid],dg_magnitude=dge)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.plot(e[valid],dge,label='from weak TDSE spectrum')
ax.axhline(dg,color='k',ls='--',label='flat-continuum depletion fit');ax.set(xlabel='Energy (a.u.)',ylabel='Energy-normalized |Dg(E)|',title='3D source calibrated on the weak pulse');ax.legend()
fig.savefig(out/'source_dipole.png',dpi=180);plt.close(fig)
strong=root/'results/3d_resonant_small'
if (strong/'spectrum.npz').exists():
    smeta=json.loads((strong/'run.json').read_text());sd=np.load(strong/'spectrum.npz');E=sd['energy']
    ps=Pulse(**smeta['config']['pulses'][0]['pulse'])
    model=abs(driven(E,ps,dg=dg,d12=d12,eg=smeta['ground_energy'],e1=e1[0],e2=e2[0],dt=.08))**2
    actual=sd['angle_integrated'];norm=simpson(actual,x=E,axis=1);modelnorm=simpson(model,x=E,axis=1)
    result['strong_run']=strong.name
    result['strong_comparison']={'tdse_probabilities':norm.tolist(),'esss_probabilities':modelnorm.tolist(),
        'shape_L1':simpson(abs(actual/norm[:,None]-model/modelnorm[:,None]),x=E,axis=1).tolist(),
        'absolute_spectrum_L1_relative_error':(simpson(abs(actual-model),x=E,axis=1)/modelnorm).tolist()}
    np.savez(out/'strong_comparison.npz',energy=E,esss=model,tdse=actual)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for j,ax in enumerate(axes):
        ax.plot(E,actual[j],label='3D TDSE, small grid');ax.plot(E,model[j],'--',label='ESSS, weak-run calibration')
        ax.set(xlabel='Energy (a.u.)',ylabel='dP/dE',title=f'Ionic channel {j+1}');ax.legend(fontsize=8)
    fig.savefig(out/'strong_esss_vs_tdse.png',dpi=180);plt.close(fig)
(out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
