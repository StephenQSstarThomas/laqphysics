#!/usr/bin/env python3
"""Resolve old and newly generated 0.75-a.u. control electrons by outgoing m."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import vector_function
parser=argparse.ArgumentParser();parser.add_argument('--out-root',default=str(Path(__file__).resolve().parents[1]/'results/followup'));args=parser.parse_args()
root=Path(args.out_root);(root/'figures').mkdir(parents=True,exist_ok=True);out=root/'pump_probe_half_frequency'
meta=json.loads((out/'run.json').read_text());data=np.load(out/'spectrum.npz');E=data['energy'];k=np.sqrt(2*E)
A,pulses=vector_function(meta['config']);T=max(p.start+p.duration for p,pol in pulses)
t=np.linspace(0,T,int(np.ceil(T/.02))+1);displacement=simpson(np.array([A(s) for s in t]),x=t,axis=0)
theta=data['theta'].reshape(20,32);phi=data['phi'].reshape(20,32)
unit=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
amp=data['amplitudes'][0].reshape(len(E),20,32)*np.exp(-1j*k[:,None,None]*(unit@displacement)[None,:,:])
modes=np.fft.fft(amp,axis=-1)/32;m=np.fft.fftfreq(32,d=1/32).astype(int);_,w=np.polynomial.legendre.leggauss(20)
resolved=2*np.pi*np.sum(abs(modes)**2*w[None,:,None],axis=1)*k[:,None]
rows=[];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].semilogy(E,np.maximum(data['angle_integrated'][0],1e-14),label='1s ionic channel')
bands=[('old pump band',[.18,.42],0),('new two-photon band',[.5,.75],2),('new three-photon band',[1.15,1.55],3)]
if max(E)>2.35:bands.append(('new four-photon band',[1.9,2.35],4))
for index,(name,window,expected) in enumerate(bands):
    select=(E>=window[0])&(E<=window[1]);prob=simpson(resolved[select],x=E[select],axis=0);total=float(sum(prob));fraction=prob/total
    peaks=find_peaks(data['angle_integrated'][0,select],prominence=.05*data['angle_integrated'][0,select].max())[0]
    row={'band':name,'energy_window':window,'recorded_1s_probability':total,'dominant_m':int(m[np.argmax(prob)]),
         'dominant_fraction':float(max(fraction)),'expected_m':expected,'m_values':m.tolist(),'m_fractions':fraction.tolist(),
         'local_peak_energies':E[select][peaks].tolist()};rows.append(row)
    axes[0].axvspan(*window,alpha=.08,color=f'C{index}')
    extent=4 if len(bands)==4 else 3;width=.8/len(bands)
    display=(m>=-extent)&(m<=extent);axes[1].bar(m[display]+(index-(len(bands)-1)/2)*width,fraction[display],width=width,label=name)
axes[0].set(xlabel='Electron energy (a.u.)',ylabel='dP / dE',ylim=(1e-10,10),title='0.75-a.u. control: old and new electrons')
axes[1].set(xlabel='Outgoing m after removing displacement phase',ylabel='Fraction in the selected band',title='New bands have the allowed photon angular momenta')
for ax in axes:ax.legend(fontsize=7);ax.title.set_fontsize(10)
fig.savefig(root/'figures/half_frequency_channels.png',dpi=180);plt.close(fig)
result={'bands':rows,'displacement_removed':displacement.tolist(),
        'scope':'Actual neutral-He pump/control TDSE, conditional on the 1s ionic channel. Does not imply a forbidden ionic 1s->2p two-photon resonance.'}
(root/'half_frequency_channels.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if not k.startswith('m_')} for r in rows],indent=2))
