#!/usr/bin/env python3
"""TDSE-calibrated separated-pulse delay model, with an unfitted complex-amplitude check.

Retains the pump's neutral ground component and old 1s-ion continuum. Conditions:
pump over, old wave beyond the interaction cutoff, fixed post-probe propagation.
This is a reduced asymptotic model, not a full TDSE run at every delay.
"""
from pathlib import Path
import sys,json,argparse,numpy as np
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import vector_function
from fedvr import make_grid
from tdse1d import cutoff
parser=argparse.ArgumentParser();parser.add_argument('--out-root',default=str(Path(__file__).resolve().parents[1]/'results/followup'));args=parser.parse_args()
root=Path(args.out_root);(root/'figures').mkdir(parents=True,exist_ok=True)
required=['pump_only','probe_only_plus','pump_probe_plus','pump_probe_minus']
if any(not (root/n/'spectrum.npz').exists() for n in required):raise SystemExit('Required direct spectra are not yet complete')
data={n:np.load(root/n/'spectrum.npz') for n in required}
meta={n:json.loads((root/n/'run.json').read_text()) for n in required}
pump=data['pump_only'];control=data['probe_only_plus'];E=pump['energy'];labels=pump['labels'];T0=144.
assert np.array_equal(E,control['energy'])
save=np.load(root/'pump_only/checkpoint.npz');Eg=float(save['ground_energy']);cg=np.vdot(save['ground'],save['psi'])*np.exp(1j*Eg*T0)
radial=dict(meta['pump_only']['config']['radial']);radial['ecs_angle']=0;grid=make_grid(**radial)
H=grid.kinetic.toarray().real+np.diag(-2*cutoff(grid.r.real,*meta['pump_only']['config']['cutoff_radii'])/grid.r.real)
Eion=float(np.linalg.eigvalsh(H)[0]);avec,pulses=vector_function(meta['pump_only']['config'])
t=np.linspace(0,T0,14401);A=np.array([avec(s) for s in t]);alpha=simpson(A,x=t,axis=0);beta=simpson(np.sum(A*A,axis=1),x=t)
theta=pump['theta'];phi=pump['phi'];unit=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=1)
boost=np.exp(1j*(np.sqrt(2*E)[:,None]*(unit@alpha)[None,:]+beta/2))
z,wt=np.polynomial.legendre.leggauss(20);weights=np.repeat(wt,32)*2*np.pi/32
delays=np.linspace(T0,T0+64,129);records=[];spectra=[];entanglement=[];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
energy_weights=simpson(np.eye(len(E)),x=E,axis=1)
joint_weight=np.sqrt(energy_weights[:,None]*np.sqrt(2*E)[:,None]*weights[None,:])
for name,pol in [('pump_probe_plus','sigma+'),('pump_probe_minus','sigma-')]:
    tr=json.loads((root/name/'ionic_transfer.json').read_text())['transfers_from_1s'][0]
    U=np.array(tr['amplitude_real'])+1j*np.array(tr['amplitude_imag'])
    bcontrol=control['amplitudes'].copy()
    if pol=='sigma-':
        indices=[list(map(tuple,labels)).index((n,l,-m)) for n,l,m in labels]
        bcontrol=bcontrol[indices].reshape(len(labels),len(E),20,32)[:,:,:,(-np.arange(32))%32]
        bcontrol=bcontrol.reshape(len(labels),len(E),-1)*((-1.)**labels[:,2])[:,None,None]
    old=U[:,None,None]*pump['amplitudes'][0][None,:,:]
    new=cg*boost[None,:,:]*bcontrol
    predicted=old+new;direct=data[name]['amplitudes']
    intensity=abs(predicted)**2*np.sqrt(2*E)[None,:,None]
    directP=data[name]['pes'];normal=float(simpson(np.sum(directP.sum(axis=0)*weights,axis=1),x=E))
    error=float(simpson(np.sum(np.sum(abs(intensity-directP),axis=0)*weights,axis=1),x=E)/normal)
    amplitude_error=float(np.sqrt(simpson(np.sum(np.sum(abs(predicted-direct)**2,axis=0)*weights,axis=1)*np.sqrt(2*E),x=E)/normal))
    totals=[]
    angle=int(np.argmin(abs(theta-np.pi/4)+abs(phi-np.pi/4)))
    aa=old[0,:,angle];bb=new[0,:,angle];mean=(abs(aa)**2+abs(bb)**2)*np.sqrt(2*E)
    visibility=2*abs(aa*bb)/(abs(aa)**2+abs(bb)**2+1e-300)
    candidates=(E>.35)&(E<.59)&(mean>1e-7)
    chosen=np.where(candidates)[0][np.argmax(visibility[candidates])]
    line=[];metrics=[]
    for delay in delays:
        delta=delay-T0
        b=old*np.exp(-1j*Eion*delta)+new*np.exp(1j*(E-Eg)*delta)[None,:,None]
        p=abs(b)**2*np.sqrt(2*E)[None,:,None]
        totals.append(np.sum(p*weights[None,None,:],axis=2));line.append(float(p[0,chosen,angle]))
        weighted=(b*joint_weight[None,:,:]).reshape(len(labels),-1);gram=weighted@weighted.conj().T;gram/=np.trace(gram)
        lam=np.linalg.eigvalsh(gram).clip(0,1);lam/=sum(lam);nonzero=lam[lam>1e-14]
        metrics.append([-float(np.sum(nonzero*np.log2(nonzero))),float(((sum(np.sqrt(lam)))**2-1)/2)])
    totals=np.array(totals);spectra.append(totals);metrics=np.array(metrics);entanglement.append(metrics)
    row={'polarization':pol,'direct_reference':name,'joint_intensity_relative_L1':error,
         'unfitted_complex_amplitude_relative_L2':amplitude_error,
         'maximum_angle_integrated_delay_change_relative':float(np.max(abs(totals-totals[0]))/totals[0].max()),
         'display_energy':float(E[chosen]),'display_theta_degrees':float(theta[angle]*180/np.pi),
         'display_phi_degrees':float(phi[angle]*180/np.pi),
         'display_mean_density':float(mean[chosen]),'ideal_azimuthal_visibility':float(visibility[chosen])}
    row['conditional_entropy_bits_range']=[float(metrics[:,0].min()),float(metrics[:,0].max())]
    row['conditional_pure_negativity_range']=[float(metrics[:,1].min()),float(metrics[:,1].max())]
    # A good global error can conceal a poor weak-tail prediction. Check the
    # displayed overlap region separately before interpreting its fringes.
    tail=(E>=.44)&(E<=.48)
    tail_norm=float(simpson(np.sum(directP[0,tail]*weights,axis=1),x=E[tail]))
    row['weak_overlap_tail_check']={'energy_window':[.44,.48],
        'joint_intensity_relative_L1':float(simpson(np.sum(abs(intensity[0,tail]-directP[0,tail])*weights,axis=1),x=E[tail])/tail_norm),
        'unfitted_complex_amplitude_relative_L2':float(np.sqrt(simpson(np.sum(abs(predicted[0,tail]-direct[0,tail])**2*weights,axis=1)*np.sqrt(2*E[tail]),x=E[tail])/tail_norm)),
        'selected_point_direct_density':float(directP[0,chosen,angle]),'selected_point_predicted_density':float(intensity[0,chosen,angle])}
    records.append(row)
    plotdelay=np.linspace(0,64,1025);frequency=E[chosen]-Eg+Eion
    coherent_cross=2*np.sqrt(2*E[chosen])*np.real(np.conj(aa[chosen])*bb[chosen]*np.exp(1j*frequency*plotdelay))
    axes[0].plot(plotdelay,mean[chosen]+coherent_cross,label=pol)
    if pol=='sigma+' and (root/'pump_probe_plus_delay192/spectrum.npz').exists():
        heldout=np.load(root/'pump_probe_plus_delay192/spectrum.npz');delta=48.
        predicted_delayed=old*np.exp(-1j*Eion*delta)+new*np.exp(1j*(E-Eg)*delta)[None,:,None]
        heldout_norm=float(simpson(np.sum(heldout['pes'].sum(axis=0)*weights,axis=1),x=E))
        delayed_error=float(simpson(np.sum(np.sum(abs(abs(predicted_delayed)**2*np.sqrt(2*E)[None,:,None]-heldout['pes']),axis=0)*weights,axis=1),x=E)/heldout_norm)
        delayed_amplitude_error=float(np.sqrt(simpson(np.sum(np.sum(abs(predicted_delayed-heldout['amplitudes'])**2,axis=0)*weights,axis=1)*np.sqrt(2*E),x=E)/heldout_norm))
        row['held_out_delay192']={'joint_intensity_relative_L1':delayed_error,'unfitted_complex_amplitude_relative_L2':delayed_amplitude_error}
        heldout_tail_norm=float(simpson(np.sum(heldout['pes'][0,tail]*weights,axis=1),x=E[tail]))
        row['held_out_delay192']['weak_tail_intensity_relative_L1']=float(simpson(np.sum(abs(abs(predicted_delayed[0,tail])**2*np.sqrt(2*E[tail])[:,None]-heldout['pes'][0,tail])*weights,axis=1),x=E[tail])/heldout_tail_norm)
    for jitter in [0,.5,2.]:
        axes[1].plot(plotdelay,mean[chosen]+coherent_cross*np.exp(-.5*(frequency*jitter)**2),label=f'{pol}, jitter {jitter} a.u.')
axes[0].set(xlabel='Additional delay (a.u.)',ylabel='Selected angle-resolved density',title='Weak overlap tail: azimuth-resolved interference')
axes[1].set(xlabel='Additional delay (a.u.)',ylabel='Ensemble-averaged density',title='Laser-delay jitter during preparation')
for ax in axes:ax.legend(fontsize=7);ax.title.set_fontsize(11)
fig.suptitle(f'1s ionic channel: E={records[0]["display_energy"]:.3f} a.u., theta={records[0]["display_theta_degrees"]:.1f} deg, phi=45 deg; separated-pulse model',fontsize=10)
fig.savefig(root/'figures/asymptotic_delay_scan.png',dpi=180);plt.close(fig)
np.savez_compressed(root/'asymptotic_delay_scan.npz',delay=delays,energy=E,angle_integrated=spectra,labels=labels,entanglement=entanglement,metric_labels=['entropy_bits','pure_negativity'])
result={'model':'separated-pulse coherent assembly calibrated with independent pump/control TDSE and ionic transfer',
 'assumptions':'old band initially 1s; neglected other old ionic and neutral bound channels; old electron beyond interaction cutoff',
 'not_full_TDSE_at_each_delay':True,'pump_ground_amplitude_real':float(cg.real),'pump_ground_amplitude_imag':float(cg.imag),
 'pump_displacement':alpha.tolist(),'pump_A2_integral':float(beta),'comparisons':records}
(root/'asymptotic_delay_scan.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records,indent=2))
