#!/usr/bin/env python3
"""Summarize actual follow-up runs; no assumed peaks or fitted energy translations."""
from pathlib import Path
import sys,json,numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import vector_function
from pulses import Pulse
from fedvr import make_grid
from tdse1d import cutoff

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/followup';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def load_run(name):
    p=OUT/name
    if not (p/'spectrum.npz').exists():return None
    return json.loads((p/'run.json').read_text()),np.load(p/'spectrum.npz')

def width(x,y):
    k=np.argmax(y);half=y[k]/2
    left=np.where(y[:k]<=half)[0];right=np.where(y[k:]<=half)[0]
    if not len(left) or not len(right):return None
    l=left[-1];r=k+right[0]
    return float(np.interp(half,y[r-1:r+1][::-1],x[r-1:r+1][::-1])-np.interp(half,y[l:l+2],x[l:l+2]))

def compare(a,b,window=None):
    e=b['energy'];mask=np.ones(len(e),bool) if window is None else (e>=window[0])&(e<=window[1])
    e=e[mask];pa=np.array([np.interp(e,a['energy'],v) for v in a['angle_integrated']]);pb=b['angle_integrated'][:,mask]
    ia=simpson(pa,x=e,axis=1);ib=simpson(pb,x=e,axis=1)
    return {'relative_yield_change':(ib/ia-1).tolist(),
        'relative_absolute_L1':(simpson(abs(pb-pa),x=e,axis=1)/ia).tolist(),
        'normalized_shape_L1':simpson(abs(pb/ib[:,None]-pa/ia[:,None]),x=e,axis=1).tolist()}

def fitted_peaks(e,y):
    positions=[];heights=[]
    for i in find_peaks(y,prominence=.05*y.max())[0]:
        p=np.polyfit(e[i-1:i+2]-e[i],y[i-1:i+2],2);offset=-p[1]/(2*p[0]) if p[0]<0 else 0.
        if abs(offset)>e[i+1]-e[i]:offset=0.
        positions.append(float(e[i]+offset));heights.append(float(np.polyval(p,offset)))
    return positions,heights

def bandwidth():
    rows=[];fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    old=np.load(ROOT/'results/3d_dt_refined/spectrum.npz')
    new=load_run('weak_24')
    for cycles,data in [(8,old)]+([(24,new[1])] if new else []):
        e=data['energy'];p=data['angle_integrated'][0];p=p/p.max();pulse=Pulse(cycles=cycles,field=.005)
        u=brentq(lambda x:(np.sinc(x)+.5*np.sinc(x+1)+.5*np.sinc(x-1))**2-.5,.1,.99)
        theory=2*u*pulse.omega/cycles
        rows.append({'support_cycles':cycles,'intensity_FWHM_cycles':float(cycles*2*np.arccos(2**(-.25))/np.pi),
            'duration_as':pulse.duration*24.1888432659,'Fourier_FWHM_au':theory,'TDSE_FWHM_au':width(e,p)})
        ax.plot(e,p,label=f'{cycles} cycles: TDSE FWHM {width(e,p):.3f} a.u.')
        det=e-.6027683898;v=det*pulse.duration/(2*np.pi)
        transform=(np.sinc(v)+.5*np.sinc(v+1)+.5*np.sinc(v-1))**2
        ax.plot(e,transform,'--',alpha=.5,label=f'{cycles} cycles: pulse Fourier limit')
    ax.set(xlabel='Electron energy (a.u.)',ylabel='Peak-normalized density',title='8 full cycles = 2.91 intensity-FWHM cycles; bandwidth explains the broad peak')
    ax.legend(fontsize=8);fig.savefig(FIG/'bandwidth_explained.png',dpi=180);plt.close(fig)
    return rows

def circular():
    runs=[load_run('circular_plus'),load_run('circular_minus')]
    if any(r is None for r in runs):return None
    a=runs[0][1];b=runs[1][1];nt=20;nf=32
    pa=a['pes'].reshape(4,len(a['energy']),nt,nf);pb=b['pes'].reshape(4,len(b['energy']),nt,nf)
    index=[]
    for n,l,m in a['labels']:index.append(list(map(tuple,b['labels'])).index((n,l,-m)))
    reflected=pb[index][:,:,:,(-np.arange(nf))%nf]
    error=float(np.max(abs(pa-reflected))/pa.max());moments=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,(meta,data),color in zip(['sigma+','sigma-'],runs,['C0','C1']):
        e=data['energy'];k=np.sqrt(2*e);amp=data['amplitudes'][0].reshape(len(e),nt,nf)
        theta=data['theta'].reshape(nt,nf);phi=data['phi'].reshape(nt,nf)
        vfun,pulses=vector_function(meta['config']);T=max(p.start+p.duration for p,pol in pulses)
        t=np.linspace(0,T,int(np.ceil(T/.02))+1);intA=simpson(np.array([vfun(x) for x in t]),x=t,axis=0)
        direction=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
        # Convert the stored Volkov coefficient to plane-wave momentum phase.
        amp=amp*np.exp(-1j*k[:,None,None]*np.einsum('tpa,a->tp',direction,intA)[None,:,:])
        modes=np.fft.fft(amp,axis=2)/nf;m=np.fft.fftfreq(nf,d=1/nf).astype(int)
        z,w=np.polynomial.legendre.leggauss(nt)
        weight=simpson(2*np.pi*np.sum(abs(modes)**2*w[None,:,None],axis=1)*k[:,None],x=e,axis=0);weight/=sum(weight)
        moments.append({'polarization':name,'m_values':m.tolist(),'normalized_weights':weight.tolist(),
                        'dominant_m':int(m[weight.argmax()]),'dominant_fraction':float(weight.max())})
        axes[0].plot(e,data['angle_integrated'][0],color=color,label=name)
        idx=data['angle_integrated'][0].argmax();pad=data['pes'][0,idx].reshape(nt,nf).mean(axis=1)/data['angle_integrated'][0,idx]
        axes[1].plot(theta[:,0]*180/np.pi,pad,'o',ms=3,color=color,label=name)
    th=np.linspace(0,np.pi,181);axes[1].plot(th*180/np.pi,3/(8*np.pi)*np.sin(th)**2,'k--',label='one-photon |Y1,+/-1|^2')
    axes[0].set(xlabel='Energy (a.u.)',ylabel='dP/dE',title='Opposite helicities: same marginal spectrum')
    axes[1].set(xlabel='Polar angle (degrees)',ylabel='Normalized angular density',title='Outgoing m = +/-1; sin^2 angular intensity')
    for ax in axes:ax.title.set_fontsize(11)
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(FIG/'circular_verification.png',dpi=180);plt.close(fig)
    return {'mirror_max_relative_error':error,'azimuthal_modes':moments}

def convergence():
    names=['res_baseline_cf4','res_l6_L2_p4','res_l6_L2_p6','res_l8_L2_p6'];rows=[];previous=None
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for name in names:
        run=load_run(name)
        if run is None:continue
        meta,a=run;e=a['energy'];P=a['angle_integrated'];peaks,heights=fitted_peaks(e,P.sum(axis=0))
        radial=dict(meta['config']['radial']);radial['ecs_angle']=0.;grid=make_grid(**radial)
        ionic_H=grid.kinetic.toarray().real+np.diag(-2*cutoff(grid.r.real,*meta['config']['cutoff_radii'])/grid.r.real)
        ionic_ground=float(np.linalg.eigvalsh(ionic_H)[0])
        r={'run':name,'ground_energy':meta['ground_energy'],'ground_residual':meta['ground_residual'],
            'lmax':meta['config']['lmax'],'total_Lmax':meta['config'].get('total_Lmax'),
            'radial_order':meta['config']['radial']['order'],'channels':meta['channels'],'nrad':meta['nrad'],
            'ionic_ground_energy':ionic_ground,
            'total_peak_energies':peaks,'total_peak_heights':heights,'linear_residual':meta.get('maximum_linear_residual_this_invocation')}
        segments=[meta]+[json.loads(path.read_text()) for path in (OUT/name).glob('run_segment*.json')]
        if any(s.get('signature')!=meta.get('signature') for s in segments):raise ValueError('inconsistent checkpoint segment configuration')
        r['linear_residual']=max(s.get('maximum_linear_residual_this_invocation',0.) for s in segments)
        r['linear_residual_scope']='maximum over recorded invocation segments'
        if previous:
            r['previous_run']=previous[0];r['change']=compare(previous[1],a)
            r['comparison_scope']='combined baseline upgrade' if previous[0]=='res_baseline_cf4' else 'one refinement axis with other settings fixed'
            shape=max(r['change']['normalized_shape_L1']);yield_change=max(abs(np.array(r['change']['relative_yield_change'])))
            r['shape_and_yield_2percent_pass']=bool(shape<.02 and yield_change<.02)
            old_peaks=previous[3];shift=float(max(abs(np.array(peaks)-old_peaks))) if len(peaks)==len(old_peaks) and len(peaks) else None
            r['maximum_peak_shift']=shift;r['all_three_gates_pass']=bool(r['shape_and_yield_2percent_pass'] and shift is not None and shift<=.001)
            # Diagnostic only: use independently calculated thresholds, never a fitted shift.
            old_meta=previous[2];delta=(meta['ground_energy']-ionic_ground)-(old_meta['ground_energy']-previous[4])
            shifted=np.array([np.interp(e-delta,previous[1]['energy'],v) for v in previous[1]['angle_integrated']])
            aint=simpson(shifted,x=e,axis=1);bint=simpson(P,x=e,axis=1)
            r['threshold_centered_shape_L1_diagnostic_only']=simpson(abs(shifted/aint[:,None]-P/bint[:,None]),x=e,axis=1).tolist()
        previous=(name,a,meta,np.array(peaks),ionic_ground);rows.append(r);ax.plot(e,P.sum(axis=0),label=name)
    ax.set(xlabel='Energy (a.u.)',ylabel='Total dP/dE',title='Convergence in absolute energy; no artificial peak alignment');ax.legend(fontsize=7)
    fig.savefig(FIG/'convergence.png',dpi=180);plt.close(fig)
    return rows

def pump_probe():
    pump=load_run('pump_only')
    if pump is None:return []
    m0,p=pump;e=p['energy'];mask=(e>=.18)&(e<=.42);old=simpson(p['angle_integrated'][0,mask],x=e[mask]);rows=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(e,p['angle_integrated'].sum(axis=0),'k--',label='pump only')
    axes[1].plot(e[mask],p['angle_integrated'][0,mask]/old,'k--',label='pump only, 1s')
    for name in ['pump_probe_plus','pump_probe_minus','pump_probe_linear','pump_probe_half_frequency']:
        run=load_run(name)
        if run is None:continue
        meta,a=run;E=a['energy'];P=a['angle_integrated'];ind=(E>=.18)&(E<=.42)
        # The half-frequency calculation uses lmax=3. Reuse its exact pre-control
        # surface history as the matched pump reference, avoiding an lmax shift.
        reference=p;reference_name='pump_only'
        prefix=OUT/name/'spectrum_pump_prefix.npz'
        if name=='pump_probe_half_frequency':
            if not prefix.exists():
                rows.append({'run':name,'status':'awaiting matched lmax=3 pump-prefix extraction'});continue
            reference=np.load(prefix);reference_name=name+'/spectrum_pump_prefix.npz'
        er=reference['energy'];ir=(er>=.18)&(er<=.42)
        old=simpson(reference['angle_integrated'][0,ir],x=er[ir])
        prob=simpson(P[:,ind],x=E[ind],axis=1);summed=P[:,ind].sum(axis=0);norm=sum(prob)
        ref=np.interp(E[ind],er,reference['angle_integrated'][0])/old
        r={'run':name,'pump_reference':reference_name,'old_window':[.18,.42],'labels':a['labels'].tolist(),'old_channel_probabilities':prob.tolist(),
           'old_window_total_relative_to_pump':float(norm/old),
           'normalized_old_spectrum_L1':float(simpson(abs(summed/norm-ref),x=E[ind]))}
        f=OUT/name/'ionic_transfer.json'
        if f.exists():
            transfer=json.loads(f.read_text())['transfers_from_1s'][0];prediction=old*np.array(transfer['probabilities'])
            r['ionic_map_predicted_old_probabilities']=prediction.tolist()
            r['ionic_map_probability_L1_relative']=float(sum(abs(prob-prediction))/old)
        rows.append(r);axes[0].plot(E,P.sum(axis=0),label=name);axes[1].plot(E[ind],summed/norm,label=name)
    axes[0].set(xlabel='Energy (a.u.)',ylabel='dP/dE',title='Separated pump (1.2) and control (1.5 / 0.75)')
    axes[1].set(xlabel='Old-electron energy (a.u.)',ylabel='Normalized old-band spectrum',title='Old band: test local ionic transfer, not an imposed doublet')
    for ax in axes:ax.legend(fontsize=7)
    fig.savefig(FIG/'pump_probe.png',dpi=180);plt.close(fig)
    return rows

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out-root',default=str(OUT))
    parser.add_argument('--sections',nargs='+',choices=['bandwidth','circular','convergence','pump-probe'],default=['bandwidth','circular','convergence','pump-probe']);args=parser.parse_args()
    OUT=Path(args.out_root);FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
    result={section.replace('-','_'):globals()[section.replace('-','_')]() for section in args.sections}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
