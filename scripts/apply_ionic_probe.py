#!/usr/bin/env python3
"""Fast three-pulse SI spectrum from a finished two-pulse run plus a He+ probe.

For a probe that starts after the two-pulse run ended (T2 <= probe start), the
tSURFF channel identity gives, for every electron momentum,
    b_c(k,T3) = sum_c' <phi_c|U_ion(T3,T2)|phi_c'> b_c'(k,T2),
with the same ionic Hamiltonian, grid, cutoff and vector potential as the full
two-electron run. Only one-electron propagation is needed, so long probes cost
minutes instead of days. It omits electrons crossing the surface after T2 and
bound ionic components outside the source channels; validate it against a full
three-pulse TDSE of the same model (scripts/compare_probe_spectra.py).

Output is a normal named case directory (run.json/spectrum.npz/figures/RUN.md)
whose run.json records method='ionic_probe_factorization' and the source run.
The map is built in the velocity gauge. Both endpoints have A=E=0, where the
field-free channel amplitudes are gauge invariant, so mixed-gauge sources are
accepted; the map is then not the literally identical discrete operator.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from pulses import Pulse
from surface_storage import atomic_json,capped_npz
from spectrum_report import publish,update_index,parameter_tag
from output_lock import exclusive_output
from run_defaults import effective_config

NUMERICAL_KEYS=('radial','lmax','M','cutoff_radii','surface','dt','surface_stride','total_Lmax','angular_representation',
                'natural_parity','time_integrator','ionic_propagator','linear_tolerance','ground_tolerance','gauge')

def end_time(config):
    pulses=[Pulse(**p['pulse']) for p in config['pulses']]
    return float(config['end_time']) if 'end_time' in config else max(p.start+p.duration for p in pulses)+config.get('post_time',60.)

def pulse_key(entry):return (Pulse(**entry['pulse']),entry.get('polarization','z'))

def spectral_grid(config):
    """Energy/angle quadrature of the amplitudes; plot_channels is presentation only."""
    return {k:v for k,v in config.get('spectrum',{}).items() if k!='plot_channels'},config.get('spectrum_energy')

def check_compatible(source,target):
    """The probe input must extend the source run without changing its physics or numerics.

    Both sides are compared after the defaults simulate_spectrum.py applies, and
    pulses as physical Pulse objects (explicit start=0 equals an omitted start)."""
    source=effective_config(source);target=effective_config(target);n=len(source['pulses'])
    if len(target['pulses'])<=n or [pulse_key(p) for p in target['pulses'][:n]]!=[pulse_key(p) for p in source['pulses']]:
        raise ValueError('probe input must start with exactly the source pulses')
    if not all(p.get('role')=='probe' for p in target['pulses'][n:]):
        raise ValueError('pulses after the source pulses must all have role=probe')
    changed=[k for k in NUMERICAL_KEYS if source.get(k)!=target.get(k)]
    if spectral_grid(source)!=spectral_grid(target):changed.append('spectrum/spectrum_energy (amplitude grid)')
    if changed:raise ValueError(f'numerical parameters differ from the source run: {changed}')
    t2=end_time(source);start=min(Pulse(**p['pulse']).start for p in target['pulses'][n:])
    if start<t2-1e-9:raise ValueError(f'probe starts at {start} before the source run ended ({t2}); run the full three-pulse TDSE instead')
    return t2

def ion_basis(config,override):
    """Ionic Hamiltonian for the probe map: the input's own (default, exactly the
    two-electron adjoint) or an explicitly converged He+ basis (radial/cutoff/lmax)."""
    if not override:return config
    override={k:v for k,v in override.items() if k!='note'}
    unknown=set(override)-{'radial','cutoff_radii','lmax'}
    if unknown:raise ValueError(f'ion override accepts radial, cutoff_radii, lmax only: {sorted(unknown)}')
    if 'radial' in override and 'cutoff_radii' not in override:
        raise ValueError('an ion basis with its own radial grid must state cutoff_radii explicitly (null = no Coulomb cutoff)')
    basis=dict(config);basis.update(override)
    if basis.get('cutoff_radii') is None:basis.pop('cutoff_radii',None)
    return basis

@exclusive_output('.case.lock')
def execute(out,source,config,run_id,signature,device,dt,ion_override=None):
    out.mkdir(parents=True,exist_ok=True)
    meta_source=json.loads((source/'run.json').read_text(encoding='utf-8'))
    t2=check_compatible(meta_source['config'],config);t3=end_time(config)
    from ionic_probe import transfer_matrix
    with np.load(source/'spectrum.npz') as d:
        if str(d['source_signature'])!=meta_source['signature']:raise ValueError('source spectrum/configuration mismatch')
        data={k:d[k] for k in ('energy','theta','phi','solid_angle_weights','amplitudes','labels')}
    labels_in=[tuple(map(int,x)) for x in data['labels']];labels_out=[tuple(map(int,x)) for x in config['ionic_channels']]
    started=time.perf_counter()
    U,info=transfer_matrix(ion_basis(config,ion_override),config['pulses'],t2,t3,labels_in,labels_out,dt=dt,device=device)
    amplitudes=np.tensordot(U,data['amplitudes'],axes=([1],[0]))
    k=np.sqrt(2*data['energy']);pes=abs(amplitudes)**2*k[None,:,None]
    total=np.sum(pes*data['solid_angle_weights'][None,None,:],axis=2)
    capped_npz(out/'spectrum.npz',config.get('storage',{}).get('max_file_bytes',4_000_000_000),energy=data['energy'],theta=data['theta'],phi=data['phi'],
               solid_angle_weights=data['solid_angle_weights'],amplitudes=amplitudes,pes=pes,angle_integrated=total,labels=np.array(labels_out),
               source_signature=signature,extraction_end_time=t3,projection_final_time=t3,
               spectrum_scope='single ionization into the explicitly recorded bound ionic channels',
               storage_method='ionic probe factorization of a completed two-pulse online spectrum')
    source_transfer=json.loads((source/'ionic_transfer.json').read_text(encoding='utf-8')) if (source/'ionic_transfer.json').exists() else {}
    transfers=[]
    for row in source_transfer.get('transfers_from_1s',[]):
        a=np.array(row['amplitude_real'])+1j*np.array(row['amplitude_imag'])
        if [tuple(map(int,x)) for x in row['labels']]!=labels_in:continue
        b=U@a;transfers.append({**{k:row[k] for k in ('requested_time','actual_time')},'labels':[list(x) for x in labels_out],
            'amplitude_real':b.real.tolist(),'amplitude_imag':b.imag.tolist(),'probabilities':(abs(b)**2).tolist(),
            'scope':'source transfer from prepared 1s mapped through the ionic probe'})
    atomic_json(out/'ionic_transfer.json',{'requested_times':config.get('ionic_transfer_times',[]),'transfers_from_1s':transfers,
                'probe_transfer_matrix':{'labels_in':[list(x) for x in labels_in],'labels_out':[list(x) for x in labels_out],
                'real':U.real.tolist(),'imag':U.imag.tolist(),**info}})
    atomic_json(out/'input.json',config)
    meta={'signature':signature,'config':config,'complete':True,'method':'ionic_probe_factorization',
          'source_run':str(source),'source_signature':meta_source['signature'],'source_end_time':t2,'end_time':t3,
          'ground_energy':meta_source.get('ground_energy'),'nrad':meta_source.get('nrad'),'dt':meta_source.get('dt'),
          'ionic_dt':info['dt_actual'],'ionic_device':device,'ion_basis_override':ion_override,'maximum_ionic_linear_residual':info['maximum_linear_residual'],
          'seconds_this_invocation':time.perf_counter()-started,'surface_storage':'derived','max_file_bytes':meta_source.get('max_file_bytes',4_000_000_000),
          'scope':'Exact tSURFF channel identity for a probe after T2; omits flux crossing the surface after T2 and components outside the source ionic channels.'}
    atomic_json(out/'run.json',meta)
    result=publish(out,run_id)
    atomic_json(out/'STATUS.json',{'state':'complete','complete_spectrum':True,'run_id':run_id,'input':'input.json','method':meta['method'],
                'spectrum':result['artifacts']['spectrum'],'figure':result['artifacts']['figure'],
                'conditional_figure':result['artifacts']['conditional_figure'],'observables':'observables.json',
                'design_targets_passed':result['target_passed'],'numerical_convergence_certified':False})
    update_index(out.parent)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,help='completed two-pulse case directory (online spectrum.npz with amplitudes)')
    p.add_argument('--config',required=True,help='three-pulse input: the source input plus role=probe pulse(s)')
    p.add_argument('--out-root',required=True);p.add_argument('--tag',default='')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda:0','lu']);p.add_argument('--ionic-dt',type=float,default=None)
    p.add_argument('--ion-basis',help='JSON with radial/cutoff_radii/lmax for a converged He+ probe map (default: the input\'s own ion)')
    a=p.parse_args();config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    override=json.loads(Path(a.ion_basis).read_text(encoding='utf-8')) if a.ion_basis else None
    source_signature=json.loads((Path(a.source)/'run.json').read_text(encoding='utf-8'))['signature']
    # The derived case is identified by its source run, the probe input and the ionic map.
    signature=hashlib.sha256(json.dumps({'config':config,'ion_basis':override,'ionic_dt':a.ionic_dt,'source':source_signature},sort_keys=True).encode()).hexdigest()
    run_id=Path(a.config).stem+'__factorized'+('__'+a.tag if a.tag else '')+'__'+signature[:10]
    if len(run_id+'__'+parameter_tag(config)+'__SI')>225:raise ValueError('case name/tag is too long for descriptive result filenames')
    result=execute(Path(a.out_root).expanduser().resolve()/run_id,Path(a.source).resolve(),config,run_id,signature,a.device,a.ionic_dt,override)
    print('COMPLETE FACTORIZED SI SPECTRUM',Path(a.out_root)/run_id/result['artifacts']['figure'],flush=True)

if __name__=='__main__':main()
