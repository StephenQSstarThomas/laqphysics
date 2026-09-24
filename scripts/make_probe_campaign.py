#!/usr/bin/env python3
"""Three-pulse inputs: selected preparation + the third, circular two-photon probe.

Pulse 1 ionizes (1.2 a.u., z), pulse 2 is the sigma+ pi transfer 1s -> 2p(+1);
pulse 3 (role=probe) is the circular low-frequency pulse of the work plan, which
drives the prepared ion 2p(+1) through a two-photon (5/36 a.u.) or three-photon
(0.14 a.u.) resonance. It starts when the two-pulse source input ends, after the
old electrons have crossed the tSURFF surface; this is also what makes the fast
ionic factorization (scripts/apply_ionic_probe.py) applicable to every case.

The He+ n=5 manifold enters at these frequencies, so every probe input uses an
extended radial box (Coulomb cutoff 32-40 bohr, surface 44, ECS from 64) and
records all n<=3 channels plus the n=5 states the probe reaches (see N5). The probe parameters come
from the prepared-ion design scan (results/probe_20260924/ion_design); they are
inputs to be run, not completed results.
"""
import json,sys
from copy import deepcopy
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from campaign_resources import estimate
from pulses import Pulse

FIELD=.02;CYCLES=48;POST=120.
PROBES={'w014':.14,'w5o36':5/36}
POLARIZATION={'sigma_minus':'sigma-','sigma_plus':'sigma+'}

def extend_grid(c,outer=64.,step=4.,order=6,cutoff=(32.,40.),surface=44.):
    edges=list(c['radial']['edges']);orders=c['radial']['order']
    orders=list(orders) if isinstance(orders,list) else [orders]*(len(edges)-1)
    extra=[float(x) for x in np.arange(edges[-1]+step,outer+1e-9,step)]
    c['radial']=dict(c['radial'],edges=edges+extra,order=orders+[order]*len(extra))
    c.update(cutoff_radii=list(cutoff),surface=float(surface))

# n=5 states reached from 2p(+1) by three circular photons (sigma-: m=-2; sigma+: m=+4)
# and their mirror images; an xy-plane field conserves the parity of l+m exactly.
# The prepared-ion design scan finds every other n>=4 final state below 1e-7.
N5=[[5,2,-2],[5,2,2],[5,4,-4],[5,4,-2],[5,4,2],[5,4,4]]

def channels(lmax,n5=True):
    labels=[[n,l,m] for n in range(1,4) for l in range(min(n,lmax+1)) for m in range(-l,l+1)]
    if n5:labels+=[x for x in N5 if x[1]<=lmax]
    return labels

def pulse_end(c):return max(Pulse(**p['pulse']).start+Pulse(**p['pulse']).duration for p in c['pulses'])

def commensurate(t,dt,stride=1):
    """Smallest end time >= t that run3d splits into exactly T/dt steps of size dt.

    run3d uses dt_actual=T/n with n=ceil(T/dt) rounded up to a multiple of the
    surface stride; matching end times give the source and the probe input the
    identical time grid up to the probe start."""
    for k in range(int(np.ceil(t/dt-1e-9)),int(np.ceil(t/dt))+1000*stride):
        T=float(f'{k*dt:.12g}')
        if k%stride==0 and T>=t-1e-9 and int(np.ceil(T/dt))==k and abs(T/k-dt)<=1e-15*dt*k:return T
    raise ValueError('no commensurate end time found')

def build(prep,omega,polarization,field=FIELD,cycles=CYCLES,n5=True,post=POST):
    """Return (source, probe) inputs sharing every numerical parameter and time grid."""
    source=deepcopy(prep);source.pop('post_time',None);dt=prep['dt'];stride=prep.get('surface_stride',1)
    source['pulses'][0]['role']='ionize';source['pulses'][1]['role']='prepare'
    start=commensurate(pulse_end(prep)+prep.get('post_time',120.),dt,stride)
    source['end_time']=start
    source['ionic_channels']=channels(prep['lmax'],n5=False)
    probe=deepcopy(source)
    probe['pulses'].append({'pulse':{'omega':float(omega),'cycles':cycles,'field':field,'start':start},'polarization':polarization,'role':'probe'})
    probe['end_time']=commensurate(pulse_end(probe)+post,dt,stride)
    probe['ionic_channels']=channels(prep['lmax'],n5=n5)
    # ~6e4 steps: keep on-line spectra only (no projected history, several times less disk).
    probe['storage']=dict(probe.get('storage',{}),mode='spectrum')
    targets=[[3,1,-1],[5,4,-2],[5,2,-2]] if polarization=='sigma-' else [[5,4,4]]
    probe['spectrum']=dict(probe.get('spectrum',{}),plot_channels=[[1,0,0],[2,1,1]]+[x for x in targets if x in probe['ionic_channels']])
    probe['probe']={'prepared_channel':[2,1,1],'target_channels':[x for x in targets if x in probe['ionic_channels']],
        'transition':('two-photon 2p(+1)->3p(-1) at 5/36; three-photon 2p(+1)->n=5 (5g-2, 5d-2) near 0.14' if polarization=='sigma-'
                      else 'sigma+: no n=3 state with m=+3 (two-photon forbidden); three-photon stretched ladder 2p(+1)->5g(+4) near 0.14'),
        'field_free_two_photon_resonance_au':5/36,'field_free_three_photon_2p_to_n5_au':(.5-.08)/3,
        'design_scan':'results/probe_20260924/ion_design/summary.json',
        'timing':'starts when the two-pulse source input ends; the ionized electrons have already crossed the surface',
        'physics_note':'An ion-only operation after the electron left cannot change the unconditional SI spectrum; coincidence spectra are redistributed. Compare with the source/no-probe control.'}
    return source,probe

def smoke():
    """Tiny, fast three-pulse chain for installation and code-path checks only."""
    # Repository layout, or the standalone bundle where the file ships as configs/handoff_smoke.json.
    path=next(p for p in [root/'packaging/minimum/handoff_smoke.json',root/'configs/handoff_smoke.json'] if p.exists())
    base=json.loads(path.read_text(encoding='utf-8'))
    base['pulses']=[{'pulse':{'omega':5.,'cycles':2,'field':.15},'polarization':'z'},
                    {'pulse':{'omega':1.5,'cycles':3,'field':.2,'start':3.},'polarization':'sigma+'}]
    base['post_time']=6.;base['ionic_transfer_times']=[3.];base['checkpoint_every']=20
    base['ionic_channels']=[[1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]]
    source,probe=build(base,.7,'sigma-',field=.05,cycles=2,n5=False,post=4.)
    probe['spectrum']['plot_channels']=[[1,0,0],[2,1,1],[2,1,-1]];probe['probe']['target_channels']=[[2,1,-1]]
    probe['probe']['scope']='installation smoke test on a tiny grid; not physics'
    source['note']='installation smoke test on a tiny grid; not physics'
    return source,probe

def validation():
    """Small model where the factorization assumptions hold: hydrogenic n<=3 ionic
    channels well inside the cutoff, a weak probe near the 2p->3d resonance that
    cannot ionize the neutral atom, and post-pulse times long enough for the flux.
    Full three-pulse TDSE vs apply_ionic_probe.py tests the factorization and the
    whole three-pulse code path at modest cost (minutes on one GPU)."""
    edges=[0,.5,1,2,4,6,8,10,12,14,16,18,20,22,24]
    base={'radial':{'edges':edges,'order':4,'tail':12,'ecs_angle':.5,'alpha':.8},'lmax':2,'M':None,'total_Lmax':2,
          'angular_representation':'bipolar','natural_parity':False,'coulomb_contraction':'blocks',
          'cutoff_radii':[12.,16.],'surface':20.,'dt':.1,'surface_stride':1,'checkpoint_every':500,
          'ground_tolerance':1e-10,'time_integrator':'cf4-pade','ionic_propagator':'cf4','linear_tolerance':1e-10,
          'pulses':[{'pulse':{'omega':1.2,'cycles':12,'field':.08},'polarization':'z'},
                    {'pulse':{'omega':1.5,'cycles':24,'field':1.5/(.37246776951390165*24),'start':70.},'polarization':'sigma+'}],
          'post_time':60.,'ionic_transfer_times':[70.],
          'spectrum':{'energy_segments':[[.05,.18,14],[.18,.75,115],[.75,1.6,35]],'theta_points':8,'phi_points':12},
          'storage':{'mode':'spectrum','max_file_bytes':4_000_000_000,'ionic_block_frames':32,'ionic_device':'cpu','replay_workers':2,'accumulator_device':'auto'}}
    source,probe=build(base,.3,'sigma-',field=.01,cycles=4,n5=False,post=60.)
    probe['probe']['target_channels']=[[3,2,0]]
    probe['spectrum']['plot_channels']=[[1,0,0],[2,1,1],[3,2,0]]
    probe['probe']['scope']='factorization/code-path validation model (lmax=2, order 4); not converged physics'
    return source,probe

def main():
    out=root/'configs/probe_20260924';resources=[];cases={}
    prep=json.loads((root/'configs/preparation_20260921/production/reference.json').read_text(encoding='utf-8'))
    extend_grid(prep)
    for key,omega in PROBES.items():
        for pol,polarization in POLARIZATION.items():
            source,probe=build(prep,omega,polarization);cases['source']=source;cases[f'probe_{key}_{pol}']=probe
    # The prepared-ion scan converges only for one-electron lmax>=6 (sigma+: 5g(+4) ionizes
    # through l=5; sigma-: 5g(-2) is overestimated 2.6x at lmax=4). A two-electron lmax=6
    # probe peaks at 40.3 GiB GPU memory (measured), beyond GPU40G: the converged probe
    # physics comes from the source + apply_ionic_probe.py --ion-basis ion_basis_converged.json,
    # and the lmax=4 full probes check that map with the source's identical ionic Hamiltonian.
    for pol in POLARIZATION:
        l6=deepcopy(cases[f'probe_w014_{pol}']);l6['lmax']=6;cases[f'probe_w014_{pol}_l6']=l6
    none=deepcopy(cases['probe_w014_sigma_minus']);none['pulses']=none['pulses'][:2];none.pop('probe')
    none['spectrum']=dict(none['spectrum'],plot_channels=[[1,0,0],[2,1,1]]);cases['no_probe_same_end']=none
    production=out/'production';production.mkdir(parents=True,exist_ok=True)
    for name,c in cases.items():
        (production/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8');resources.append({'case':name,**estimate(c)})
    plan={'cases':list(cases),'primary_cases':['source','probe_w014_sigma_minus','probe_w014_sigma_plus'],
          'factorization_checks':[{'source':'source','probe':name} for name in cases if name.startswith('probe_') and cases[name]['lmax']==cases['source']['lmax']],
          'pairs':[],'acceptance':{'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,'energy_windows':[[.18,.42]]},
          'scope':'Three-pulse probe inputs on the extended box; source=two-pulse preparation ending at the probe start. None is certified by configuration alone.',
          'optional_cases':['probe_w5o36_sigma_minus','probe_w5o36_sigma_plus','no_probe_same_end','probe_w014_sigma_minus_l6','probe_w014_sigma_plus_l6'],
          'optional_note':'*_l6 need a GPU with >=48 GB (measured peak 40.3 GiB allocated); they do not fit GPU40G.',
          'recommended_order':'source first; then apply_ionic_probe.py with configs/probe_20260924/ion_basis_converged.json gives the converged probe result from it in about an hour of CPU/GPU; the two full lmax=4 probes at 0.14 a.u. check that map at production scale (identical ionic Hamiltonian). 5/36, no_probe_same_end and *_l6 are optional.'}
    (production/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    (production/'resource_estimates.json').write_text(json.dumps(resources,indent=2)+'\n',encoding='utf-8')
    for folder,maker in [('smoke',smoke),('validation',validation)]:
        small=out/folder;small.mkdir(exist_ok=True);source,probe=maker();inputs={'source':source,'probe':probe}
        if folder=='validation':
            # Opposite helicity: sigma+ absorption from 2p(+1) reaches 3d(+2) near the same resonance.
            plus=deepcopy(probe);plus['pulses'][2]['polarization']='sigma+';plus['probe']['target_channels']=[[3,2,2]]
            plus['spectrum']=dict(plus['spectrum'],plot_channels=[[1,0,0],[2,1,1],[3,2,2]]);inputs['probe_sigma_plus']=plus
        for name,c in inputs.items():(small/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
        (small/'plan.json').write_text(json.dumps({'cases':list(inputs),'factorization_checks':[{'source':'source','probe':x} for x in inputs if x!='source'],
            'scope':f'{folder} inputs for the three-pulse chain; not production physics'},indent=2)+'\n',encoding='utf-8')
    for r in resources:print(f"{r['case']:28s} steps={r['steps']:6d} radial={r['radial_points']} channels={r['angular_channels']} "
                             f"ion channels={r['ionic_final_channels']} persistent={r['persistent_output_estimate_bytes']/1e9:.2f} GB largest file={r['largest_output_file_upper_bytes']/1e9:.2f} GB")

if __name__=='__main__':main()
