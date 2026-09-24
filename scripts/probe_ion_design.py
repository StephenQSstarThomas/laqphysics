#!/usr/bin/env python3
"""Design scan for the third (two-photon) probe on the prepared He+ 2p(+1) ion.

Each case is one continuum-inclusive He+ TDSE (python/ionic_tdse.py) on the SAME
radial grid, Coulomb cutoff and lmax as a two-electron input, starting in 2p(+1)
with the probe alone. Results are final bound populations (n<=6) and the lost
(ionized/unrecorded) probability. This chooses probe parameters before any
two-electron propagation; it is not an SI spectrum.
"""
import argparse,json,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))

def case_config(base,probe,dt,lmax):
    keys=('radial','cutoff_radii')
    c={k:base[k] for k in keys if k in base}
    c.update(lmax=int(lmax if lmax is not None else base['lmax']),initial=[2,1,1],dt=dt,linear_tolerance=1e-11,checkpoint_every=200,
             pulses=[{'pulse':{'omega':probe['omega'],'cycles':probe['cycles'],'field':probe['field']},'polarization':probe['polarization']}])
    return c

def name_of(p):
    pol={'sigma+':'sp','sigma-':'sm','z':'z'}[p['polarization']]
    return f"w{p['omega']:.6f}_F{p['field']:g}_N{p['cycles']:g}_{pol}".replace('.','p')

def worker(args):
    config,out,device,threads=args
    os.environ['OMP_NUM_THREADS']=str(threads)
    import torch;torch.set_num_threads(threads)
    import numpy as np
    from ionic_tdse import run
    out=Path(out)
    previous=json.loads((out/'run.json').read_text(encoding='utf-8')) if (out/'run.json').exists() else None
    if previous is not None and previous['config']!=config:
        raise ValueError(f'{out} holds a different case (grid, lmax, dt or pulse); choose another --out')
    meta=previous if previous is not None and previous.get('complete') else run(config,out,device=device,resume=(out/'checkpoint.npz').exists())
    with np.load(out/'populations.npz') as d:labels=[tuple(map(int,x)) for x in d['labels']];prob=d['probabilities']
    top=sorted(zip(prob.tolist(),labels),reverse=True)[:8]
    return {'case':out.name,'config':config,'P_2p_plus1':float(prob[labels.index((2,1,1))]),
            'P_3p_minus1':float(prob[labels.index((3,1,-1))]) if (3,1,-1) in labels else None,
            'P_3p':meta['P_3p'],'P_3d':meta['P_3d'],'bound_n_le_6':meta['bound_n_le_6_probability'],
            'lost_or_unrecorded':1-meta['bound_n_le_6_probability'],'largest_final_states':[{'label':list(l),'probability':p} for p,l in top],
            'maximum_linear_residual':meta['maximum_linear_residual'],'seconds':meta['seconds_this_invocation']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True,help='two-electron input supplying radial grid, cutoff and lmax')
    p.add_argument('--scan',required=True,help='JSON list of {omega,field,cycles,polarization}')
    p.add_argument('--out',required=True);p.add_argument('--dt',type=float,default=.25);p.add_argument('--lmax',type=int)
    p.add_argument('--devices',nargs='+',default=['cpu']);p.add_argument('--jobs',type=int,default=4);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    base=json.loads(Path(a.base).read_text(encoding='utf-8'));scan=json.loads(Path(a.scan).read_text(encoding='utf-8'))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tasks=[]
    for i,probe in enumerate(scan):
        tasks.append((case_config(base,probe,a.dt,a.lmax),str(out/name_of(probe)),a.devices[i%len(a.devices)],a.threads))
    rows=[]
    with ProcessPoolExecutor(a.jobs,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            row=future.result();rows.append(row);print(json.dumps({k:row[k] for k in ('case','P_2p_plus1','P_3p_minus1','P_3p','P_3d','lost_or_unrecorded')}),flush=True)
    rows.sort(key=lambda r:r['case'])
    summary={'base_input':str(Path(a.base)),'dt':a.dt,'initial':'He+ 2p(+1), probe alone from t=0','cases':rows,
             'scope':'Prepared-ion design scan on the two-electron model ionic Hamiltonian; not an SI spectrum or a convergence certificate.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
