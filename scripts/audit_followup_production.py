#!/usr/bin/env python3
"""Report pending/failed/passed gates from actual production data, without peak shifts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from analyze_followup import fitted_peaks
p=argparse.ArgumentParser();p.add_argument('--out-root',default='results/followup')
p.add_argument('--plan',default='configs/followup/production_plan.json');p.add_argument('--config-dir');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.out_root)
plan_path=Path(a.plan);plan=json.loads(plan_path.read_text());config_dir=Path(a.config_dir) if a.config_dir else plan_path.parent
rows=[];cache={}
def load(name):
    if name in cache:return cache[name]
    folder=out/name
    if not (folder/'run.json').exists() or not (folder/'spectrum.npz').exists():return None
    meta=json.loads((folder/'run.json').read_text())
    if not meta['complete']:return None
    config=json.loads((config_dir/(name+'.json')).read_text())
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if signature!=meta['signature']:raise ValueError(f'{name}: configuration mismatch')
    cache[name]=np.load(folder/'spectrum.npz');return cache[name]

def compare_window(first,second,window):
    mask=np.ones(len(second['energy']),bool) if window is None else (second['energy']>=window[0])&(second['energy']<=window[1])
    e=second['energy'][mask];qa=first['angle_integrated'];qb=second['angle_integrated'][:,mask]
    if len(e)<3:return {'status':'failed','reason':'insufficient energy samples','energy_window':window}
    def indices(data):return [list(map(tuple,data['labels'])).index(tuple(x)) for x in plan['acceptance']['gate_channels']]
    pa=np.array([np.interp(e,first['energy'],q) for q in qa[indices(first)]]);pb=qb[indices(second)]
    ya=simpson(pa,x=e,axis=1);yb=simpson(pb,x=e,axis=1)
    if np.any(ya<=0) or np.any(yb<=0):return {'status':'failed','reason':'a gate channel has zero yield','energy_window':window}
    shape=simpson(abs(pa/ya[:,None]-pb/yb[:,None]),x=e,axis=1);change=yb/ya-1
    peaks_a,_=fitted_peaks(e,np.array([np.interp(e,first['energy'],q) for q in qa]).sum(axis=0))
    peaks_b,_=fitted_peaks(e,qb.sum(axis=0))
    shift=float(np.max(abs(np.array(peaks_a)-peaks_b))) if len(peaks_a)==len(peaks_b) and peaks_a else None
    criteria=plan['acceptance'];passed=shift is not None and shift<=criteria['peak_shift_au'] and max(shape)<=criteria['normalized_shape_L1'] and max(abs(change))<=criteria['relative_yield']
    return {'status':'passed' if passed else 'failed','energy_window':[float(e[0]),float(e[-1])],
            'normalized_shape_L1':shape.tolist(),'relative_yield_change':change.tolist(),'maximum_peak_shift':shift,
            'all_recorded_channel_yields':simpson(qb,x=e,axis=1).tolist(),'labels':second['labels'].tolist()}
for pair in plan['pairs']:
    first=load(pair['reference']);second=load(pair['refinement']);row=dict(pair)
    if first is None or second is None:row['status']='pending'
    else:
        row['windows']=[compare_window(first,second,window) for window in plan['acceptance'].get('energy_windows',[None])]
        row['status']='passed' if all(w['status']=='passed' for w in row['windows']) else 'failed'
    rows.append(row)
all_cases=all(load(name) is not None for name in plan['cases'])
result={'all_gates_passed':bool(rows and all(r['status']=='passed' for r in rows)),'all_cases_complete':all_cases,
        'comparisons':rows,'criteria':plan['acceptance'],'scope':plan['scope'],'completion_requires':plan['completion_requires']}
out.mkdir(parents=True,exist_ok=True);(out/'production_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
