#!/usr/bin/env python3
"""Audit prepared-He+ refinements independently of neutral two-electron spectra."""
import argparse,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out-root',default='results/convergence_complete/ions')
p.add_argument('--reference',default='results/followup/ion_2p_counter');a=p.parse_args()
root=Path(a.out_root);reference=Path(a.reference);base=json.loads((reference/'run.json').read_text());bp=np.load(reference/'populations.npz');rows=[]
for name in ['inner10','outer8','l6','dt0125','extent80','tail36','angle06']:
    folder=root/name;path=folder/'run.json';row={'case':name,'status':'pending'}
    if path.exists():
        meta=json.loads(path.read_text())
        if meta.get('complete'):
            p=np.load(folder/'populations.npz');labels=list(map(tuple,p['labels']));reference_labels=list(map(tuple,bp['labels']))
            if labels!=reference_labels:raise ValueError('bound-channel lists differ')
            if not np.isfinite(p['probabilities']).all():raise ValueError('nonfinite bound population')
            change=meta['P_3p']/base['P_3p']-1
            passed=abs(change)<=.02 and meta['P_3d']<1e-10 and 0<=meta['inner_norm']<=1+1e-6 and meta['maximum_linear_residual']<=1.01*meta['config']['linear_tolerance']
            row.update(status='passed' if passed else 'failed',P_3p=meta['P_3p'],relative_P_3p_change=change,P_3d=meta['P_3d'],
                       P_3p_m_minus1=float(p['probabilities'][labels.index((3,1,-1))]),
                       inner_norm=meta['inner_norm'],bound_population_L1_difference=float(sum(abs(p['probabilities']-bp['probabilities']))),
                       maximum_linear_residual=meta['maximum_linear_residual'])
    rows.append(row)
result={'scope':'prepared He+ 2p+1 driven by counter-rotating 224-cycle pulse at omega=5/36; final bound populations',
        'reference':str(reference),'reference_P_3p':base['P_3p'],'criteria':{'relative_P_3p_change':.02,'final_P_3d_maximum':1e-10},
        'all_defined_ion_axes_passed':all(r['status']=='passed' for r in rows),'comparisons':rows}
root.mkdir(parents=True,exist_ok=True);temp=root/'acceptance.tmp.json';temp.write_text(json.dumps(result,indent=2)+'\n');os.replace(temp,root/'acceptance.json')
for row in rows:print(row)
