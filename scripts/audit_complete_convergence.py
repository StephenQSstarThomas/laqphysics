#!/usr/bin/env python3
"""Audit actual full results, including extraction gates and missing checks."""
import argparse,hashlib,json,sys,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from convergence import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/convergence_complete/production_plan.json')
    p.add_argument('--out-root',default='results/convergence_complete');a=p.parse_args()
    plan_path=Path(a.plan);plan=json.loads(plan_path.read_text());root=Path(a.out_root);rows=[]
    criteria=plan['acceptance']
    def load(case,filename='spectrum.npz'):
        folder=root/case
        if not (folder/'run.json').exists() or not (folder/filename).exists():return None
        meta=json.loads((folder/'run.json').read_text())
        if not meta.get('complete'):return None
        config=json.loads((plan_path.parent/(case+'.json')).read_text())
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        if meta['signature']!=signature:raise ValueError(f'{case}: configuration signature mismatch')
        data=np.load(folder/filename)
        if str(data['source_signature'])!=signature:raise ValueError(f'{case}/{filename}: stale spectrum')
        return data
    for pair in plan['pairs']+plan.get('extraction_pairs',[]):
        row=dict(pair);first=load(pair['reference'],pair.get('reference_file','spectrum.npz'))
        second=load(pair['refinement'],pair.get('refinement_file','spectrum.npz'))
        if first is None or second is None:row['status']='pending'
        else:
            row['windows']=[compare(first,second,pair.get('acceptance',criteria),window) for window in pair.get('energy_windows',criteria.get('energy_windows',[None]))]
            row['status']='passed' if all(x['status']=='passed' for x in row['windows']) else 'failed'
        rows.append(row)
    complete=all(load(name) is not None for name in plan['cases'])
    result={'scope':plan['scope'],'all_cases_complete':complete,
            'all_defined_gates_passed':bool(complete and rows and all(r['status']=='passed' for r in rows)),
            'comparisons':rows,'criteria':criteria,'completion_requires':plan['completion_requires']}
    temp=root/'complete_acceptance.tmp.json';temp.write_text(json.dumps(result,indent=2)+'\n');os.replace(temp,root/'complete_acceptance.json')
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    for row in rows:print(row['status'].upper(),row['axis'])

if __name__=='__main__':main()
