#!/usr/bin/env python3
from pathlib import Path
import sys,json,time,argparse,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from fedvr import make_grid
from ground_s import GroundS
p=argparse.ArgumentParser();p.add_argument('--high-accuracy',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'results/followup';rows=[]
edges=[0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12,16]
cases=[(8,10),(10,12),(12,14)] if a.high_accuracy else [(4,6),(6,6),(8,6),(8,8),(8,10),(10,10)]
filename='ground_high_accuracy.json' if a.high_accuracy else 'ground_refinement.json'
for lmax,core_order in cases:
    orders=[core_order if b<=2 else 6 for b in edges[1:]];start=time.perf_counter()
    g=make_grid(edges,order=orders);h=GroundS.from_grid(g,lmax)
    E,x,res=h.solve(tol=1e-9,cache_dir=out/'ground_cache',expand=False)
    row={'lmax':lmax,'core_order':core_order,'outer_order':6,'nrad':h.n,'channels':h.nc,'energy':E,
         'error_vs_Schwartz':E-(-2.9037243770341196),'residual':res,'seconds':time.perf_counter()-start}
    row['radial']={'edges':edges,'order':orders};row['solver']='preconditioned LOBPCG'
    rows.append(row);print(row,flush=True);(out/filename).write_text(json.dumps(rows,indent=2)+'\n')
