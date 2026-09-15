#!/usr/bin/env python3
"""Run the predeclared quadrature checks as soon as each base cache completes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/convergence_complete/production_plan.json')
p.add_argument('--out-root',default='results/convergence_complete');p.add_argument('--snapshot',required=True);p.add_argument('--case',action='append');a=p.parse_args()
root=Path(a.out_root).resolve();snapshot=Path(a.snapshot).resolve();plan=json.loads(Path(a.plan).read_text())
pending=[j for j in plan['extraction_jobs'] if a.case is None or j['case'] in a.case];done=[]
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
while pending:
    ready=[j for j in pending if (root/j['case']/'spectrum.npz').exists() and (root/j['case']/'channel_projection.json').exists()]
    if not ready:time.sleep(10);continue
    job=ready[0];out=root/job['case'];name=job['file']
    if not (out/name).exists():
        command=[sys.executable,str(snapshot/'scripts/extract_projected.py'),'--out',str(out),'--name',name,'--ionic-device','cpu',*job['args']]
        with (out/(Path(name).stem+'.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    pending.remove(job);done.append(job);print('COMPLETE',job,flush=True)
    temporary=root/'quadrature_queue.tmp.json';temporary.write_text(json.dumps({'case_filter':a.case,'completed':done,'pending':pending},indent=2)+'\n');os.replace(temporary,root/'quadrature_queue.json')
