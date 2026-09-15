#!/usr/bin/env python3
"""Bounded local queue; resume verified checkpoints and stop on failed jobs.

The immutable snapshot supplies every numerical worker. This driver may wait
for one externally running case, then reuses that GPU without overlapping it.
It never labels propagation alone as convergence.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def atomic_json(path,value):
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--config-dir',required=True)
    p.add_argument('--out-root',required=True);p.add_argument('--gpu',required=True);p.add_argument('--memory-fraction',type=float,default=.45)
    p.add_argument('--after');p.add_argument('--after-pid',type=int);p.add_argument('--case',action='append',required=True)
    p.add_argument('--surface-root',required=True);p.add_argument('--queue-name',required=True);a=p.parse_args()
    root=Path(a.out_root).resolve();snap=Path(a.snapshot).resolve();configs=Path(a.config_dir).resolve()
    state={'snapshot':str(snap),'gpu':a.gpu,'completed_propagations':[],'completed_spectra':[],
           'pending':a.case.copy(),'waiting_for':a.after,'active':None,'failed':None}
    record=root/(a.queue_name+'.json');atomic_json(record,state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,HELIUM_SURFACE_ROOT=a.surface_root,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    projections=[];worker=None
    def poll_projections():
        for name,process,log in projections[:]:
            status=process.poll()
            if status is not None:
                log.close();projections.remove((name,process,log))
                if status:raise RuntimeError(f'{name}: extraction failed ({status}); inspect {name}_extract.log')
                state['completed_spectra'].append(name);atomic_json(record,state)
    def extract(name):
        # Each queue allows only one projection at a time to bound CPU and I/O.
        while projections:poll_projections();time.sleep(5)
        log=(root/(name+'_extract.log')).open('w')
        process=subprocess.Popen([sys.executable,str(snap/'scripts/extract_projected.py'),'--out',str(root/name),'--ionic-device','cpu'],env=env,stdout=log,stderr=subprocess.STDOUT)
        projections.append((name,process,log))
    try:
        if a.after:
            while True:
                path=root/a.after/'run.json';meta=json.loads(path.read_text()) if path.exists() else {}
                alive=a.after_pid and Path(f'/proc/{a.after_pid}').exists()
                if meta.get('complete') and not alive:break
                if a.after_pid and not alive and not meta.get('complete'):raise RuntimeError(f'{a.after}: prerequisite worker ended without complete result')
                time.sleep(10)
            state['waiting_for']=None;atomic_json(record,state)
        for name in a.case:
            poll_projections();state['active']=name;state['pending'].remove(name);atomic_json(record,state)
            with (root/(name+'_gpu'+a.gpu+'.log')).open('a') as log:
                command=[sys.executable,str(snap/'scripts/followup_campaign.py'),'--case',name,'--config-dir',str(configs),'--out-root',str(root),
                         '--device','cuda:0','--memory-fraction',str(a.memory_fraction),'--preconditioner-precision','complex64','--propagate-only']
                print('START',name,flush=True);worker=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                while worker.poll() is None:poll_projections();time.sleep(10)
                if worker.returncode:raise RuntimeError(f'{name}: propagation failed ({worker.returncode})')
            meta=json.loads((root/name/'run.json').read_text())
            if not meta.get('complete'):raise RuntimeError(f'{name}: checkpointed early; resume queue explicitly')
            state['completed_propagations'].append(name);state['active']=None;atomic_json(record,state)
            # This helper performs the history checks before starting extraction.
            with (root/(name+'_validation.log')).open('w') as log:
                command=[sys.executable,str(snap/'scripts/wait_extract.py'),'--out',str(root/name),'--snapshot',str(snap),'--validate-only']
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            extract(name);print('PROPAGATED',name,flush=True)
        while projections:poll_projections();time.sleep(5)
    except BaseException as error:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGUSR1)
        state['failed']=str(error);atomic_json(record,state);raise
    atomic_json(record,state);print('QUEUE COMPLETE; run acceptance audit separately',flush=True)

if __name__=='__main__':main()
