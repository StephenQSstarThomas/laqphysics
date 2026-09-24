#!/usr/bin/env python3
"""Unpack a minimum bundle in isolation (no .git) and run its documented entry points.

The outer environment is deliberately hostile to text encoding (LC_ALL=C,
PYTHONUTF8=0), like the node that produced the ASCII error; handoff.sh must cope.
Writes a JSON record and one log per step.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--logs',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--gpu',help='CUDA_VISIBLE_DEVICES value for an extra probe smoke on cuda:0');a=p.parse_args()
    digest=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    sidecar=a.archive.with_name(a.archive.name+'.sha256').read_text(encoding='utf-8').split()[0]
    if digest!=sidecar:raise SystemExit('archive does not match its .sha256 sidecar')
    a.work.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
    unpack=Path(tempfile.mkdtemp(prefix='unpacked_',dir=a.work))
    with tarfile.open(a.archive) as archive:archive.extractall(unpack,filter='data')
    root=next(unpack.iterdir());outputs=a.work/'local_runs'
    env={k:v for k,v in os.environ.items() if not k.startswith(('HELIUM_','LC_','LANG','PYTHONUTF8','PYTHONIOENCODING','SLURM_'))}
    env.update(LC_ALL='C',PYTHONUTF8='0',PYTHONCOERCECLOCALE='0',HELIUM_PYTHON=sys.executable,HELIUM_OUTPUT_ROOT=str(outputs),
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HELIUM_TORCH_THREADS='2')
    originals={f:f.read_bytes() for f in (root/'configs/probe_20260924').rglob('*.json')}
    steps=[('verify_initial',['./handoff.sh','verify'],{}),
           ('compile_and_both_test_suites',['./handoff.sh','test'],{}),
           ('cpu_smoke',['./handoff.sh','smoke','cpu'],{}),
           ('probe_smoke_cpu_full_and_factorized',['./handoff.sh','probe-smoke','cpu'],{}),
           *([('probe_smoke_gpu_full_and_factorized',['./handoff.sh','probe-smoke','cuda:0'],{'CUDA_VISIBLE_DEVICES':a.gpu,'HELIUM_OUTPUT_ROOT':str(a.work/'local_runs_gpu')})] if a.gpu else []),
           ('replot_completed_cases',['./handoff.sh','render'],{}),
           ('production_resource_estimate',['./handoff.sh','estimate'],{}),
           ('probe_resource_estimate',['./handoff.sh','estimate','configs/probe_20260924/production/probe_w014_sigma_minus.json'],{}),
           ('regenerate_probe_inputs',[sys.executable,'scripts/make_probe_campaign.py'],{}),
           ('snapshot_without_git',['./handoff.sh','snapshot'],{}),
           ('slurm_body_with_real_frozen_worker',[sys.executable,'results/probe_20260924/validation/check_slurm_probe.py',str(a.work/'slurm_check')],{}),
           ('verify_after_running',['./handoff.sh','verify'],{})]
    rows=[]
    for name,command,extra in steps:
        start=time.time();log=a.logs/f'{name}.log'
        with log.open('w',encoding='utf-8') as stream:
            run=subprocess.run(command,cwd=root,env={**env,**extra},stdout=stream,stderr=subprocess.STDOUT,text=True)
        rows.append({'check':name,'command':' '.join(command),'exit_code':run.returncode,'seconds':round(time.time()-start,1),'log':log.name})
        print(name,run.returncode,flush=True)
        if run.returncode:break
    regenerated_identical=all(f.read_bytes()==data for f,data in originals.items())
    passed=all(r['exit_code']==0 for r in rows) and len(rows)==len(steps) and regenerated_identical
    record={'passed':passed,'archive':a.archive.name,'archive_sha256':digest,'archive_bytes':a.archive.stat().st_size,
            'package_directory':str(root),'git_directory_present':(root/'.git').exists(),
            'outer_environment':'LC_ALL=C, PYTHONUTF8=0, PYTHONCOERCECLOCALE=0 (ASCII default text encoding)',
            'checks':rows,'regenerated_probe_inputs_byte_identical':regenerated_identical,
            'source_commit':json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))['source_commit'],
            'scope':'Isolated unpack without Git: compile, Release/Debug tests, two-pulse and three-pulse smoke runs (full TDSE and ionic factorization), replot, estimates, input regeneration, snapshot, Slurm body with the frozen worker (srun mocked). No real cluster submission.'}
    a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:record[k] for k in ('passed','archive_sha256','regenerated_probe_inputs_byte_identical')},indent=2))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
