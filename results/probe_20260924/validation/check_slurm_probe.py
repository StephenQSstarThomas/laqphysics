#!/usr/bin/env python3
"""Run the real Slurm body (spectra_cpu.slurm -> spectrum_job.sh -> frozen snapshot ->
simulate_spectrum.py) on the three-pulse smoke plan. Only `srun` is mocked: it
executes its command line. The outer environment is deliberately hostile to text
encoding (LC_ALL=C, PYTHONUTF8=0), as on the node that produced the ASCII error.
Usage: check_slurm_probe.py [WORK_DIR] [REPORT_JSON]; the report defaults to WORK_DIR,
so running it inside a verified bundle never modifies a package member."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
repo=Path(__file__).resolve().parents[3];work=Path(sys.argv[1] if len(sys.argv)>1 else repo/'local_runs/slurm_probe_check')
work.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='mock_bin_',dir=work) as temp:
    binary=Path(temp);(binary/'srun').write_text('#!/usr/bin/env bash\n[[ "$1" == --cpu-bind=* ]] && shift\nexec "$@"\n');(binary/'srun').chmod(0o755)
    env={k:v for k,v in os.environ.items() if not k.startswith(('HELIUM_','LC_','LANG','SLURM_'))}
    env.update(PATH=str(binary)+os.pathsep+os.environ['PATH'],LC_ALL='C',PYTHONUTF8='0',PYTHONCOERCECLOCALE='0',
               HELIUM_REPO=str(repo),HELIUM_OUTPUT_ROOT=str(work/'outputs'),HELIUM_PYTHON=sys.executable,
               HELIUM_PLAN='configs/probe_20260924/smoke/plan.json',HELIUM_RUN_TAG='slurmcheck',SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='4')
    rows=[]
    for task in ['0','1']:
        run=subprocess.run(['bash',str(repo/'scripts/spectra_cpu.slurm')],cwd=repo,env=dict(env,SLURM_ARRAY_TASK_ID=task),capture_output=True,text=True)
        rows.append({'array_task':task,'exit_code':run.returncode,'stdout_tail':run.stdout.strip().splitlines()[-2:],'stderr_tail':run.stderr.strip().splitlines()[-3:] if run.returncode else []})
        if run.returncode:print(run.stdout,run.stderr);raise SystemExit(f'array task {task} failed')
statuses={p.parent.name:json.loads(p.read_text(encoding='utf-8'))['state'] for p in (work/'outputs').glob('*/STATUS.json')}
snapshots=sorted(x.name for x in (work/'outputs/code').iterdir() if x.is_dir())
index=(work/'outputs/INDEX.md').read_text(encoding='utf-8')
report={'jobs':rows,'case_states':statuses,'frozen_snapshots':snapshots,'index_has_probe_column':'第三束探测' in index,
        'environment':'LC_ALL=C, PYTHONUTF8=0 outside; spectrum_job.sh exports PYTHONUTF8=1','srun':'mocked: executes its argument list',
        'passed':all(r['exit_code']==0 for r in rows) and set(statuses.values())=={'complete'} and len(statuses)==2}
report_path=Path(sys.argv[2]) if len(sys.argv)>2 else work/'slurm_probe_check.json'
report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
