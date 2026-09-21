import json,os,subprocess,tempfile,hashlib
from pathlib import Path
repo=Path('/playpen/shiqiu/task-jiukongdian');result=[]
files=list(repo.joinpath('scripts').glob('*.slurm'))+list(repo.joinpath('scripts').glob('*.sh'))
for f in files:subprocess.run(['bash','-n',str(f)],check=True)
with tempfile.TemporaryDirectory(prefix='slurm_mock_',dir='/playpen1/shiqiu/laqphysics-data/test_tmp') as temp:
    p=Path(temp);binary=p/'bin';binary.mkdir();record=p/'dispatch.json'
    for name,code in {
        'make':'#!/bin/sh\nexit 0\n',
        'srun':'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
Path(os.environ['MOCK_DISPATCH']).write_text(json.dumps({'argv':sys.argv[1:],'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','HELIUM_DEVICE']}},indent=2))
''',
        'sbatch':'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
assert Path('logs').is_dir()
Path(os.environ['MOCK_DISPATCH']).write_text(json.dumps({'argv':sys.argv[1:],'logs_existed_before_submission':True},indent=2))
'''} .items():
        (binary/name).write_text(code);(binary/name).chmod(0o755)
    base=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],HELIUM_REPO=str(repo),HELIUM_OUTPUT_ROOT=str(p/'outputs'),
              HELIUM_SNAPSHOT=str(p/'frozen'),HELIUM_PYTHON='python',SLURM_NTASKS='1',SLURM_ARRAY_TASK_ID='0',MOCK_DISPATCH=str(record))
    for key in ['HELIUM_CONFIG','HELIUM_CONFIG_DIR','HELIUM_PLAN','HELIUM_TORCH_THREADS','HELIUM_OMP_THREADS','HELIUM_STORAGE_MODE']:base.pop(key,None)
    for kind,cores,device in [('gpu',16,'cuda:0'),('cpu',64,'cpu')]:
        text=(repo/f'scripts/spectra_{kind}.slurm').read_text()
        assert f'#SBATCH --cpus-per-task={cores}' in text and '#SBATCH --ntasks=1' in text and '#SBATCH --nodes=1' in text
        assert ('#SBATCH --gres=gpu:1' in text)==(kind=='gpu')
        subprocess.run(['bash',str(repo/f'scripts/spectra_{kind}.slurm')],cwd=repo,env=dict(base,SLURM_CPUS_PER_TASK=str(cores)),check=True,capture_output=True,text=True)
        row=json.loads(record.read_text());argv=row['argv']
        assert argv[argv.index('--device')+1]==device
        assert argv[argv.index('--cpu-threads')+1]==str(cores)
        assert argv[argv.index('--config')+1]=='configs/preparation_20260921/production/reference.json'
        assert argv[:1]==['--cpu-bind=cores']
        result.append({'kind':kind,'cores':cores,**row,'passed':True})
    failed=subprocess.run(['bash',str(repo/'scripts/spectra_gpu.slurm')],cwd=repo,env=dict(base,SLURM_NTASKS='2'),capture_output=True,text=True)
    assert failed.returncode==2 and 'one process' in failed.stderr
    subprocess.run(['bash',str(repo/'scripts/submit_spectra.sh'),'gpu','--array=0-15%4'],cwd=repo,env=base,check=True,capture_output=True,text=True)
    submit=json.loads(record.read_text());assert submit['argv']==['--array=0-15%4','scripts/spectra_gpu.slurm']
report={'shell_syntax_files':len(files),'dispatch':result,'multiple_MPI_tasks_rejected':True,'submit':submit,
        'scope':'Shell syntax and mocked dispatch contract; no actual Slurm controller or cluster allocation exercised.'}
Path('/playpen1/shiqiu/laqphysics-data/iteration20260921/slurm_contract.json').write_text(json.dumps(report,indent=2)+'\n')
print('Slurm script contract checks passed; no actual job submitted.')
