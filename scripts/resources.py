#!/usr/bin/env python3
from pathlib import Path
import sys,json,subprocess,platform,os,shutil
import numpy as np,scipy,matplotlib,sympy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from fedvr import make_grid
from angular import channels
from pulses import Pulse
root=Path(__file__).resolve().parents[1];rows=[]
for file in sorted((root/'configs').glob('3d*.json')):
    c=json.loads(file.read_text());g=make_grid(**c['radial']);n=len(g.r);nc=len(channels(c['lmax'],c.get('M',0)))
    theta=(g.r.real>c['surface']).astype(float);T=g.kinetic.toarray()
    nsurf=int(np.sum(np.max(abs(T*(theta[None,:]-theta[:,None])),axis=1)>1e-14))
    duration=max(Pulse(**a['pulse']).duration+a['pulse'].get('start',0) for a in c['pulses'])+c.get('post_time',60)
    steps=int(np.ceil(duration/c['dt']));stride=c.get('surface_stride',1);steps+=(-steps)%stride
    state=n*n*nc*16;kdim=c.get('krylov_dimension',48)
    rows.append({'config':file.name,'radial_points':n,'angular_channels':nc,'steps':steps,
        'wavefunction_mib':state/2**20,'krylov_basis_gib':state*(kdim+1)/2**30,
        'surface_history_gib':(steps//stride+1)*n*nsurf*nc*16/2**30,
        'note':'BLAS conjugate-transpose uses one Q buffer. Estimate excludes work arrays/operators; time must be benchmarked.'})
def command(args):
    try:return subprocess.run(args,check=False,text=True,capture_output=True).stdout.strip()
    except FileNotFoundError:return 'not installed'
result={'platform':platform.platform(),'affinity_cpus':len(os.sched_getaffinity(0)),
        'cpu':command(['lscpu']),'memory':command(['free','-h']),
        'gpu':command(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv']),
        'slurm_sbatch':shutil.which('sbatch'),'playpen_free_gib':shutil.disk_usage(root).free/2**30,
        'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'sympy':sympy.__version__},
        'compiler':command(['gfortran','--version']).splitlines()[0],'production_estimates':rows}
(root/'results/resources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
