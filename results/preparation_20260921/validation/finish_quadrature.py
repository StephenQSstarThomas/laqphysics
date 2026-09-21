import json,subprocess,sys,time
from pathlib import Path
r=Path('/playpen1/shiqiu/laqphysics-data/iteration20260921');case=r/'runs/selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837'
snapshot=Path((r/'accelerated_snapshot.txt').read_text().strip());repo=Path('/playpen/shiqiu/task-jiukongdian')
while True:
    status=json.loads((case/'STATUS.json').read_text())
    if status['state']=='failed':raise RuntimeError('Fine physical run failed; do not process an incomplete result')
    if status['state']=='complete':break
    time.sleep(30)
check=json.loads((r/'fine_grid_execution_regression.json').read_text());assert check['passed']
command=[sys.executable,str(repo/'scripts/refine_online_spectrum.py'),'--out',str(case),'--snapshot',str(snapshot),'--device','cuda:1','--cpu-threads','2']
for name,options in [('spectrum_angles.npz',['--theta','24','--phi','32']),('spectrum_energy.npz',['--energy','.18','.75','2281']),('spectrum_stride2.npz',['--time-stride','2'])]:
    subprocess.run(command+['--name',name]+options,check=True)
sys.path.insert(0,str(repo/'python'));from convergence import compare
import numpy as np
criteria={'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,'gate_total':True,'gate_channels':[[1,0,0],[2,1,1]]}
base=np.load(case/'spectrum.npz');rows=[]
for name in ['spectrum_angles.npz','spectrum_energy.npz','spectrum_stride2.npz']:
    with np.load(case/name) as data:
        windows=[compare(base,data,criteria,w) for w in [[.18,.42],[.52,.68]]]
    rows.append({'file':name,'passed':all(w['status']=='passed' for w in windows),'windows':windows})
report={'configuration_signature':json.loads((case/'run.json').read_text())['signature'],'criteria':criteria,'comparisons':rows,
        'all_passed':all(r['passed'] for r in rows),'scope':'Extraction quadrature at the fine local model, not spatial/boundary/ionic production convergence.'}
(r/'fine_spectral_quadrature.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
