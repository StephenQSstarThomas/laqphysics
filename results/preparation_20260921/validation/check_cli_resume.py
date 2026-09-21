import json,subprocess,sys
from pathlib import Path
repo=Path('/playpen/shiqiu/task-jiukongdian');r=Path('/playpen1/shiqiu/laqphysics-data/iteration20260921')
sys.path.insert(0,str(repo/'tests'));sys.path.insert(0,str(repo/'python'))
from test_streaming_surface import configuration
c=configuration('projected');input=r/'cli_resume_smoke.json';input.write_text(json.dumps(c,indent=2)+'\n')
out=r/'cli_smoke';cmd=[sys.executable,str(repo/'scripts/simulate_spectrum.py'),'--config',str(input),'--out-root',str(out),'--device','cuda:1','--cpu-threads','2','--memory-fraction','.25']
first=subprocess.run(cmd+['--max-steps','7']).returncode
assert first==95
case=next(out.glob('cli_resume_smoke__*'));status=json.loads((case/'STATUS.json').read_text())
assert status['state']=='checkpointed' and not status['complete_spectrum']
assert not (case/'spectrum.npz').exists()
subprocess.run(cmd,check=True)
status=json.loads((case/'STATUS.json').read_text());obs=json.loads((case/'observables.json').read_text());assert status['state']=='complete' and status['complete_spectrum']
assert all((case/path).is_file() for path in obs['artifacts'].values())
files=[p for p in case.rglob('*') if p.is_file()];largest=max(p.stat().st_size for p in files)
assert largest<=c['storage']['max_file_bytes'] and not (case/'flux.npy').exists()
record={'passed':True,'first_exit':first,'resumed_exit':0,'complete_status':status,'largest_actual_file_bytes':largest,
        'file_cap_bytes':c['storage']['max_file_bytes'],'no_raw_flux':True,'all_named_artifacts_present':True,
        'scope':'Tiny circular CUDA command-line integration test: checkpoint, same-command resume, mandatory SI data/PNG/PDF/CSV and actual-size audit.',
        'case':str(case)}
(r/'cli_resume_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
