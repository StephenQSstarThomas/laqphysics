#!/usr/bin/env python3
"""One input file -> resumable propagation -> mandatory named SI data and figures."""
import argparse,hashlib,json,os,re,sys,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from surface_storage import atomic_json
from campaign_resources import estimate
from spectrum_report import publish,pulse_rows,update_index,parameter_tag
from output_lock import exclusive_output

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out-root',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--memory-fraction',type=float,default=.8)
    p.add_argument('--cpu-threads',type=int,default=int(os.environ.get('HELIUM_TORCH_THREADS','2')))
    p.add_argument('--tag',default='');p.add_argument('--estimate-only',action='store_true');p.add_argument('--max-steps',type=int)
    p.add_argument('--storage',choices=['spectrum','projected','raw_shards'])
    a=p.parse_args();input_path=Path(a.config).resolve();config=json.loads(input_path.read_text())
    template=json.loads(json.dumps(config))
    config.setdefault('storage',{'mode':'spectrum','max_file_bytes':4_000_000_000,'ionic_block_frames':128,'ionic_device':'cpu'})
    config['storage'].setdefault('accumulator_device','auto')
    if a.storage:config['storage']['mode']=a.storage
    config.setdefault('time_integrator','cf4-pade');config.setdefault('ionic_propagator','cf4')
    config.setdefault('spectrum',{'theta_points':24 if config.get('M',0) is None else 32,'phi_points':32 if config.get('M',0) is None else 1})
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    name=input_path.stem+('__'+a.tag if a.tag else '')
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+',name):raise ValueError('config stem/tag must use letters, digits, _, -, + or .')
    run_id=name+'__'+signature[:10];out=Path(a.out_root).expanduser().resolve()/run_id
    if len(run_id+'__'+parameter_tag(config)+'__SI')>225:raise ValueError('case name/tag is too long for descriptive result filenames')
    resource=estimate(config)
    if a.estimate_only:
        print(json.dumps({'run_id':run_id,'output':str(out),'input':str(input_path),'resources':resource,'pulses':pulse_rows(config)},indent=2));return 0
    return execute_case(out,a,config,template,input_path,signature,run_id,resource)

@exclusive_output('.case.lock')
def execute_case(out,a,config,template,input_path,signature,run_id,resource):
    out.mkdir(parents=True,exist_ok=True)
    # Finished data also retain their numerical provenance. A new code version
    # may re-render them, but must not silently present them as a fresh run.
    if (out/'run.json').exists():
        existing=json.loads((out/'run.json').read_text())
        if existing.get('complete') and existing['config'].get('storage',{}).get('mode') in ('spectrum','projected'):
            from streaming_surface import numerical_signature
            layout=json.loads((out/'surface_online/layout.json').read_text())
            if layout['recipe']['kernel']!=numerical_signature():
                raise ValueError('completed result uses another numerical kernel; use its original snapshot, a fresh --tag for a new calculation, or render_spectra.py for figures only')
    if (out/'input.json').exists() and json.loads((out/'input.json').read_text())!=config:raise ValueError('output input mismatch')
    atomic_json(out/'input.json',config);atomic_json(out/'input_template.json',template);atomic_json(out/'resource_estimate.json',resource)
    import datetime
    attempts=json.loads((out/'attempts.json').read_text()) if (out/'attempts.json').exists() else []
    attempts.append({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
                     'device':a.device,'cpu_threads':a.cpu_threads,
                     'numerical_snapshot':json.loads((root/'numerical_snapshot.json').read_text())['id'] if (root/'numerical_snapshot.json').exists() else None})
    atomic_json(out/'attempts.json',attempts)
    import shutil
    # Budget checkpoint replacement and the retained artifacts, not just raw flux.
    free=shutil.disk_usage(out).free
    if resource['temporary_write_peak_estimate_bytes']>free*.8 and not (out/'checkpoint.npz').exists():
        raise ValueError('insufficient scratch space for projected/spectrum state and atomic checkpoints')
    (out/'RUN.md').write_text(f'# {run_id}\n\n输入：input.json\n\n状态：运行中，尚无完整单电离谱。\n\n资源估算：resource_estimate.json\n')
    print('RUN_ID',run_id,'\nINPUT',input_path,'\nOUTPUT',out,flush=True)
    try:
        import torch
        replay_workers=int(config['storage'].get('replay_workers',0))
        if a.cpu_threads<=replay_workers:raise ValueError('CPU budget must leave at least one parent thread besides replay workers')
        torch.set_num_threads(a.cpu_threads-replay_workers)
        if a.device.startswith('cuda'):
            budget=torch.cuda.get_device_properties(a.device).total_memory*a.memory_fraction
            free_gpu,_=torch.cuda.mem_get_info(a.device)
            if resource['FGMRES_Q_Z_bytes']+resource['spectral_accumulator_pair_bytes']>min(budget,free_gpu):
                raise ValueError('FGMRES basis alone exceeds available/requested GPU memory; choose a larger GPU or a smaller numerical basis')
            torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device=a.device)
        from run3d import run,extract_run
        meta=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
        if meta and meta.get('signature')!=signature:raise ValueError('configuration changed')
        atomic_json(out/'STATUS.json',{'state':'running','run_id':run_id,'input':str(input_path),'output':str(out),
                    'requested_device':a.device,'cpu_budget':a.cpu_threads,'torch_cpu_threads':torch.get_num_threads(),'replay_workers':replay_workers,'resources':resource})
        update_index(out.parent)
        if not meta.get('complete'):
            meta=run(config,out,resume=(out/'checkpoint.npz').exists(),max_steps=a.max_steps,backend='torch',device=a.device,
                     ground_cache=Path(a.out_root)/'ground_cache',preconditioner_precision='complex64')
        if not meta.get('complete'):
            atomic_json(out/'STATUS.json',{'state':'checkpointed','phase':meta.get('phase'),'complete_spectrum':False,
                        'run_id':run_id,'completed_steps':meta['completed_steps'],'nsteps':meta['nsteps'],
                        'resume':'Repeat the same command with the same input and output root; this is not a completed SI spectrum.'})
            update_index(out.parent)
            print('CHECKPOINTED; repeat the same command to finish the mandatory spectrum.',flush=True);return 95
        if not (out/'spectrum.npz').exists():extract_run(out)
        result=publish(out,run_id)
        oversized=[str(f) for f in out.rglob('*') if f.is_file() and f.stat().st_size>resource['max_file_bytes']]
        if oversized:raise ValueError(f'output exceeded file cap: {oversized}')
        atomic_json(out/'STATUS.json',{'state':'complete','complete_spectrum':True,'run_id':run_id,'input':'input.json',
                    'spectrum':result['artifacts']['spectrum'],'figure':result['artifacts']['figure'],
                    'conditional_figure':result['artifacts']['conditional_figure'],
                    'observables':'observables.json','design_targets_passed':result['target_passed'],
                    'numerical_convergence_certified':False})
        files=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file() and f.name!='output_manifest.json']
        atomic_json(out/'output_manifest.json',{'run_id':run_id,'configuration_signature':signature,'files':files,
                    'largest_actual_file_bytes':max(f['bytes'] for f in files),'max_file_bytes':resource['max_file_bytes']})
        update_index(out.parent)
        print('COMPLETE SI SPECTRUM',out/result['artifacts']['figure'],flush=True);return 0
    except BaseException as error:
        atomic_json(out/'STATUS.json',{'state':'failed','run_id':run_id,'error':str(error),'complete_spectrum':False})
        update_index(out.parent)
        raise

if __name__=='__main__':raise SystemExit(main())
