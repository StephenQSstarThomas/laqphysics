#!/usr/bin/env python3
"""Resume the focused follow-up campaign; each completed spectrum is reusable."""
import argparse,json,gc,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import run,extract_run

GROUPS={
 'bandwidth':['weak_24'],
 'circular':['circular_plus','circular_minus'],
 'convergence':['res_baseline_cf4','res_l6_L2_p4','res_l6_L2_p6','res_l8_L2_p6'],
 'pump-probe':['pump_only','pump_probe_plus','pump_probe_minus','pump_probe_linear','probe_only_plus','pump_probe_half_frequency']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,default='bandwidth');p.add_argument('--case',action='append')
 p.add_argument('--config-dir',default='configs/followup');p.add_argument('--out-root',default='results/followup')
 p.add_argument('--backend',choices=['torch','fortran'],default='torch');p.add_argument('--device',default='cuda:0')
 p.add_argument('--coulomb-contraction',choices=['sparse','blocks'])
 p.add_argument('--memory-fraction',type=float,default=.2);p.add_argument('--list',action='store_true');a=p.parse_args()
 names=a.case or GROUPS[a.group]
 if a.list:
  print('\n'.join(names));return
 if a.backend=='torch':
  import torch
  if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device=a.device)
 root=Path(a.out_root)
 for name in names:
  config=json.loads((Path(a.config_dir)/(name+'.json')).read_text());out=root/name
  meta=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
  expected=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
  if meta and meta.get('signature')!=expected:raise ValueError(f'{name}: configuration changed; use a new output directory')
  if not meta.get('complete',False):
   meta=run(config,out,resume=(out/'checkpoint.npz').exists(),backend=a.backend,device=a.device,ground_cache=root/'ground_cache',coulomb_contraction=a.coulomb_contraction)
  if not meta.get('complete',False):
   print('Checkpoint saved; resume this campaign to continue.',flush=True);return
  gc.collect()
  if a.backend=='torch' and a.device.startswith('cuda'):torch.cuda.empty_cache()
  if not (out/'spectrum.npz').exists():extract_run(out)
  if 'pump_reference_time' in config and not (out/'spectrum_pump_prefix.npz').exists():
   extract_run(out,'spectrum_pump_prefix.npz',stop_time=config['pump_reference_time'])
  print('COMPLETE',name,flush=True)
if __name__=='__main__':main()
