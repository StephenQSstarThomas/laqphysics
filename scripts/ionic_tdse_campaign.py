#!/usr/bin/env python3
import argparse,json,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ionic_tdse import run
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
p.add_argument('--resume',action='store_true');p.add_argument('--auto-resume',action='store_true');p.add_argument('--max-steps',type=int);a=p.parse_args()
config=json.loads(Path(a.config).read_text());out=Path(a.out)
if a.auto_resume:
    if (out/'run.json').exists():
        meta=json.loads((out/'run.json').read_text())
        if meta['config']!=config:raise ValueError('config changed; choose a new output directory')
        if meta.get('complete'):print('Already complete:',out);raise SystemExit(0)
    a.resume=(out/'checkpoint.npz').exists()
run(config,out,a.device,a.resume,a.max_steps)
