#!/usr/bin/env python3
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tdse1d import run,extract
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--out',required=True);p.add_argument('--extract',action='store_true')
a=p.parse_args()
if not a.extract:run(json.loads(Path(a.config).read_text()),a.out)
extract(a.out)
