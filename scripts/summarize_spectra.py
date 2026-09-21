#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from spectrum_report import update_index
p=argparse.ArgumentParser();p.add_argument('--out-root',required=True);a=p.parse_args()
rows=update_index(a.out_root)
for row in rows:print(row['state'],row['run_id'],'peak ratio:',row['peak_ratio'],'old 2p+1:',row['old_2p_plus1_fraction'])
print('INDEX:',Path(a.out_root)/'INDEX.md')
