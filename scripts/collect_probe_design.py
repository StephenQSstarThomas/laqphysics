#!/usr/bin/env python3
"""Merge prepared-ion probe scans (probe_ion_design.py outputs) into one table.

Each argument is LABEL=DIR; DIR holds one subdirectory per probe case with the
He+ TDSE run.json/populations.npz. Writes JSON and a Markdown table of the final
populations that decide the probe design (2p+1 kept, 3p-1, the n=5 states and
the lost/ionized remainder)."""
import argparse,json
from pathlib import Path
import numpy as np

KEYS=[(2,1,1),(3,1,-1),(5,2,-2),(5,4,-2),(5,4,4)]

def main():
    p=argparse.ArgumentParser();p.add_argument('scans',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    rows=[]
    for item in a.scans:
        basis,folder=item.split('=',1)
        for case in sorted(Path(folder).iterdir()):
            if not (case/'run.json').exists():continue
            meta=json.loads((case/'run.json').read_text(encoding='utf-8'))
            if not meta.get('complete'):continue
            with np.load(case/'populations.npz') as d:labels=[tuple(map(int,x)) for x in d['labels']];prob=d['probabilities']
            pulse=meta['config']['pulses'][0]
            rows.append({'ion_basis':basis,'lmax':meta['config']['lmax'],'radial_edge':meta['config']['radial']['edges'][-1],
                         'cutoff_radii':meta['config'].get('cutoff_radii'),'omega':pulse['pulse']['omega'],'field':pulse['pulse']['field'],
                         'cycles':pulse['pulse']['cycles'],'polarization':pulse['polarization'],
                         'final':{str(list(k)):float(prob[labels.index(k)]) if k in labels else None for k in KEYS},
                         'lost_or_unrecorded':1-float(meta['bound_n_le_6_probability']),'maximum_linear_residual':meta['maximum_linear_residual'],
                         'dt':meta['dt'],'case':case.name})
    rows.sort(key=lambda r:(r['omega'],r['polarization'],r['field'],r['cycles'],r['lmax'],r['radial_edge'],r['ion_basis']))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'initial':'He+ 2p(+1)','populations':'final bound populations after the probe (A=0); lost = ionized or n>6',
        'cases':rows},indent=2)+'\n',encoding='utf-8')
    fmt=lambda x:'—' if x is None else (f'{x:.4f}' if x>.01 else f'{x:.1e}')
    lines=['| ω | 偏振 | F | 周期 | 离子基 | lmax | 保留 2p+1 | 3p-1 | 5d-2 | 5g-2 | 5g+4 | 电离/未记录 |','|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        f=r['final'];lines.append(f"| {r['omega']:.6g} | {r['polarization']} | {r['field']:g} | {r['cycles']:g} | {r['ion_basis']} | {r['lmax']} | "
            +' | '.join(fmt(f[str(list(k))]) for k in KEYS)+f" | {fmt(r['lost_or_unrecorded'])} |")
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()
