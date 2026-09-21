#!/usr/bin/env python3
"""Re-integrate saved Q on CPU/GPU using the originating numerical snapshot."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--snapshot',type=Path)
    p.add_argument('--name',required=True);p.add_argument('--device',default='cpu');p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--theta',type=int);p.add_argument('--phi',type=int);p.add_argument('--energy',type=float,nargs=3)
    p.add_argument('--time-stride',type=int,default=1)
    a=p.parse_args();snapshot=(a.snapshot or Path(__file__).resolve().parents[1]).resolve();sys.path.insert(0,str(snapshot/'python'))
    from output_lock import exclusive_output
    @exclusive_output('.projection.lock')
    def execute(out):
        from streaming_surface import VolkovAccumulator,TorchVolkovAccumulator,numerical_signature
        from surface_storage import ShardedArray,atomic_json,file_limit
        from run3d import vector_function
        import torch
        torch.set_num_threads(a.cpu_threads)
        out=Path(out);meta=json.loads((out/'run.json').read_text());config=json.loads(json.dumps(meta['config']))
        if not meta['complete'] or config.get('storage',{}).get('mode')!='projected':raise ValueError('requires a completed projected-mode propagation')
        if Path(a.name).name!=a.name or not a.name.endswith('.npz') or a.name=='spectrum.npz':raise ValueError('use a distinct NPZ basename for the quadrature refinement')
        if a.time_stride<1:raise ValueError('time stride must be positive')
        layout=json.loads((out/'surface_online/layout.json').read_text());done=json.loads((out/'surface_online/complete.json').read_text())
        if not done['complete'] or layout['recipe']['configuration']!=meta['signature'] or layout['recipe']['kernel']!=numerical_signature():
            raise ValueError('projection/configuration/kernel mismatch; use the original numerical snapshot')
        if a.theta:config.setdefault('spectrum',{})['theta_points']=a.theta
        if a.phi:config.setdefault('spectrum',{})['phi_points']=a.phi
        if a.energy:
            lo,hi,count=a.energy
            if int(count)!=count or count<3 or hi<=lo:raise ValueError('invalid energy grid')
            config.setdefault('spectrum',{}).pop('energy_segments',None);config['spectrum_energy']=[lo,hi,int(count)]
        r=np.array(layout['r']);weights=np.array(layout['weights']);dt=layout['times']['dt']
        contract=SimpleNamespace(h=SimpleNamespace(r=r,grid=SimpleNamespace(weights=weights),surface_indices=np.arange(len(r))),
                                labels=layout['labels'],outerstates=[tuple(x) for x in layout['outer_states']])
        A,_=vector_function(config);limit=file_limit(config['storage'].get('max_file_bytes',4_000_000_000))
        accumulator=(TorchVolkovAccumulator(contract,config,A,limit,a.device) if a.device.startswith('cuda') else VolkovAccumulator(contract,config,A,limit))
        history=ShardedArray(out/'surface_online/projected');indices=np.arange(0,len(history),a.time_stride)
        if indices[-1]!=len(history)-1:indices=np.r_[indices,len(history)-1]
        if len(indices)<2:raise ValueError('insufficient integration times')
        start=time.perf_counter()
        for sample,index in enumerate(indices):accumulator.add(sample,index*dt,history[index])
        accumulator.write(out/a.name,meta['signature'],limit);history.close()
        atomic_json(out/(a.name+'.json'),{'configuration_signature':meta['signature'],'kernel':layout['recipe']['kernel'],
            'snapshot':str(snapshot),'device':a.device,'energy_points':len(accumulator.energy),'angular_points':len(accumulator.theta),
            'time_stride':a.time_stride,'flux_integral_end_time':accumulator.time,'projection_final_time':(layout['times']['count']-1)*dt,
            'seconds':time.perf_counter()-start,'propagation_rerun':False,'scope':'Saved ionic projection re-integrated coherently; ionic terminal time is unchanged.'})
        print('Wrote',out/a.name,flush=True)
    execute(a.out)

if __name__=='__main__':main()
