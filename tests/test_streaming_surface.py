import json
from pathlib import Path
import numpy as np
import pytest
from run3d import setup,run,vector_function
from surface_projection import project,integrate
from surface_storage import ShardedArray
from streaming_surface import OnlineSurface

def configuration(mode='spectrum',gauge=None):
    c={'radial':{'edges':[0,1,2,4,6,8],'order':3,'tail':6,'ecs_angle':.3},
       'lmax':1,'M':None,'total_Lmax':2,'angular_representation':'bipolar',
       'cutoff_radii':[1,2],'surface':5.,'pulses':[{'pulse':{'omega':5.,'cycles':2,'field':.15},'polarization':'sigma+'}],
       'dt':.12,'post_time':.2,'checkpoint_every':5,'ground_tolerance':1e-10,
       'time_integrator':'cf4-pade','ionic_propagator':'cf4','linear_tolerance':1e-11,
       'ionic_channels':[[1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]],
       'spectrum_energy':[.2,1.2,7],'spectrum':{'theta_points':4,'phi_points':4},
       'storage':{'mode':mode,'max_file_bytes':500000,'ionic_block_frames':4,'ionic_device':'lu'}}
    if gauge:c['gauge']=gauge
    return c

def test_shards_cross_boundaries_and_reopen(tmp_path):
    shape=(23,17,7);expected=np.arange(np.prod(shape)).reshape(shape)*(1+2j)
    a=ShardedArray(tmp_path/'a',shape,mode='w+',max_file_bytes=12000)
    for i in reversed(range(len(a))):a[i]=expected[i]
    a.close();b=ShardedArray(tmp_path/'a')
    np.testing.assert_array_equal(b[:],expected)
    assert len(list((tmp_path/'a').glob('part*.npy')))>1
    assert all(p.stat().st_size<=12000 for p in (tmp_path/'a').glob('part*.npy'))

@pytest.mark.parametrize('coupled',[False,True,'linear_dense'])
def test_online_random_flux_matches_existing_backward_extraction(coupled,tmp_path):
    c=configuration('projected');c['storage']['max_file_bytes']=100000
    if not coupled:c.pop('angular_representation');c.pop('total_Lmax')
    elif coupled=='linear_dense':
        c.pop('angular_representation');c['M']=0;c['pulses'][0]['polarization']='z'
    _,h=setup(c);h.prepare_surface(5.);A,_=vector_function(c);times=np.linspace(0,2*np.pi*2/5,17);dt=times[1]
    online=OnlineSurface(h,c,tmp_path,times,A);assert online.prepare()
    rng=np.random.default_rng(424);f=rng.normal(size=(len(times),h.n,len(h.surface_indices),h.nc))+1j*rng.normal(size=(len(times),h.n,len(h.surface_indices),h.nc))
    for i in range(len(times)):online.record_frame(i,f[i])
    base=getattr(h,'uncoupled',h);base.prepare_surface(5.)
    q,lm,info=project(base,f,times,A,c['ionic_channels'],dt,propagator='cf4',angular_transform=getattr(h,'transform',None))
    ref,_=integrate(q,lm,base.r[base.surface_indices].real,base.grid.weights[base.surface_indices].real,times,A,
                    online.accumulator.energy,(online.accumulator.theta,online.accumulator.phi),dt)
    np.testing.assert_allclose(online.accumulator.amplitude,ref,rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(online.projected[:],q,rtol=2e-11,atol=2e-11)
    assert not (tmp_path/'flux.npy').exists()

@pytest.mark.parametrize('gauge',[None,{'type':'mixed','inner':.5,'outer':1.5}])
def test_driven_tdse_online_restart_and_offline_agree(gauge,tmp_path):
    c=configuration(gauge=gauge);options={'backend':'torch','device':'cpu','preconditioner_precision':'complex64','ground_cache':tmp_path/'ground'}
    full=tmp_path/'full';split=tmp_path/'split';raw=tmp_path/'raw'
    run(c,full,**options);run(c,split,max_steps=7,**options);run(c,split,resume=True,**options)
    a=np.load(full/'spectrum.npz');b=np.load(split/'spectrum.npz')
    np.testing.assert_allclose(a['amplitudes'],b['amplitudes'],rtol=2e-11,atol=2e-11)
    rc=dict(c,storage=dict(c['storage'],mode='raw_shards'));m=run(rc,raw,**options)
    _,h=setup(rc);h.prepare_surface(5.);base=getattr(h,'uncoupled',h);base.prepare_surface(5.);A,_=vector_function(rc)
    flux=ShardedArray(raw/'flux_shards');times=np.arange(len(flux))*m['dt'];factory=None;field=None
    if gauge:
        from mixed_gauge import MixedIonic,parameters
        from run3d import electric_function
        E=electric_function(rc);factory=lambda h,mag:MixedIonic(h,mag,.5,1.5);field=lambda t:parameters(A,E,t)
    q,lm,_=project(base,flux,times,A,c['ionic_channels'],m['dt'],propagator='cf4',angular_transform=h.transform,ion_factory=factory,ionic_field=field)
    expected,_=integrate(q,lm,base.r[base.surface_indices].real,base.grid.weights[base.surface_indices].real,times,A,a['energy'],(a['theta'],a['phi']),m['dt'])
    np.testing.assert_allclose(a['amplitudes'],expected,rtol=2e-10,atol=2e-11)
    assert all(p.stat().st_size<=500000 for p in tmp_path.rglob('*.npz') if 'ground' not in p.parts)

def test_ionic_preparation_interrupt_resumes_from_endpoint(tmp_path):
    c=configuration();_,h=setup(c);h.prepare_surface(5.);A,_=vector_function(c);t=np.linspace(0,2*np.pi*2/5,17)
    a=OnlineSurface(h,c,tmp_path,t,A)
    assert not a.prepare(lambda:True);assert 0<a.lowest<len(t)-1
    b=OnlineSurface(h,c,tmp_path,t,A,resume=True);assert b.prepare()
    fresh=OnlineSurface(h,c,tmp_path/'fresh',t,A);assert fresh.prepare()
    for i in range(len(t)):np.testing.assert_allclose(b.chi_at(i),fresh.chi_at(i),atol=2e-12)

def test_uncommitted_accumulator_does_not_replace_checkpoint_bank(tmp_path):
    c=configuration();options={'backend':'torch','device':'cpu','ground_cache':tmp_path/'ground'}
    run(c,tmp_path/'split',max_steps=5,**options)
    p=tmp_path/'split';saved=np.load(p/'checkpoint.npz');bank=int(saved['surface_generation'])
    # Simulate a crash after staging the alternate bank but before wave checkpoint commit.
    alternate=p/'surface_online'/f'accumulator_{1-bank}.npz';alternate.write_bytes(b'incomplete, uncommitted output')
    run(c,p,resume=True,**options);assert json.loads((p/'run.json').read_text())['complete']

def test_projected_cli_reintegrates_same_complex_spectrum(tmp_path):
    import importlib.util
    from argparse import Namespace
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('projected_cli_test',root/'scripts/extract_projected.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c=configuration('projected');run(c,tmp_path/'run',backend='torch',device='cpu',ground_cache=tmp_path/'ground')
    args=Namespace(name='reintegrated.npz',energy=None,theta=None,phi=None,angular_rule=None,post_time=None,time_stride=1,reproject=False,ionic_device=None)
    module.execute(tmp_path/'run',args)
    np.testing.assert_allclose(np.load(tmp_path/'run/spectrum.npz')['amplitudes'],np.load(tmp_path/'run/reintegrated.npz')['amplitudes'],atol=2e-12,rtol=2e-12)
    # Re-grid in angle/energy and coarsen time with the new snapshot-aware CLI;
    # compare against the independent reverse-order integral, including phase.
    import os,subprocess,sys
    args.name='independent_refinement.npz';args.theta=6;args.phi=8;args.time_stride=2;args.energy=[.2,1.2,9]
    module.execute(tmp_path/'run',args)
    subprocess.run([sys.executable,str(root/'scripts/refine_online_spectrum.py'),'--out',str(tmp_path/'run'),
        '--name','gpu_refinement.npz','--device',os.environ.get('HELIUM_TEST_DEVICE','cpu'),
        '--theta','6','--phi','8','--time-stride','2','--energy','.2','1.2','9'],check=True)
    with np.load(tmp_path/'run/gpu_refinement.npz') as actual,np.load(tmp_path/'run/independent_refinement.npz') as reference:
        np.testing.assert_allclose(actual['amplitudes'].reshape(5,9,6,8)[:,:,::-1,:].reshape(5,9,48),reference['amplitudes'],rtol=3e-12,atol=3e-12)

@pytest.mark.parametrize('gauge',[None,{'type':'mixed','inner':.5,'outer':1.5}])
def test_parallel_CPU_replay_matches_serial_spectrum(gauge,tmp_path):
    from copy import deepcopy
    c=configuration(gauge=gauge);parallel=deepcopy(c);parallel['storage']['replay_workers']=2
    options={'backend':'torch','device':'cpu','ground_cache':tmp_path/'ground'}
    run(c,tmp_path/'serial',**options);run(parallel,tmp_path/'parallel',max_steps=7,**options)
    run(parallel,tmp_path/'parallel',resume=True,**options)
    np.testing.assert_allclose(np.load(tmp_path/'serial/spectrum.npz')['amplitudes'],
                               np.load(tmp_path/'parallel/spectrum.npz')['amplitudes'],rtol=2e-11,atol=2e-11)

def test_gpu_accumulator_formula_and_checkpoint_match_numpy(tmp_path):
    import os,torch
    from streaming_surface import TorchVolkovAccumulator
    c=configuration();_,h=setup(c);h.prepare_surface(5.);A,_=vector_function(c);t=np.linspace(0,2*np.pi*2/5,17)
    online=OnlineSurface(h,c,tmp_path,t,A)
    device=os.environ.get('HELIUM_TEST_DEVICE','cpu')
    acc=TorchVolkovAccumulator(online.contract,c,A,500000,device)
    rng=np.random.default_rng(91);q=rng.normal(size=(len(t),*online.contract.shape))+1j*rng.normal(size=(len(t),*online.contract.shape))
    for i in range(8):acc.add(i,t[i],q[i])
    saved={k:np.array(v,copy=True) for k,v in acc.payload().items()}
    resumed=TorchVolkovAccumulator(online.contract,c,A,500000,device);resumed.restore(saved)
    for i in range(8,len(t)):resumed.add(i,t[i],q[i])
    base=online.contract.h
    expected,_=integrate(q,online.contract.outerstates,base.r[base.surface_indices].real,base.grid.weights[base.surface_indices].real,
                         t,A,acc.energy,(acc.theta,acc.phi),t[1])
    assert resumed.amplitude.dtype==torch.complex128
    np.testing.assert_allclose(resumed.payload()['amplitude'],expected,atol=3e-12,rtol=3e-12)
    online.close()
