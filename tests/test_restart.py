import json
import os
import signal
import numpy as np
import pytest
from run3d import run

def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    c={'radial':{'edges':[0,1,2,4,6,8],'order':3,'tail':6,'ecs_angle':.3},
       'lmax':1,'M':0,'cutoff_radii':[1,2],'surface':5.,
       'pulses':[{'pulse':{'omega':1.5,'cycles':2,'field':.01},'polarization':'z'}],
       'dt':.02,'post_time':0.,'surface_stride':1,'checkpoint_every':2,
       'krylov_tolerance':1e-11,'ground_tolerance':1e-10,'spectrum_energy':[.3,.9,21]}
    uninterrupted=tmp_path/'full';restarted=tmp_path/'restart'
    run(c,uninterrupted,max_steps=4)
    run(c,restarted,max_steps=2);run(c,restarted,resume=True,max_steps=2)
    a=np.load(uninterrupted/'checkpoint.npz');b=np.load(restarted/'checkpoint.npz')
    np.testing.assert_allclose(a['psi'],b['psi'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.load(uninterrupted/'flux.npy')[:5],np.load(restarted/'flux.npy')[:5],atol=1e-12)
    changed=dict(c);changed['dt']=.01
    import pytest
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM)}
    with pytest.raises(ValueError,match='config changed'):run(changed,restarted,resume=True,max_steps=2)
    assert all(signal.getsignal(s)==handler for s,handler in handlers.items())

def test_prefix_extraction_uses_only_flushed_history(tmp_path):
    from run3d import extract_run
    import pytest
    config={'radial':{'edges':[0,1,2,4,6,8],'order':3,'tail':6,'ecs_angle':.3},
            'lmax':1,'M':0,'cutoff_radii':[1,2],'surface':5.,
            'pulses':[{'pulse':{'omega':1.5,'cycles':2,'field':0.},'polarization':'z'}],
            'dt':.02,'post_time':0.,'checkpoint_every':2,'spectrum_energy':[.3,.9,9]}
    meta=run(config,tmp_path,max_steps=4);end=2*meta['dt']
    extract_run(tmp_path,'prefix_a.npz',stop_time=end)
    flux=np.load(tmp_path/'flux.npy',mmap_mode='r+');flux[3:]=1e20*(1+1j);flux.flush()
    # A live first invocation has config/checkpoint files before run.json exists.
    (tmp_path/'run.json').unlink()
    extract_run(tmp_path,'prefix_b.npz',stop_time=end)
    np.testing.assert_array_equal(np.load(tmp_path/'prefix_a.npz')['amplitudes'],np.load(tmp_path/'prefix_b.npz')['amplitudes'])
    with pytest.raises(ValueError,match='flushed checkpoint'):
        extract_run(tmp_path,'invalid.npz',stop_time=5*meta['dt'])
    assert not (tmp_path/'spectrum.npz').exists()

def test_equal_case_names_in_distinct_campaigns_have_distinct_scratch(tmp_path,monkeypatch):
    monkeypatch.setenv('HELIUM_SURFACE_ROOT',str(tmp_path/'scratch'))
    c={'radial':{'edges':[0,1,2,4,6,8],'order':3,'tail':6,'ecs_angle':.3},
       'lmax':1,'M':0,'cutoff_radii':[1,2],'surface':5.,
       'pulses':[{'pulse':{'omega':1.5,'cycles':2,'field':.01},'polarization':'z'}],
       'dt':.02,'post_time':0.,'checkpoint_every':2}
    first=tmp_path/'campaign_a/case';second=tmp_path/'campaign_b/case'
    run(c,first,max_steps=2);saved=np.load(first/'flux.npy')[:3].copy()
    run(c,second,max_steps=2)
    assert (first/'flux.npy').resolve()!=(second/'flux.npy').resolve()
    np.testing.assert_array_equal(np.load(first/'flux.npy')[:3],saved)

@pytest.mark.parametrize('gauge',[{'type':'velocity'},{'type':'mixed','inner':.5,'outer':1.5}])
def test_flexible_cf4_circular_resume_matches_continuous(gauge,tmp_path):
    # The interruption occurs during the driven pulse, with all magnetic
    # sectors and the actual approximate-preconditioner production solver.
    config={'radial':{'edges':[0,1,2,4,6,8],'order':3,'tail':6,'ecs_angle':.3},
            'lmax':1,'M':None,'angular_representation':'bipolar','total_Lmax':2,
            'cutoff_radii':[1,2],'surface':5.,'gauge':gauge,
            'pulses':[{'pulse':{'omega':5.,'cycles':2,'field':.2},'polarization':'sigma+'}],
            'dt':.15,'post_time':0.,'surface_stride':1,'checkpoint_every':1,
            'time_integrator':'cf4-pade','ionic_propagator':'cf4',
            'linear_tolerance':1e-11,'ground_tolerance':1e-10,'coulomb_contraction':'blocks'}
    options={'backend':'torch','device':os.environ.get('HELIUM_TEST_DEVICE','cpu'),
             'preconditioner_precision':'complex64','ground_cache':tmp_path/'ground_cache'}
    full=tmp_path/'full';split=tmp_path/'split'
    run(config,full,max_steps=12,**options)
    run(config,split,max_steps=5,**options);run(config,split,max_steps=7,resume=True,**options)
    a=np.load(full/'checkpoint.npz');b=np.load(split/'checkpoint.npz')
    assert int(a['step'])==int(b['step'])==12
    np.testing.assert_allclose(a['psi'],b['psi'],rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(np.load(full/'flux.npy')[:13],np.load(split/'flux.npy')[:13],rtol=3e-12,atol=3e-12)
