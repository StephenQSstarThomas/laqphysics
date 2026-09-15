import numpy as np
import pytest
torch=pytest.importorskip('torch')
from types import SimpleNamespace
from fedvr import make_grid
from surface3d import Ionic
from ionic_tdse import IonGPU,run

def test_one_electron_velocity_operator_and_scalar_A2():
    grid=make_grid([0,.5,1,2,4,8],4,tail=8,ecs_angle=.4)
    dummy=SimpleNamespace(grid=grid,n=len(grid.r),r=grid.r,cut=np.ones(len(grid.r)),
                          ch=[((l,m),(0,0)) for l in range(3) for m in range(-l,l+1)])
    ion=Ionic(dummy);op=IonGPU(ion,'cpu');rng=np.random.default_rng(132)
    x=rng.normal(size=op.size)+1j*rng.normal(size=op.size);field=[.2,-.3,.1]
    np.testing.assert_allclose(op.host(op.apply(op.tensor(x),field)),ion.matrix(field)@x,rtol=5e-14,atol=2e-11)
    np.testing.assert_allclose(op.host(op.apply(op.tensor(x),field)+op.apply(op.tensor(x),-np.array(field))-2*op.apply(op.tensor(x),[0,0,0])),np.dot(field,field)*x,atol=2e-11)

def test_ionic_checkpoint_resume_and_config_guard(tmp_path):
    config={'radial':{'edges':[0,.5,1,2,4,8],'order':4,'tail':8,'ecs_angle':.4},
            'lmax':2,'initial':[2,1,1],'dt':.02,'checkpoint_every':2,
            'pulses':[{'pulse':{'omega':1.5,'cycles':2,'field':.01},'polarization':'sigma-'}]}
    run(config,tmp_path/'full','cpu',max_steps=4)
    run(config,tmp_path/'split','cpu',max_steps=2)
    run(config,tmp_path/'split','cpu',resume=True,max_steps=2)
    np.testing.assert_allclose(np.load(tmp_path/'full/checkpoint.npz')['psi'],np.load(tmp_path/'split/checkpoint.npz')['psi'],atol=2e-14,rtol=2e-14)
    with pytest.raises(ValueError,match='config changed'):
        run({**config,'dt':.01},tmp_path/'split','cpu',resume=True,max_steps=1)
