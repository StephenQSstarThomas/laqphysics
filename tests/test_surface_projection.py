import json
import numpy as np
import pytest
from fedvr import make_grid
from helium3d import Helium
from coupled_angular import CoupledHelium
from surface3d import extract
from surface_projection import project,integrate

@pytest.mark.parametrize('coupled',[False,True])
@pytest.mark.parametrize('fold',[False,True])
def test_projected_history_matches_direct_integral(coupled,fold,tmp_path):
    h=Helium(make_grid([0,.5,1,2,4,6,8],3,tail=6,ecs_angle=.3),2,M=0,cutoff_radii=[1,2])
    h.prepare_surface(5.)
    c=CoupledHelium(h,2) if coupled else h
    t=np.linspace(0,.4,9);A=lambda s:np.array([0.,0.,.04*np.sin(np.pi*s/.4)**2])
    labels=[(1,0,0),(2,0,0),(2,1,-1),(2,1,0),(2,1,1)]
    rng=np.random.default_rng(617);f=rng.normal(size=(len(t),h.n,len(h.surface_indices),c.nc))+1j*rng.normal(size=(len(t),h.n,len(h.surface_indices),c.nc))
    transform=c.transform if coupled else None;E=np.linspace(.3,.9,7);directions=(np.linspace(0,np.pi,13),np.zeros(13))
    expected,_=extract(h,f,t,A,labels,E,directions,t[1],angular_transform=transform,propagator='cf4')
    q,lm,meta=project(h,f,t,A,labels,t[1],angular_transform=transform,propagator='cf4',fold_m=fold,output=tmp_path/'q.npy')
    # Real cache metadata must survive JSON encoding, including dense-coupled
    # and product paths where NumPy comparison scalars previously leaked out.
    record=tmp_path/'projection.json';record.write_text(json.dumps(meta))
    assert json.loads(record.read_text())==meta
    got,_=integrate(q,lm,h.r[h.surface_indices].real,h.grid.weights[h.surface_indices].real,t,A,E,directions,t[1])
    np.testing.assert_allclose(got,expected,rtol=3e-12,atol=3e-12)

def test_circular_projection_retains_all_m_sectors():
    h=Helium(make_grid([0,.5,1,2,4,6,8],3,tail=6,ecs_angle=.3),1,M=None,cutoff_radii=[1,2]);h.prepare_surface(5)
    t=np.linspace(0,.4,9);A=lambda s:np.array([.03*np.sin(np.pi*s/.4)**2,.01*np.sin(2*np.pi*s/.4),0.])
    labels=[(1,0,0),(2,1,-1),(2,1,0),(2,1,1)];rng=np.random.default_rng(33)
    f=rng.normal(size=(len(t),h.n,len(h.surface_indices),h.nc))+1j*rng.normal(size=(len(t),h.n,len(h.surface_indices),h.nc))
    E=np.linspace(.2,.8,5);angles=(np.linspace(.1,3,11),np.linspace(0,6,11))
    expected,_=extract(h,f,t,A,labels,E,angles,t[1],propagator='cf4')
    q,lm,meta=project(h,f,t,A,labels,t[1],propagator='cf4')
    assert not meta['folded_m_sectors']
    got,_=integrate(q,lm,h.r[h.surface_indices].real,h.grid.weights[h.surface_indices].real,t,A,E,angles,t[1])
    np.testing.assert_allclose(got,expected,rtol=3e-12,atol=3e-12)
