import numpy as np
from convergence import compare

def test_peak_translation_is_rejected_without_alignment():
    e=np.linspace(.54,.65,221)
    def data(offset):
        y=np.exp(-((e-.588-offset)/.004)**2)+.8*np.exp(-((e-.605-offset)/.004)**2)
        return {'energy':e,'angle_integrated':np.array([y,.7*y]),'labels':np.array([[1,0,0],[2,1,0]])}
    limits={'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,'gate_channels':[[1,0,0],[2,1,0]]}
    same=compare(data(0),data(0),limits);shifted=compare(data(0),data(.002),limits)
    assert same['status']=='passed' and shifted['status']=='failed'
    assert abs(shifted['maximum_peak_shift']-.002)<1e-12

def test_equal_yield_and_shape_do_not_hide_yield_failure():
    e=np.linspace(.1,.9,161);y=np.exp(-((e-.4)/.05)**2)
    a={'energy':e,'angle_integrated':y[None,:],'labels':np.array([[1,0,0]])};b=dict(a,angle_integrated=1.03*y[None,:])
    limits={'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,'gate_channels':[[1,0,0]]}
    result=compare(a,b,limits)
    assert result['normalized_shape_L1'][0]<1e-14 and result['status']=='failed'

def test_peak_comparison_uses_native_grids_without_interpolation_cusp():
    def data(n):
        e=np.linspace(.2,.6,n);y=np.exp(-((e-.40017)/.015)**2)
        return {'energy':e,'angle_integrated':y[None,:],'labels':np.array([[1,0,0]])}
    limits={'peak_shift_au':5e-6,'normalized_shape_L1':.02,'relative_yield':.02,'gate_channels':[[1,0,0]]}
    result=compare(data(201),data(1001),limits)
    assert result['maximum_peak_shift']<5e-6 and result['status']=='passed'
