"""Fixed, absolute-energy convergence metrics, with no fitted peak alignment."""
import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks

def peaks(energy,density):
    answer=[]
    for i in find_peaks(density,prominence=.05*np.max(density))[0]:
        fit=np.polyfit(energy[i-1:i+2]-energy[i],density[i-1:i+2],2)
        offset=-fit[1]/(2*fit[0]) if fit[0]<0 else 0.
        if abs(offset)>max(energy[i+1]-energy[i],energy[i]-energy[i-1]):offset=0.
        answer.append(float(energy[i]+offset))
    return answer

def compare(first,second,criteria,window=None):
    low=max(first['energy'][0],second['energy'][0]);high=min(first['energy'][-1],second['energy'][-1])
    if window is not None:low=max(low,window[0]);high=min(high,window[1])
    e=second['energy'];e=e[(e>=low-1e-12)&(e<=high+1e-12)]
    if len(e)<3:raise ValueError('fewer than three common energy samples')
    a=np.array([np.interp(e,first['energy'],row) for row in first['angle_integrated']])
    b=np.array([np.interp(e,second['energy'],row) for row in second['angle_integrated']])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('nonfinite spectrum')
    if min(a.min(),b.min())<0:raise ValueError('negative probability density')
    labels_a=list(map(tuple,first['labels']));labels_b=list(map(tuple,second['labels']))
    if set(labels_a)!=set(labels_b):raise ValueError('recorded ionic channels differ')
    ca=a[[labels_a.index(tuple(x)) for x in criteria['gate_channels']]]
    cb=b[[labels_b.index(tuple(x)) for x in criteria['gate_channels']]]
    ya=simpson(ca,x=e,axis=1);yb=simpson(cb,x=e,axis=1)
    if min(ya.min(),yb.min())<=0:raise ValueError('gate channel has zero yield')
    shape=simpson(abs(ca/ya[:,None]-cb/yb[:,None]),x=e,axis=1);change=yb/ya-1
    # Locate peaks on each native grid. Fitting a parabola to an interpolated
    # coarse-grid cusp introduces an artificial shift when the grid is refined.
    def native_peaks(data):
        energy=data['energy'];mask=(energy>=e[0]-1e-12)&(energy<=e[-1]+1e-12)
        return peaks(energy[mask],data['angle_integrated'][:,mask].sum(axis=0))
    pa=native_peaks(first);pb=native_peaks(second)
    shift=float(max(abs(np.array(pa)-pb))) if pa and len(pa)==len(pb) else None
    passed=shift is not None and shift<=criteria['peak_shift_au'] and max(shape)<=criteria['normalized_shape_L1'] and max(abs(change))<=criteria['relative_yield']
    return {'status':'passed' if passed else 'failed','energy_window':[float(e[0]),float(e[-1])],
            'gate_channels':criteria['gate_channels'],'reference_yields':ya.tolist(),'refined_yields':yb.tolist(),
            'normalized_shape_L1':shape.tolist(),'relative_yield_change':change.tolist(),
            'reference_peaks':pa,'refined_peaks':pb,'maximum_peak_shift':shift,
            'all_recorded_channel_yields':simpson(b,x=e,axis=1).tolist(),'labels':second['labels'].tolist()}
