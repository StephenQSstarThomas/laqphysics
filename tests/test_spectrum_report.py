import json
import numpy as np
from spectrum_report import publish,update_index

def test_narrow_single_pulse_range_still_has_named_SI_figure(tmp_path):
    case=tmp_path/'resonant__example';case.mkdir()
    config={'pulses':[{'pulse':{'omega':1.5,'cycles':180,'field':.0534},'polarization':'z'}]}
    (case/'run.json').write_text(json.dumps({'complete':True,'signature':'example','config':config}))
    (case/'input.json').write_text(json.dumps(config))
    e=np.linspace(.54,.65,31);p=np.array([np.exp(-((e-.59)/.004)**2),.4*np.exp(-((e-.605)/.006)**2)])
    np.savez(case/'spectrum.npz',source_signature='example',energy=e,angle_integrated=p,labels=[[1,0,0],[2,1,0]])
    result=publish(case,case.name)
    assert result['peak_height_ratio_03_over_06'] is None and result['target_passed'] is None
    assert (case/result['artifacts']['figure']).exists()
    assert 'F0p0534N180z' in result['artifacts']['figure']
    compact=np.load(case/result['artifacts']['spectrum'])
    np.testing.assert_array_equal(compact['total_density'],p.sum(axis=0))
    assert (case/'RUN.md').exists()

def test_both_bands_and_numpy_channel_labels_serialize(tmp_path):
    c={'pulses':[{'pulse':{'omega':1.2,'cycles':24,'field':.08},'polarization':'z'},
                 {'pulse':{'omega':1.5,'cycles':201,'field':.02,'start':240},'polarization':'sigma+'}],
       'design':{'target':'old-band preparation'}}
    (tmp_path/'run.json').write_text(json.dumps({'complete':True,'signature':'example','config':c}))
    e=np.linspace(.1,1.,101);p=np.array([.1*np.exp(-((e-.6)/.01)**2),np.exp(-((e-.3)/.03)**2)])
    np.savez(tmp_path/'spectrum.npz',source_signature='example',energy=e,angle_integrated=p,labels=np.array([[1,0,0],[2,1,1]],dtype=np.int64))
    (tmp_path/'ionic_transfer.json').write_text(json.dumps({'transfers_from_1s':[{'labels':[[1,0,0],[2,1,1]],'probabilities':[.001,.998]}]}))
    result=publish(tmp_path,'complete_example')
    assert result['target_passed'] and result['prepared_ion_P_2p_plus1']==.998
    assert (tmp_path/result['artifacts']['conditional_figure']).exists()
    assert json.loads((tmp_path/'observables.json').read_text())['channel_peak_comparisons'][1]['label']==[2,1,1]
