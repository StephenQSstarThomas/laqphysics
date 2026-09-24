"""Third (probe) pulse: ionic transfer map, campaign inputs and report diagnostics."""
import json,importlib.util
from pathlib import Path
import numpy as np
import pytest
from ionic_probe import transfer_matrix,factorized_amplitudes,ion_model,model_levels

root=Path(__file__).resolve().parents[1]
ION={'radial':{'edges':[0,.5,1,2,4,8,12,16,20],'order':5,'tail':8,'ecs_angle':.4,'alpha':.8},'lmax':2,'dt':.1}

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,root/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_transfer_backends_agree_and_obey_circular_selection_rules():
    pulses=[{'pulse':{'omega':.2778,'cycles':3,'field':.02},'polarization':'sigma-','role':'probe'}]
    T=2*np.pi*3/.2778;labels=[(2,1,1),(3,2,0),(3,2,2),(3,2,1),(3,1,0)]
    lu,_=transfer_matrix(ION,pulses,0.,T,[(2,1,1)],labels,device='lu')
    cpu,info=transfer_matrix(ION,pulses,0.,T,[(2,1,1)],labels,device='cpu')
    np.testing.assert_allclose(cpu,lu,rtol=1e-9,atol=1e-11)
    assert info['maximum_linear_residual']<1e-10
    # An xy-plane field conserves the parity of l+m exactly (z reflection).
    assert abs(lu[3,0])<1e-12 and abs(lu[4,0])<1e-12
    # sigma- absorption 2p(+1)->3d(0) is near resonant; 3d(+2) needs the counter-rotating term.
    plus,_=transfer_matrix(ION,[dict(pulses[0],polarization='sigma+')],0.,T,[(2,1,1)],labels,device='lu')
    assert abs(lu[1,0])**2>1e-4 and abs(lu[1,0])>100*abs(plus[1,0]) and abs(plus[2,0])>100*abs(lu[2,0])

def test_field_free_map_is_diagonal_phase_of_this_grid():
    pulses=[{'pulse':{'omega':.3,'cycles':2,'field':0.},'polarization':'sigma+','role':'probe'}]
    labels=[(1,0,0),(2,1,1),(2,1,-1)];T=40.
    U,_=transfer_matrix(ION,pulses,0.,T,labels,labels,device='lu')
    assert np.max(abs(U-np.diag(np.diag(U))))<1e-8
    levels=model_levels(ion_model(ION),labels)
    for j,(n,l,m) in enumerate(labels):
        assert abs(abs(U[j,j])-1)<2e-4
        assert abs(np.angle(U[j,j]*np.exp(1j*levels[(n,l)].real*T)))<2e-3

def test_factorized_amplitudes_is_the_channel_matrix_product():
    rng=np.random.default_rng(3);b=rng.normal(size=(3,5,4))+1j*rng.normal(size=(3,5,4));U=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    np.testing.assert_allclose(factorized_amplitudes(b,[(1,0,0),(2,1,1),(2,1,0)],U,[(2,1,1),(3,1,-1)]),np.einsum('ij,jek->iek',U,b))
    with pytest.raises(ValueError):factorized_amplitudes(b,[(1,0,0)],U,[(2,1,1),(3,1,-1)])

def test_generated_probe_inputs_extend_their_source_exactly():
    from campaign_resources import estimate
    from run3d import vector_function
    module=load_script('apply_ionic_probe')
    for folder in ['production','validation','smoke']:
        base=root/'configs/probe_20260924'/folder;source=json.loads((base/'source.json').read_text(encoding='utf-8'))
        for path in sorted(base.glob('probe*.json')):
            probe=json.loads(path.read_text(encoding='utf-8'))
            if probe['lmax']==source['lmax']:assert module.check_compatible(source,probe)==source['end_time']
            else:
                with pytest.raises(ValueError,match='lmax'):module.check_compatible(source,probe)
                assert module.check_compatible(source,{**probe,'lmax':source['lmax']})==source['end_time']
            assert [p.get('role') for p in probe['pulses']]==['ionize','prepare','probe']
            assert probe['pulses'][2]['pulse']['start']==source['end_time']
            assert all(0<=l<=probe['lmax'] and l<n and abs(m)<=l for n,l,m in probe['ionic_channels'])
            assert set(map(tuple,probe['probe']['target_channels']))<=set(map(tuple,probe['ionic_channels']))
            A,pulses=vector_function(probe);T=module.end_time(probe)
            assert np.max(abs(A(T)))<1e-10 and np.max(abs(A(source['end_time'])))<1e-10
            resources=estimate(probe);assert resources['largest_output_file_upper_bytes']<=5_000_000_000
    plan=json.loads((root/'configs/probe_20260924/production/plan.json').read_text(encoding='utf-8'))
    assert all((root/'configs/probe_20260924/production'/(name+'.json')).exists() for name in plan['cases'])
    assert set(plan['primary_cases'])<=set(plan['cases'])

def test_factorization_refuses_changed_numerics_and_early_probes():
    module=load_script('apply_ionic_probe')
    source=json.loads((root/'configs/probe_20260924/validation/source.json').read_text(encoding='utf-8'))
    probe=json.loads((root/'configs/probe_20260924/validation/probe.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError,match='numerical parameters'):module.check_compatible(source,{**probe,'dt':.05})
    early=json.loads(json.dumps(probe));early['pulses'][2]['pulse']['start']=source['end_time']-10
    with pytest.raises(ValueError,match='before the source run ended'):module.check_compatible(source,early)
    unmarked=json.loads(json.dumps(probe));unmarked['pulses'][2].pop('role')
    with pytest.raises(ValueError,match='role=probe'):module.check_compatible(source,unmarked)

def test_report_marks_probe_runs_and_finds_an_old_band_doublet(tmp_path):
    from spectrum_report import publish,update_index
    c={'pulses':[{'pulse':{'omega':1.2,'cycles':24,'field':.08},'polarization':'z','role':'ionize'},
                 {'pulse':{'omega':1.5,'cycles':201,'field':.02,'start':240},'polarization':'sigma+','role':'prepare'},
                 {'pulse':{'omega':.14,'cycles':48,'field':.02,'start':1200},'polarization':'sigma-','role':'probe'}],
       'lmax':4,'design':{'target':'preparation'},'probe':{'target_channels':[[3,1,-1]]}}
    case=tmp_path/'probe_case__abc';case.mkdir()
    (case/'run.json').write_text(json.dumps({'complete':True,'signature':'s','config':c}));(case/'input.json').write_text(json.dumps(c))
    (case/'STATUS.json').write_text(json.dumps({'state':'running'}))
    e=np.linspace(.1,1.,901);doublet=np.exp(-((e-.29)/.006)**2)+.8*np.exp(-((e-.315)/.006)**2)
    P=np.array([.01*np.exp(-((e-.6)/.01)**2),doublet,.2*doublet])
    np.savez(case/'spectrum.npz',source_signature='s',energy=e,angle_integrated=P,labels=[[1,0,0],[2,1,1],[3,1,-1]])
    result=publish(case,case.name);probe=result['probe']
    assert result['target_applicable'] is False and result['target_passed'] is None
    assert probe['old_band_total_doublet'] and abs(probe['old_band_total_splitting_au']-.025)<2e-3
    assert abs(probe['old_band_channel_fractions']['[3, 1, -1]']-.2/1.2)<1e-6
    assert 'p3w0p14F0p02N48smt1200' in result['artifacts']['figure'] and (case/result['artifacts']['figure']).exists()
    update_index(tmp_path);index=(tmp_path/'INDEX.md').read_text(encoding='utf-8')
    assert '第三束探测' in index and '旧能区极大值 2' in index
    assert '| 3 | probe | sigma- |' in (case/'RUN.md').read_text(encoding='utf-8')

def test_band_edges_on_a_flank_are_not_reported_as_peaks():
    from spectrum_report import band_peaks
    e=np.linspace(.1,.5,401);single=np.exp(-((e-.3)/.02)**2)+.4*np.exp(-((e-.45)/.02)**2)
    peaks=band_peaks(e,single,.18,.42)
    assert len(peaks)==1 and abs(peaks[0]['energy']-.3)<1e-3

def test_ecs_c_dual_projection_is_locked():
    """Bound states reaching into the ECS region: only the analytic c-dual gives a
    unitary field-free map (kets used as duals give |U_nn| up to 1.13 here)."""
    ion={'radial':{'edges':[0,.5,1,2,4,6],'order':6,'tail':16,'ecs_angle':.5,'alpha':.8},'lmax':2}
    pulses=[{'pulse':{'omega':.3,'cycles':2,'field':0.},'polarization':'sigma+','role':'probe'}]
    labels=[(2,1,1),(3,1,1),(3,2,0)]
    U,_=transfer_matrix(ion,pulses,0.,20.,labels,labels,dt=.1,device='lu')
    np.testing.assert_allclose(abs(np.diag(U)),1,atol=1e-4);assert np.max(abs(U-np.diag(np.diag(U))))<1e-8

def test_compatibility_uses_run_defaults_physical_pulses_and_grid():
    from run_defaults import effective_config
    module=load_script('apply_ionic_probe')
    source=json.loads((root/'configs/probe_20260924/validation/source.json').read_text(encoding='utf-8'))
    probe=json.loads((root/'configs/probe_20260924/validation/probe.json').read_text(encoding='utf-8'))
    recorded=effective_config(source)            # what run.json holds after simulate_spectrum
    bare=json.loads(json.dumps(probe));bare.pop('time_integrator');bare.pop('ionic_propagator')
    bare['pulses'][0]['pulse']['start']=0.       # explicit default start is the same pulse
    assert module.check_compatible(recorded,bare)==source['end_time']
    moved=json.loads(json.dumps(probe));moved['spectrum']['theta_points']+=2
    with pytest.raises(ValueError,match='amplitude grid'):module.check_compatible(recorded,moved)
    basis=module.ion_basis(probe,{'lmax':4,'note':'x'});assert basis['lmax']==4 and basis['radial']==probe['radial'] and basis['cutoff_radii']==probe['cutoff_radii']
    with pytest.raises(ValueError,match='cutoff_radii explicitly'):module.ion_basis(probe,{'radial':probe['radial']})
    assert 'cutoff_radii' not in module.ion_basis(probe,{'radial':probe['radial'],'cutoff_radii':None})

def test_factorized_case_end_to_end_on_a_finished_source(tmp_path):
    """Tiny source propagated with the production driver, then the factorized probe:
    named outputs, channel order, and the 1s transfer mapped through the same U."""
    from run3d import run
    from run_defaults import effective_config
    module=load_script('apply_ionic_probe')
    source=effective_config(json.loads((root/'configs/probe_20260924/smoke/source.json').read_text(encoding='utf-8')))
    probe=json.loads((root/'configs/probe_20260924/smoke/probe.json').read_text(encoding='utf-8'))
    run(source,tmp_path/'source',backend='torch',device='cpu',ground_cache=tmp_path/'ground')
    out=tmp_path/'derived'/'probe__factorized__test'
    result=module.execute(out,tmp_path/'source',probe,out.name,'test-signature','lu',None)
    status=json.loads((out/'STATUS.json').read_text(encoding='utf-8'));meta=json.loads((out/'run.json').read_text(encoding='utf-8'))
    assert status['state']=='complete' and meta['method']=='ionic_probe_factorization' and (out/result['artifacts']['figure']).exists()
    with np.load(out/'spectrum.npz') as d,np.load(tmp_path/'source/spectrum.npz') as s:
        assert [tuple(x) for x in d['labels']]==[tuple(x) for x in probe['ionic_channels']]
        U,_=transfer_matrix(probe,probe['pulses'],source['end_time'],module.end_time(probe),[tuple(x) for x in s['labels']],d['labels'],device='lu')
        np.testing.assert_allclose(d['amplitudes'],np.tensordot(U,s['amplitudes'],axes=([1],[0])),rtol=1e-12,atol=1e-14)
    transfer=json.loads((out/'ionic_transfer.json').read_text(encoding='utf-8'))
    original=json.loads((tmp_path/'source/ionic_transfer.json').read_text(encoding='utf-8'))['transfers_from_1s'][0]
    mapped=U@(np.array(original['amplitude_real'])+1j*np.array(original['amplitude_imag']))
    np.testing.assert_allclose(transfer['transfers_from_1s'][0]['probabilities'],abs(mapped)**2,rtol=1e-12,atol=1e-15)
    assert 'probe__factorized' in (tmp_path/'derived/INDEX.md').read_text(encoding='utf-8')
