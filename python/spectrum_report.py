"""Mandatory, consistently named SI spectra and a concise human-readable run card."""
import json,os,shutil,fcntl
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from pulses import Pulse
from surface_storage import atomic_json,capped_npz

def parameter_tag(config):
    def number(x):return format(float(x),'.6g').replace('.','p')
    tags=[]
    for i,entry in enumerate(config['pulses']):
        p=Pulse(**entry['pulse']);pol={'sigma+':'sp','sigma-':'sm','z':'z'}[entry.get('polarization','z')]
        tag=f'p{i+1}w{number(p.omega)}F{number(p.field)}N{number(p.cycles)}{pol}'
        if p.start:tag+='t'+number(p.start)
        tags.append(tag)
    return '_'.join(tags)

def pulse_rows(config):
    result=[]
    for i,entry in enumerate(config['pulses']):
        p=Pulse(**entry['pulse']);result.append({'number':i+1,'polarization':entry.get('polarization','z'),**p.__dict__,
                'duration_au':p.duration,'duration_fs':p.duration*.0241888432659,
                'intensity_FWHM_fs':p.duration*.0241888432659*2*np.arccos(2**(-.25))/np.pi,
                'equivalent_cycle_average_peak_intensity_W_cm2':3.5094452e16*p.field**2})
    return result

def publish(out,run_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);meta=json.loads((out/'run.json').read_text())
    if not meta.get('complete'):raise ValueError('cannot label an incomplete propagation as a final SI spectrum')
    source=out/'spectrum.npz'
    if not source.exists() and (out/'observables.json').exists():
        source=out/json.loads((out/'observables.json').read_text())['artifacts']['spectrum']
    # Publishing energy spectra does not need the much larger angular amplitudes.
    with np.load(source) as loaded:d={k:loaded[k] for k in ['energy','angle_integrated','labels','source_signature']}
    if str(d['source_signature'])!=meta['signature']:raise ValueError('spectrum/configuration mismatch')
    e=d['energy'];P=d['angle_integrated'];total=P.sum(axis=0);labels=list(map(tuple,d['labels']))
    if not np.isfinite(P).all() or np.any(P<0):raise ValueError('invalid spectrum')
    windows=[]
    for low,high in [(.18,.42),(.52,.68)]:
        mask=(e>=low)&(e<=high)
        if mask.sum()<3:
            windows.append({'window':[low,high],'available':False,'complete_window':False,'peak_height':None,'peak_energy_grid':None,'yield':None,'channel_yields':None});continue
        y=simpson(P[:,mask],x=e[mask],axis=1)
        windows.append({'window':[low,high],'available':True,'complete_window':bool(e[0]<=low and e[-1]>=high),
                        'energy_window_used':[float(e[mask][0]),float(e[mask][-1])],'peak_height':float(total[mask].max()),'peak_energy_grid':float(e[mask][total[mask].argmax()]),
                        'yield':float(sum(y)),'channel_yields':y.tolist()})
    target=labels.index((2,1,1)) if (2,1,1) in labels else None
    fraction=windows[0]['channel_yields'][target]/windows[0]['yield'] if target is not None and windows[0]['yield'] and windows[0]['yield']>1e-10 else None
    both=all(w['complete_window'] for w in windows)
    ratio=windows[0]['peak_height']/windows[1]['peak_height'] if both and windows[1]['peak_height'] else None
    yr=windows[0]['yield']/windows[1]['yield'] if both and windows[1]['yield'] else None
    applicable=bool(meta['config'].get('design')) and len(meta['config']['pulses'])>=2
    transfer=json.loads((out/'ionic_transfer.json').read_text()) if (out/'ionic_transfer.json').exists() else {}
    channel_yields=simpson(P,x=e,axis=1);conditional_valid=channel_yields>1e-10
    conditional=np.divide(P,channel_yields[:,None],out=np.zeros_like(P),where=conditional_valid[:,None])
    channel_comparisons=[]
    if both:
        old_mask=(e>=.18)&(e<=.42);new_mask=(e>=.52)&(e<=.68)
        for label,row in zip(labels,P):
            first=float(row[old_mask].max());second=float(row[new_mask].max())
            channel_comparisons.append({'label':list(map(int,label)),'peak_03':first,'peak_06':second,'peak_ratio_03_over_06':first/second if second>0 else None})
    prepared=None
    for row in transfer.get('transfers_from_1s',[]):
        tlabels=list(map(tuple,row['labels']))
        if (2,1,1) in tlabels:prepared=float(row['probabilities'][tlabels.index((2,1,1))])
    stem=run_id+'__'+parameter_tag(meta['config'])+'__SI'
    if len(stem)>225:raise ValueError('result name too long; shorten the case stem/tag')
    artifacts={'spectrum':stem+'.npz','csv':stem+'.csv','figure':'figures/'+stem+'.png','pdf':'figures/'+stem+'.pdf',
               'conditional_figure':'figures/'+stem+'__conditional.png','full_angular_amplitudes':'spectrum.npz'}
    result={'run_id':run_id,'configuration_signature':meta['signature'],'complete':True,'labels':d['labels'].tolist(),'bands':windows,'artifacts':artifacts,
            'peak_height_ratio_03_over_06':ratio,'yield_ratio_03_over_06':yr,'old_band_2p_plus1_fraction':fraction,
            'prepared_ion_P_2p_plus1':prepared,
            'channel_peak_comparisons':channel_comparisons,'channel_yields_on_energy_grid':channel_yields.tolist(),
            'conditional_normalization':'Unit area over the explicitly computed energy grid, separately for each ion; not absolute counts.',
            'predeclared_targets':{'peak_height_ratio_min':3.,'band_yield_ratio_min':5.,'old_band_2p_plus1_fraction_min':.95,'prepared_ion_transfer_min':.95},
            'target_applicable':applicable,
            'target_passed':bool(ratio is not None and yr is not None and fraction is not None and prepared is not None and ratio>=3 and yr>=5 and fraction>=.95 and prepared>=.95) if applicable else None,
            'scope':'Sum of explicitly recorded bound ionic channels, plus channel-resolved SI spectra; no claim of complete ionic-channel or spatial convergence.',
            'pulses':pulse_rows(meta['config']),'ionic_transfer':transfer}
    atomic_json(out/'observables.json',result)
    data_name=artifacts['spectrum'];destination=out/data_name
    # The primary SI product is small and directly plottable. Full complex
    # angle-resolved amplitudes remain in spectrum.npz for further research.
    capped_npz(destination,meta.get('max_file_bytes',4_000_000_000),energy=e,angle_integrated=P,total_density=total,
               conditional_density_on_grid=conditional,conditional_valid=conditional_valid,channel_yields_on_grid=channel_yields,
               labels=d['labels'],source_signature=meta['signature'],run_id=run_id,
               scope='single ionization summed over explicitly recorded bound ionic channels')
    table=np.column_stack([e,total,*P]);header='energy_au,total_recorded_SI,'+','.join(f'n{n}_l{l}_m{m:+d}' for n,l,m in labels)
    np.savetxt(out/artifacts['csv'],table,delimiter=',',header=header,comments='')
    fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    polarizations={p.get('polarization','z') for p in meta['config']['pulses']}
    default_channels=[[1,0,0]]+([[2,1,1]] if 'sigma+' in polarizations else [])+([[2,1,-1]] if 'sigma-' in polarizations else [])
    if polarizations=={'z'}:default_channels.append([2,1,0])
    chosen=[tuple(x) for x in meta['config'].get('spectrum',{}).get('plot_channels',default_channels)]
    for ax in axes:
        ax.plot(e,total,color='black',lw=1.6,label='SI: sum of recorded ionic channels')
        for j,label in enumerate(chosen):
            color=['C0','C3','C2','C4','C5'][j%5]
            if label in labels:ax.plot(e,P[labels.index(label)],color=color,lw=1.1,label=f'ion (n,l,m)={label}')
        ax.axvspan(.18,.42,color='C0',alpha=.07);ax.axvspan(.52,.68,color='C3',alpha=.07);ax.grid(alpha=.18)
        ax.set(xlabel='Electron energy (a.u.)',ylabel='dP / dE (a.u.)')
    axes[0].set_xlim((.1,.78) if both else (e[0],e[-1]));axes[0].set_ylim(bottom=0);axes[0].legend(fontsize=8)
    axes[0].set_title(f'Absolute SI spectrum | peak ratio {ratio:.2f}, band-yield ratio {yr:.2f}\nOld-band ion 2p(+1) fraction: {fraction:.4f}' if fraction is not None and ratio is not None and yr is not None else 'Absolute single-ionization spectrum')
    axes[1].set_yscale('log');axes[1].set_xlim(e[0],e[-1]);axes[1].set_ylim(max(total.max()*1e-8,1e-14),max(total.max()*1.5,1e-12))
    pulses=result['pulses'];title='; '.join(f"{p['polarization']}: w={p['omega']:g}, F={p['field']:.5g}, N={p['cycles']:g}, start={p['start']:g}" for p in pulses)
    numerical=f"lmax={meta['config'].get('lmax','?')}, Lmax={meta['config'].get('total_Lmax','product')}, Nr={meta.get('nrad','?')}, dt={meta.get('dt',float('nan')):.6g}; recorded ion n<={max(n for n,l,m in labels)}"
    fig.suptitle(run_id+'\n'+title+'\n'+numerical,fontsize=8.5)
    figures=out/'figures';figures.mkdir(exist_ok=True);figure=figures/stem
    for suffix in ['png','pdf']:
        temp=figure.with_name(figure.name+'.tmp.'+suffix);fig.savefig(temp,dpi=220);os.replace(temp,Path(str(figure)+'.'+suffix))
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for j,label in enumerate(chosen):
        if label in labels:
            index=labels.index(label)
            if conditional_valid[index]:ax.plot(e,conditional[index],color=['C0','C3','C2','C4','C5'][j%5],label=f'ion {label}; P(window)={channel_yields[index]:.5g}')
    ax.set(xlabel='Electron energy (a.u.)',ylabel='Conditional density (a.u.)',xlim=(.1,.78) if both else (e[0],e[-1]),
           title='Ion-conditioned spectra: unit area on the computed energy grid\nThese heights are not absolute signal strengths')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.suptitle(run_id,fontsize=8.5)
    temp=out/'figures'/f'{stem}__conditional.tmp.png';fig.savefig(temp,dpi=220);os.replace(temp,out/artifacts['conditional_figure']);plt.close(fig)
    resources=json.loads((out/'resource_estimate.json').read_text()) if (out/'resource_estimate.json').exists() else {}
    lines=[f'# {run_id}','',f"[输入参数](input.json)（签名 `{meta['signature']}`）。",'',
           f"[主谱数据 NPZ]({data_name}) · [CSV]({artifacts['csv']})。",f"[主图 PNG]({artifacts['figure']}) · [PDF]({artifacts['pdf']})。",'',
           '主谱 NPZ 存能量、通道积分谱与总谱；完整复振幅和角分布保存在 spectrum.npz。','',
           f"[归一化条件谱]({artifacts['conditional_figure']})：各通道在计算能窗内积分归一，仅比较形状，不能据此判断峰的绝对强弱。",'',
           '图中各曲线使用同一绝对概率密度刻度，未按各自峰高归一化。总谱仅求和已记录的束缚离子通道。','',
           '| 脉冲 | 偏振 | ω / a.u. | F / a.u. | 周期数 | 起点 / a.u. | 支撑宽度 / fs |','|---|---|---:|---:|---:|---:|---:|']
    for p in pulses:lines.append(f"| {p['number']} | {p['polarization']} | {p['omega']:.8g} | {p['field']:.8g} | {p['cycles']:g} | {p['start']:g} | {p['duration_fs']:.4f} |")
    lines+=['','`F` 为线偏振峰值电场；圆偏振两个分量各为 F/√2，具有相同周期平均强度。包络为 sin²。',
            '`dt`、径向网格、lmax、total_Lmax 和离子通道均在 input.json；这些是待继续收敛的数值参数。','',
            f'0.3/0.6 峰高比：{ratio}；能区积分产额比：{yr}（None 表示该能区未计算）。',
            f'旧能区 2p+1 条件份额：{fraction}。详细通道与准备态离子转移见 observables.json。',
            f'从已准备的离子 1s 到最终 2p+1 的绝对转移概率：{prepared}；与上面的条件份额分别报告。',
            f"是否达到本次制备目标：{result['target_passed']}。该标志不等于完整数值收敛。",'',
            f"存储模式：{meta.get('surface_storage','legacy')}；文件上限：{meta.get('max_file_bytes',5000000000)} 字节。",
            f"资源估算：resource_estimate.json；预计持久数组 {resources.get('persistent_output_estimate_bytes','unknown')} 字节。"]
    (out/'RUN.md').write_text('\n'.join(lines)+'\n')
    return result

def update_index(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not (folder/'STATUS.json').exists():continue
            status=json.loads((folder/'STATUS.json').read_text());c=json.loads((folder/'input.json').read_text())
            meta=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else {}
            obs=json.loads((folder/'observables.json').read_text()) if (folder/'observables.json').exists() else {}
            rows.append({'run_id':folder.name,'state':status['state'],'input':str(folder.name+'/input.json'),
                'run_card':folder.name+'/RUN.md','figure':folder.name+'/'+status['figure'] if status.get('figure') else None,
                'spectrum':folder.name+'/'+status['spectrum'] if status.get('spectrum') else None,
                'pulses':pulse_rows(c),'peak_ratio':obs.get('peak_height_ratio_03_over_06'),
                'yield_ratio':obs.get('yield_ratio_03_over_06'),'old_2p_plus1_fraction':obs.get('old_band_2p_plus1_fraction'),
                'prepared_ion_P_2p_plus1':obs.get('prepared_ion_P_2p_plus1'),
                'resolution':{'lmax':c['lmax'],'Lmax':c.get('total_Lmax'),'radial_points':meta.get('nrad'),
                              'recorded_ion_nmax':max(x[0] for x in c.get('ionic_channels',[[1,0,0]]))},
                'targets_passed':obs.get('target_passed')})
        rows.sort(key=lambda r:(r['state']!='complete',r['targets_passed'] is not True,-r['resolution']['lmax'],r['run_id']))
        atomic_json(root/'catalog.json',{'cases':rows,'note':'Only state=complete has the mandatory final spectrum and figures.'})
        text=['# 单电离谱结果索引','','每行是一份独立输入；完整参数在 input.json，参数含义和结果说明在 RUN.md。',
              '所有图采用绝对概率密度。同一配置重提会恢复检查点，不会新建一个伪重复结果。','',
              '| 模拟与输入 | 状态 | 主图 | 0.3/0.6 峰高比 | 能区产额比 | 旧能区 2p+1 份额 |',
              '|---|---|---|---:|---:|---:|']
        def number(x):return '—' if x is None else f'{x:.5g}'
        for r in rows:
            figure=f"[单电离谱]({r['figure']})" if r['figure'] else '等待完成'
            text.append(f"| [{r['run_id'].split('__')[0]}]({r['run_card']}) · [输入]({r['input']}) | {r['state']} | {figure} | {number(r['peak_ratio'])} | {number(r['yield_ratio'])} | {number(r['old_2p_plus1_fraction'])} |")
        recommended=next((r for r in rows if r['state']=='complete' and r['targets_passed'] is True),None)
        if recommended:text.insert(2,f"**优先查看：[当前通过目标的最高角基结果]({recommended['figure']})**。对应参数与离子通道范围见该行输入和 RUN.md。\n")
        if (root/'figures/preparation_before_after.png').exists():
            text.insert(2,'[改参数前后对照](figures/preparation_before_after.png) · [固定 π 面积的峰宽/峰高对照](figures/fixed_pi_area_width_vs_height.png) · [单脉冲控制组核验](figures/separated_pulse_controls.png)。\n')
        (root/'INDEX.md').write_text('\n'.join(text)+'\n')
    return rows
