#!/usr/bin/env python3
"""Regenerate numerical tables and plots from finished runs, never from expected answers."""
from pathlib import Path
import sys,json,numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from esss import driven,reduced_ion
from pulses import Pulse

def main():
    root=Path(__file__).resolve().parents[1];results=root/'results';out=results/'figures';out.mkdir(exist_ok=True)
    ground=json.loads((results/'ground_convergence.json').read_text(encoding='utf-8'));rows=[]
    one=[];three=[]
    for directory in sorted(results.iterdir()):
        if not directory.is_dir() or not (directory/'run.json').exists() or not (directory/'spectrum.npz').exists():continue
        meta=json.loads((directory/'run.json').read_text(encoding='utf-8'))
        if meta.get('synthetic',False):continue
        data=np.load(directory/'spectrum.npz');e=data['energy']
        is3='angle_integrated' in data;P=data['angle_integrated'] if is3 else data['pes']
        prob=simpson(P,x=e,axis=1);peaks=[e[find_peaks(y,prominence=y.max()*.1)[0]].tolist() for y in P]
        centroid=simpson(P*e,x=e,axis=1)/prob
        row={'run':directory.name,'dimension':'3d' if is3 else '1d','energy_maxima':e[P.argmax(axis=1)].tolist(),
             'peaks_10percent_prominence':peaks,'energy_centroids':centroid.tolist(),'channel_probabilities':prob.tolist(),
             'energy_window':[float(e[0]),float(e[-1])],'dt':meta['dt']}
        # Conditional pure-state mode entanglement, restricted to recorded channels/window.
        we=simpson(np.eye(len(e)),x=e,axis=1);k=np.sqrt(2*e)
        if is3:
            theta=data['theta']
            if meta['config'].get('M',0)==0:
                wa=simpson(np.eye(len(theta)),x=theta,axis=1)*2*np.pi*np.sin(theta)
            else:
                from numpy.polynomial.legendre import leggauss
                wa=np.repeat(leggauss(20)[1],32)*2*np.pi/32
            amp=data['amplitudes'].reshape(len(P),-1);weights=((we*k)[:,None]*wa[None,:]).ravel()
        else:
            amp=data['amplitudes_k'];weights=np.tile(we/k,2)
        rho,ent=reduced_ion(amp,weights)
        if is3 and meta['config'].get('M',0)==0:
            # For total M=0, the electron has m=-m_ion. Integrating azimuth
            # removes cross terms between distinct ionic m even though only phi=0
            # is stored for the cylindrically symmetric probability distributions.
            m=data['labels'][:,2];rho*=m[:,None]==m[None,:]
            lam=np.linalg.eigvalsh(rho).clip(0,1);nz=lam[lam>1e-15]
            ent.update(purity=float(sum(lam**2)),entropy_bits=float(-sum(nz*np.log2(nz))),
                       negativity_pure=float(((sum(np.sqrt(lam)))**2-1)/2))
        row['conditional_mode_entanglement']={**ent,'rho_real':rho.real.tolist(),'rho_imag':rho.imag.tolist(),
            'scope':'pure projection onto recorded ionic states and energy window; excludes spin dynamics and omitted channels'}
        if is3:
            hist=json.loads((directory/'history.json').read_text(encoding='utf-8'));pg=hist[-1]['ground_population'];row['ground_population']=pg
            if (directory/'checkpoint.npz').exists():
                from fedvr import make_grid
                grid=make_grid(**meta['config']['radial']);f=grid.interior_weights/abs(grid.weights)
                psi=np.load(directory/'checkpoint.npz')['psi'].reshape(len(f),len(f),meta['channels'],order='F')
                row['physical_inner_probability']=float(np.sum(abs(psi)**2*f[:,None,None]*f[None,:,None]))
            row['channel_yield_vs_depletion_relative_difference']=float((sum(prob)-(1-pg))/(1-pg))
            three.append((directory.name,e,P,data,meta))
        else:
            pg=meta['final_ground_population'];row['ground_population']=pg
            p=Pulse(**meta['config']['pulse'])
            model=abs(driven(e,p,eg=meta['energy'],e1=meta['ionic_energies'][0],e2=meta['ionic_energies'][1],d12=meta['d12'],dt=.1))**2
            l1=simpson(abs(P-model),x=e,axis=1)/simpson(model,x=e,axis=1)
            normalized=simpson(abs(P/prob[:,None]-model/simpson(model,x=e,axis=1)[:,None]),x=e,axis=1)
            row['esss_absolute_spectrum_L1_relative_error']=l1.tolist();row['esss_normalized_shape_L1_error']=normalized.tolist()
            row['esss_dg']='published 1D fitted value 0.5213; not re-fitted here'
            one.append((directory.name,e,P,model))
        rows.append(row)
    if one:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for name,e,P,model in one:
            for j in range(2):axes[j].plot(e,P[j],label=name)
        for j in range(2):
            axes[j].plot(e,model[j],'k--',label='ESSS, actual numerical thresholds')
            axes[j].set(xlabel='Electron energy (a.u.)',ylabel='dP/dE',title=f'1D ionic channel {j+1}',xlim=(.54,.65));axes[j].legend(fontsize=8)
        fig.savefig(out/'1d_tdse_convergence.png',dpi=180);plt.close(fig)
    if three:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        for name,e,P,data,meta in three:
            if 'resonant' not in name:ax.plot(e,P[0],label=name)
        ax.set(xlabel='Electron energy (a.u.)',ylabel='dP/dE',title='3D two-electron short-pulse TDSE, 1s ionic channel');ax.legend(fontsize=8)
        fig.savefig(out/'3d_short_spectrum.png',dpi=180);plt.close(fig)
        name,e,P,data,meta=next((x for x in reversed(three) if 'resonant' not in x[0]),three[-1])
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        im=ax.pcolormesh(data['theta']*180/np.pi,e,data['pes'][0],shading='auto')
        ax.set(xlabel='Emission angle (degrees)',ylabel='Energy (a.u.)',title=f'3D TDSE angular spectrum: {name}')
        fig.colorbar(im,ax=ax,label='dP/dE/dOmega');fig.savefig(out/'3d_short_angular.png',dpi=180);plt.close(fig)
        for name,e,P,data,meta in three:
            if 'resonant' not in name:continue
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            for j,y in enumerate(P):ax.plot(e,y,label=str(data['labels'][j]))
            ax.plot(e,P.sum(axis=0),'k--',label='sum of recorded channels')
            ax.set(xlabel='Energy (a.u.)',ylabel='dP/dE',title=f'3D TDSE: {name} (angular/radial convergence still required)');ax.legend()
            fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    summary={'ground_convergence':ground,'spectra':rows,'generated_from_completed_spectra_only':True}
    (results/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 数值结果（由原始文件自动汇总）','','运行 `python scripts/summarize_results.py` 可再生本文件。误差及未收敛条件须结合《物理与算法核验》《调试记录》阅读。','','## 基态收敛','','| 模型 | 网格/阶数 | lmax | 通道数 | 基态能量 | 本征残差 |','|---|---:|---:|---:|---:|---:|']
    for r in ground:
        mesh=f"N={r['n']}, dx={r['dx']:.5g}" if r['model']=='1d' else f"Nr={r['nrad']}, p={r['order']}"
        lines.append(f"| {r['model']} | {mesh} | {r.get('lmax','—')} | {r.get('channels','—')} | {r['energy']:.10f} | {r['residual']:.2e} |")
    lines+=['','## 已完成的谱计算','','主峰坐标是所保存能量网格的最大值，不是拟合后的无限精度峰位。有限能窗的积分不等于所有电离通道的总概率。','']
    for row in rows:
        lines += [f"### {row['run']}",'',f"- 时间步：{row['dt']:.8g}",f"- 各通道主峰：{row['energy_maxima']}",f"- 各通道能量质心：{row['energy_centroids']}",f"- 各通道积分概率：{row['channel_probabilities']}",f"- 高于 10% 显著性的局域极大值：{row['peaks_10percent_prominence']}",f"- 最终基态存活率：{row['ground_population']:.10g}"]
        if 'esss_normalized_shape_L1_error' in row:lines += [f"- 与 ESSS 归一化形状 L¹ 差：{row['esss_normalized_shape_L1_error']}",f"- 与 ESSS 绝对谱 L¹ 相对差：{row['esss_absolute_spectrum_L1_relative_error']}"]
        if 'channel_yield_vs_depletion_relative_difference' in row:lines += [f"- 记录通道产额相对基态耗尽的差：{row['channel_yield_vs_depletion_relative_difference']:.5g}（含有限能窗/其他通道效应）"]
        ent=row['conditional_mode_entanglement'];lines += [f"- 所记录单电离子空间的条件模式纠缠：S={ent['entropy_bits']:.6g} bit，purity={ent['purity']:.6g}，negativity={ent['negativity_pure']:.6g}（不含自旋动力学和遗漏通道）"]
        lines+=['']
    lines += ['## 图','','- `results/figures/1d_tdse_convergence.png`','- `results/figures/3d_short_spectrum.png`','- `results/figures/3d_short_angular.png`','- `results/esss/published_esss_panels.png`','- `results/esss/angular_esss.png`（归一化形状模型）','- `results/esss/duration_scan.gif`','', '长共振 TDSE 图若存在，仍应检查配置中的 lmax、径向阶数、dt、势截断半径及边界收敛，不可仅凭出现双峰宣称完整收敛。']
    (root/'docs/数值结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
