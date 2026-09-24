# 精细两脉冲算例 + 收敛 He+ 探测：三脉冲预览

来源：[selected_F008_N48_pi201_refined_n3_accelerated](../../preparation_20260921/selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837/RUN.md)（lmax 3、Lmax 3、156 径向点，n≤3 共 14 个离子通道，结束于 1301.95 a.u.）。完整复振幅 `spectrum.npz` 在原服务器持久盘，路径见各算例 `run.json` 的 `source_run`。

探测输入：[σ−](refined_probe_w014_sigma_minus.json)、[σ+](refined_probe_w014_sigma_plus.json)。第三束 ω=0.14、F=0.02、48 周期，起点 1302 a.u.；离子通道 n≤3 加 5d±2、5g±2、5g±4。

```bash
python scripts/apply_ionic_probe.py --source <精细算例目录> \
  --config results/probe_20260924/refined_preview/refined_probe_w014_sigma_minus.json \
  --ion-basis configs/probe_20260924/ion_basis_converged.json --ionic-dt 0.25 --device cuda:0 --out-root <输出根>
```

本机每种手性约 8–11 分钟（GPU，20 个伴随通道，R96、lmax 7 的 He+ 基）。结果与对照图：[probe_vs_source_old_band.png](probe_vs_source_old_band.png)、[json](probe_vs_source_old_band.json)，以及两个命名算例目录中的总谱/符合谱图、NPZ/CSV 和 RUN.md。

二电子部分不是生产收敛网格，数值仅作预览；结论（总谱形状不变，离子末态重新分配，σ+ 使离子电离约 2.3%）与 He+ 扫描和因子化验证一致。
