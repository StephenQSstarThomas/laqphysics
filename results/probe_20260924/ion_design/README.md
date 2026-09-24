# 制备好的 He+ 受第三束作用：设计扫描

每一行是一次含连续态的 He+ TDSE（`python/ionic_tdse.py`）：初态 2p(+1)，只加探测脉冲，dt=0.25。结果为探测结束（A=0）后 n≤6 的束缚布居；“电离/未记录”= 1 − Σ(n≤6)。表格见 [summary.md](summary.md)，完整数据见 [summary.json](summary.json)。

离子基（径向网格、库仑截断）取自下列输入：

| 标签 | 基础输入 | 说明 |
|---|---|---|
| production-grid(R32,cut16-20) | [production_base.json](production_base.json) | 两脉冲生产 reference 的网格 |
| R64,cut32-40 | [extended_grid_base.json](extended_grid_base.json) | 三脉冲生产输入所用的扩展网格 |
| R96,cut60-80 | [bigbox_base.json](bigbox_base.json) | 收敛检查 |
| R96,no-cutoff | [bigbox_nocut_base.json](bigbox_nocut_base.json) | 物理 He+（无截断），即 `configs/probe_20260924/ion_basis_converged.json` |

复现（扫描列表为 [scan1.json](scan1.json)、[scan_main.json](scan_main.json)、[scan_014.json](scan_014.json)）：

```bash
python scripts/probe_ion_design.py --base results/probe_20260924/ion_design/extended_grid_base.json \
  --scan results/probe_20260924/ion_design/scan1.json --out <输出目录> --devices cuda:0 --jobs 4
python scripts/probe_ion_design.py --base results/probe_20260924/ion_design/bigbox_nocut_base.json \
  --scan results/probe_20260924/ion_design/scan_014.json --lmax 7 --out <输出目录> --devices cuda:0 --jobs 2
python scripts/collect_probe_design.py "LABEL=<输出目录>" ... --out results/probe_20260924/ion_design/summary
```

本机每个 48 周期算例约 5–15 分钟（GPU，lmax 4–8）。
