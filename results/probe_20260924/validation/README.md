# 第三束探测与编码修复：核验记录（2026-09-24）

本目录记录本轮实际运行过的检查及结果。没有运行生产规模的三脉冲任务。说明文档见 [第三束探测](../../../docs/probe20260924/第三束探测.md)。

## 软件测试

| 套件 | 结果 | 日志 |
|---|---|---|
| Release 库，全部测试 | 107 passed | [tests_release.log](tests_release.log) |
| Debug 库（`-fcheck=all`、浮点陷阱），全部测试 | 107 passed | [tests_debug.log](tests_debug.log) |
| CUDA 专项（Torch 后端、恢复、在线 GPU 累加、离子 TDSE） | 13 passed（`HELIUM_TEST_DEVICE=cuda:0`） | [tests_cuda.log](tests_cuda.log) |

新增 11 项（原有 96 项全部保留）：`tests/test_probe.py` 10 项（离子转移的 LU/CPU 后端一致性、xy 平面场的 l+m 宇称选择定则与手性对比、无场时为对角相位、因子化代数、生成输入的时间网格与兼容性、错误输入被拒绝、报告中的探测字段与双峰判据、边沿不计为峰、ECS 区 c-对偶约定、默认值与网格一致性检查、因子化端到端），以及 `tests/test_spectrum_report.py` 中的强制 ASCII locale 回归测试。

## 编码报错

[encoding_fix.json](encoding_fix.json)：用 20260921 交付包原代码，在发布前把 `LC_CTYPE` 切到 C（[复现脚本](repro_old_code_midrun_ascii.py)），得到与超算日志完全相同的两处报错行。新代码下，原样重提和 `render_spectra.py` 两种方式都完成了发布：检查点未被改写，传播未重算。数值内核签名在 main、本分支和 20260921 包中一致。

## 三脉冲链路

| 检查 | 结果 | 证据 |
|---|---|---|
| 小网格三脉冲，强制 ASCII locale，第 120 步打断→续跑 | source、probe 均 complete；INDEX 含“第三束探测”列 | 本地运行 |
| 真实 Slurm 作业体（只模拟 `srun`）+ 冻结快照，外部 `LC_ALL=C PYTHONUTF8=0` | 两个数组任务 exit 0，complete | [slurm_probe_check.json](slurm_probe_check.json)，[脚本](check_slurm_probe.py) |
| GPU 验证模型：source + σ− probe（打断/续跑）+ σ+ probe，各 2 个 CPU 回放进程 | 全部 complete，最大线性残差 1.0e-10 | [model_runs](model_runs) |
| source 与 probe 在探测开始前是否逐步相同 | 各检查点范数/基态布居差 0，dt 相同 | 本地比较 |
| 因子化 vs 完整 TDSE（σ−、σ+，旧/新能区） | 产额 ≤3.1e-4，形状 L1 ≤1.1e-3，转移概率 7 位一致 | `factorized_vs_full_*` |
| CPU（不打断）vs GPU（打断后续跑），σ− probe 完整三脉冲 | 末态波函数相对差 1.1e-12，复谱振幅 6.6e-12，角积分谱 3.9e-12，离子转移振幅逐位相同 | [cpu_vs_gpu.json](cpu_vs_gpu.json) |

## 独立代码审计

对整个改动集另做一次只读审计。结论：因子化的物理与代数（伴随传播、c-对偶投影、时间网格、A=0 端点）无误，数值内核未改动。审计确认并已修复以下问题，每项都有测试覆盖：

| 问题 | 修复 |
|---|---|
| `apply_ionic_probe` 拿 run.json 中已补默认值的配置与原始探测输入逐项比较，导致合法输入被拒；脉冲按字典比较，显式写 `start: 0` 也被拒 | 新增 `python/run_defaults.py`，与 simulate_spectrum 共用同一补默认值函数（43 个现有输入的 run_id 全部不变）；脉冲按物理 `Pulse` 对象比较 |
| 未检查能量/角度网格；派生算例的编号不含 source 签名，换 source 会覆盖旧结果 | 要求网格一致（只忽略 plot_channels）；签名包含 source 签名 |
| `--ion-basis` 只给部分字段时会崩溃，或静默去掉库仑截断 | 合并到输入的离子基上；替换径向网格时必须显式给出 `cutoff_radii`（null 表示无截断） |
| 新的 GPU 峰值检查会挡住“原样重提以补发布”的恢复方式 | 已完成传播的任务跳过显存检查 |
| 独立包内 `make_probe_campaign.py` 找不到 smoke 模板 | 兼容仓库与独立包两种路径 |
| 对照 JSON 的来源路径与 INDEX 中完整/因子化算例的名称不易区分 | 记录每个算例的真实目录、方法、签名；INDEX 显示去掉签名后的完整名称 |
| `handoff.sh probe-smoke` 可能选中旧的 source 目录 | 由输入计算出确定的输出目录 |
| `probe_ion_design.py` 输出目录中已有其他设置的结果时会被静默复用 | 设置不同即报错 |
| 测试未锁定 c-对偶约定，也缺少因子化端到端测试 | 新增 ECS 区内束缚态测试（错误约定下对角元的模可达 1.13）和端到端测试 |

另外核实并记录：`commensurate()` 现在考虑 surface_stride；直接 LU 求解不报告迭代残差（记为 null）；混合规范的 source 被接受，理由写在脚本说明中。

## 生产输入的实测资源

[production_step_benchmark_l4.json](production_step_benchmark_l4.json) 与 [_l6.json](production_step_benchmark_l6.json)：真实生产网格与通道，本机 RTX 6000 Ada（FP64 较弱，与他人共享）。lmax 4 探测单步 2.67 s，峰值 21.0 GiB；lmax 6 单步 5.47 s，峰值 40.3 GiB（超出 GPU40G）。这些测量同时确定了 `GPU_peak_estimate_bytes` 的系数（峰值约为 FGMRES 基的 1.65–1.70 倍，已在三种规模上核对）。

## 探测设计

He+ 设计扫描见 [../ion_design/summary.md](../ion_design/summary.md)；精细模型预览见 [../refined_preview](../refined_preview)。
