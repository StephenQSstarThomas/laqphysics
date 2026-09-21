# 氦原子残余离子共振与单电离：计算与核验

实体位置：`/playpen/shiqiu/task-jiukongdian`。原 `/home/shiqiu/task-jiukongdian` 保留软链接入口；迁移校验见 [migration.json](migration.json)。

## 先读这些交付文档

- [独立最小交接代码包（约 5.4 MB）](deliveries/README.md)：无需 Git 和原服务器路径，含中文菜单、编译测试、谱图重画与 Slurm 入口，已独立解压实跑核验。
- [单电离谱运行手册（2026-09-21）](docs/iteration20260921/使用说明.md)：每次任务的输入、命名谱图、CPU/GPU Slurm、5 GB 文件上限与在线提谱。
- [本轮结果索引](results/preparation_20260921/INDEX.md)与[结论及剩余工作](results/preparation_20260921/SUMMARY.md)：总谱与离子符合谱、0.3 主峰制备方案及细化核验。
- [前轮最小验证交接](docs/convergence/最小验证交接.md)：新增算法的直接交叉验证、高精度长算例检查、可复现交接及合作者的 Slurm 待办。
- [完整收敛运行协议](docs/convergence/运行协议.md)：混合规范推导、通用圆偏振角基、缓存和真实残差核验。
- [专家反馈后的研究与实现](docs/followup/专家反馈与研究.md)：脉冲带宽、实际圆偏振/多脉冲计算、含连续态的离子控制、高精度算法与新发现的投影问题。
- [后续交付与复现](docs/followup/交付与复现.md)：本轮完成状态、复现命令、生产收敛任务及资源限制。
- [交付摘要](docs/交付摘要.md)：实际完成项、定量结果、修复的问题及剩余生产验收。
- [物理与算法核验](docs/物理与算法核验.md)：可观测量、选择定则、ESSS/TDSE 方程、边界选择与适用范围。
- [数值结果](docs/数值结果.md)：从实际完成的计算自动生成的基态、谱和误差表。
- [调试记录](docs/调试记录.md)：真实失败、定位、修复、逐模块快照及尚未通过的生产级验收。
- [文献核查与疑点](docs/文献核查与疑点.md)：文献适用范围及单独审计的相位记号疑点。
- [论文初稿](docs/论文初稿.md)：方法与验证工作稿，不把未收敛算例写成研究结论。

原计划的两项前提需要修正：电离结束后仅对离子的局域保迹操作不改变电子无条件谱；电偶极近似下两个光子的 2p→3d 跃迁受宇称禁止。具体推导与替代方案已写入报告。

## 实现范围

| 组件 | 实现与证据 |
|---|---|
| ESSS | 文献解析积分、独立微分方程、有限能带回耦、复数/能量依赖源耦合 |
| 原子角代数 | 精确 Wigner/Gaunt 矩阵、氢样径向偶极矩、单/双光子选择定则 |
| 1D 双电子 TDSE | Fortran/OpenMP FFT、独立基态求解、真实离子反向传播的 tSURFF、CAP |
| 1D 独立边界对照 | 双侧 FE-DVR＋irECS；Fortran 算符与稀疏矩阵对照；离子矩阵指数分裂 |
| 3D 双电子 TDSE | FE-DVR、乘积/耦合球谐基、Fortran/OpenMP 与可选 complex128 CUDA、Arnoldi / CF4–Padé、tSURFF + irECS |
| 圆偏振、多脉冲 | 双手性三维实跑、延迟线/圆偏振控制、独立控制脉冲、准备态 He⁺ 含连续态控制；完整生产收敛按报告逐项验收 |
| 退相干 | 有限连续谱密度矩阵 Lindblad 扫描、迹/正性检查、条件 negativity、局域操作对照 |
| 可复现性 | Debug/Release 测试、每步模块快照、三维检查点及恢复测试、参数文件和结果元数据 |

三维长共振和多脉冲的**生产级**角动量、径向、库仑截断与时间收敛须依照报告继续验收。已有的小规模实跑结果见数值结果表；Slurm 配置的存在不代表该作业已经运行。这里不包含单通道约化 R-matrix 的伪对比，也没有将 CAP 称为 irECS。

## 编译与验证

基础依赖：gfortran（支持 OpenMP）和 `requirements.txt` 中的 Python 包。原 Fortran 路径无需 GPU。后续 CF4–Padé 及 Torch CPU/CUDA 路径另需 `requirements-gpu.txt`；安装适配目标机器的 PyTorch CPU/CUDA 构建。

```bash
cd /playpen/shiqiu/task-jiukongdian
python -m pip install -r requirements.txt   # 缺少依赖时执行
make all debug
make test
HELIUM_LIBRARY=libhelium_debug.so OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  HELIUM_LIBRARY=libhelium_debug.so python scripts/debug_modules.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/benchmark_boundary.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/analyze_esss.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/decoherence_scan.py
```

测试和基准日志已随结果保存，无需重新计算才能阅读报告。运行时应限制 BLAS 线程，防止其与 OpenMP 叠加。

## 本地单电离谱任务

推荐使用以下入口；完整任务会生成总谱、指定离子终态符合谱、PNG/PDF/CSV，重复命令自动恢复：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python scripts/simulate_spectrum.py \
  --config configs/preparation_20260921/longer_pump_F008_N48_pi201.json \
  --out-root /persistent/scratch/helium-preparation --device cuda:0 --cpu-threads 2
```

下列底层入口保留给旧算例和独立诊断。三维底层恢复仍须显式提供 `--resume`。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python scripts/run1d.py --config configs/1d_dt0005.json --out results/my-1d

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  python -m run3d --config configs/3d_absorber_refined.json --out results/my-3d

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  python -m run3d --out results/my-3d --extract
```

有场计算的配置标注很重要：`1d_local.json` 的 dt=0.04 是保留的粗时间步排错算例；用细化配置进行物理比较。`3d_short.json` 的低阶吸收尾元也是粗基准。原参数不被悄悄覆盖，以便重现误差。

独立的一维 FE-DVR/irECS 对照可运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  python -m line_dvr --out results/my-1d-dvr --cycles 60 --dt .01
```

它把两个单电子离子哈密顿量的矩阵指数与电子间势作 Strang 分裂；局部误差已与完整双电子矩阵指数/Arnoldi 对照。也可用 `--integrator arnoldi` 作更昂贵的传播器对照。一维大外区 FFT 与该方法一起用于定位边界误差。

三维恢复与短程诊断：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  python -m run3d --config configs/3d_absorber_refined.json \
  --out results/my-resume-test --max-steps 20

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  python -m run3d --config configs/3d_absorber_refined.json \
  --out results/my-resume-test --resume
```

SIGUSR1/SIGTERM 会请求在下一个表面采样步保存三维检查点并退出；再次运行同一配置恢复。改变物理或网格配置会被 SHA-256 参数签名拒绝。一维版本保存表面历史和最终波函数，**不支持中途恢复**。

## Slurm 交付

本机没有可用的 `sbatch`；本轮完成了 shell 语法、模拟调度参数检查及真实本地 CPU/CUDA 数值测试。超算入口按用户已跑通的 `GPU40G` 模板修改，GPU 每任务 **1 卡 + 16 核**，CPU 每任务 **64 核**，详见 [运行手册](docs/iteration20260921/使用说明.md)。

在目标机器先执行 `make all debug`，使用其编译器构建动态库。

```bash
export HELIUM_REPO=/path/on/cluster/laqphysics
export HELIUM_PYTHON=/path/to/python
export HELIUM_OUTPUT_ROOT=/persistent/scratch/helium-preparation

scripts/submit_spectra.sh gpu          # 默认 production/reference 单个任务
scripts/submit_spectra.sh cpu          # 同一物理输入，纯 CPU
# 全部 16 项生产/控制输入，最多四个一卡任务同时运行：
scripts/submit_spectra.sh gpu --array=0-15%4
```

三维每项任务推进一份双电子波函数，使用线程和可选 CPU 离子回放进程。独立参数通过作业数组并行，不使用多节点 MPI。保持输入、输出根目录和 tag 不变可恢复；跨代码修改续跑时必须使用原 `HELIUM_SNAPSHOT`。完整谱及求积/收敛验收步骤见运行手册。

旧一维入口保留：

```bash
mkdir -p logs
HELIUM_CONFIG=configs/1d_production.json \
HELIUM_OUT=/path/to/scratch/helium-1d \
sbatch scripts/helium1d.slurm
```

三维提谱配置支持 `ionic_channels: [[n,l,m], ...]`，例如加入 `[3,1,-1]` 检验修正后的 3p 末态，l 必须不超过 lmax。`scripts/ionic_controls.py` 另提供包含完整 m 子态的束缚基控制算例；其中没有连续态损失，结果只用于验证共振与选择定则。

资源检查：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/resources.py
```

基础配置资源见 `results/resources.json`；新两脉冲生产输入的逐项估算见 `configs/preparation_20260921/production/resource_estimates.json`。新谱任务直接累计复振幅，可选保留缩小的离子投影历史，默认单文件 4 GB、硬上限十进制 5 GB。完整高 l、多半径及其他脉冲扫描仍需要更多总机时；配置和脚本的存在不代表已完成数值收敛。

## 专家反馈后的计算入口

```bash
# 按配置签名跳过已完成项、从检查点恢复；圆偏振使用全部 M。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python scripts/followup_campaign.py --group circular --device cuda:0
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python scripts/followup_campaign.py --group pump-probe --device cuda:0

# 只使用 CPU 也能运行新的隐式传播器。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python scripts/ionic_tdse_campaign.py --auto-resume --device cpu \
  --config configs/followup/ion_2p_counter.json --out results/followup/ion_2p_counter

# 前轮线偏振参考与七个细化输入，现使用在线提谱和命名输出。
# 先设置上文 HELIUM_OUTPUT_ROOT；旧结果仍可用 audit_followup_production.py 审计。
mkdir -p logs
sbatch scripts/followup_production.slurm
python scripts/audit_spectra_campaign.py --plan configs/followup/production_plan.json \
  --out-root "$HELIUM_OUTPUT_ROOT" --tag production

# 由实际结果再生表格和科学图。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/analyze_followup.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/analyze_ionic_followup.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/followup_entanglement.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/asymptotic_delay_scan.py
```

前轮图在 `results/followup/figures/`，原图和粗基准保留。本次主谱见 `results/preparation_20260921/INDEX.md`。另交付 `configs/followup_multipulse/` 的 10 项多脉冲细化候选；独立输出目录、Slurm 数组及逐能窗验收命令见后续交付文档。`production_acceptance.json` 中的待运行/未通过项不能当成已经收敛；阈值对齐仅用于诊断，验收采用绝对能量坐标。

## 结果、图与复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/summarize_results.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/calibrate3d.py
```

后一个脚本由独立弱脉冲 TDSE 标定三维 ESSS 源强度，并在长共振小规模谱存在时作比较；不是重新拟合该强场目标谱。

`results/esss/` 包含文献模型谱、角分辨模型图和 GIF；`results/figures/` 包含实际 TDSE 图。各 TDSE 输出目录中的 `run.json` 记录实际 dt、基态、网格和完成状态，`spectrum.npz` 保存复振幅及谱，三维另有 `history.json`、`flux.npy` 和 `checkpoint.npz`。不要把 irECS 区的系数平方和当成物理概率。

源码在 `src/`、`python/`；自动检查在 `tests/`。论文全文的本地可检索副本在 `docs/literature/`，公式解释仍以原 PDF 为准。
