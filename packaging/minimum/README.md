# laqphysics：独立最小交接代码包

本包可脱离原仓库和 `/playpen` 路径运行，包含三维传播/在线提谱源码、全部 96 项测试、本轮 6 个完整算例的小型谱与图、输入、物理说明及 16 项 Slurm 生产输入。可从头重算，也可直接重画已有能谱。

**先看：[主图与输入索引](results/preparation_20260921/INDEX.md) · [已完成/待完成](results/preparation_20260921/SUMMARY.md) · [详细运行说明](docs/iteration20260921/使用说明.md)。**

## 第一步：环境和校验

需要 Linux、Bash、make、gfortran/OpenMP、Python 及下面的依赖。包内没有安装器、预编译平台库或 GPU 驱动。选择目标机器适用的 PyTorch CPU/CUDA 环境；已测版本记录在 `results/preparation_20260921/validation/validation_environment.json`。

```bash
# 在解压目录内；已有合适环境时跳过安装。
python3 -m pip install -r requirements-gpu.txt
./handoff.sh verify
./handoff.sh test
```

`test` 会在目标机编译 Release/Debug 两个 Fortran 库，再运行两套 96 项测试。新日志默认在 `local_runs/checks/`；原始交付结果保持不变。

也可直接执行 `./handoff.sh`，按中文菜单选择校验、测试、小算例、重画或估算。

## 第二步：走通传播—谱—图

```bash
./handoff.sh smoke          # CPU；小网格、23 个时间步的圆偏振安装检查
./handoff.sh smoke cuda:0   # 可选，在已分配的 GPU 上执行
./handoff.sh render         # 从包内小型主谱重画所有已交付谱图
```

小算例输入在 `configs/handoff_smoke.json`；它只验证运行链路，不用于判断物理峰位或制备目标。新输出在 `local_runs/smoke/`，其中 `INDEX.md` 链接实际输入、总谱/符合谱 PNG/PDF、CSV 和完成状态。

重画结果位于 `local_runs/replotted/INDEX.md`。主图采用绝对密度；条件归一谱单独标注。`HELIUM_PYTHON` 可指定 Python 路径，`HELIUM_OUTPUT_ROOT` 可替换上述新输出根目录。

## 第三步：交给 Slurm

```bash
export HELIUM_OUTPUT_ROOT=/persistent/scratch/laqphysics
export HELIUM_PYTHON=/path/to/python
./handoff.sh estimate
./handoff.sh submit gpu                         # 默认 reference：1 GPU + 16 核
# 或 ./handoff.sh submit cpu                    # 64 核
# 扩展到整批之前，先确认单项运行及资源配置：
./handoff.sh submit gpu --array=0-15%4
```

GPU 分区沿用提供的 `GPU40G`；其他 partition/account 等可通过 sbatch 参数覆盖。本机测试覆盖脚本和数值入口，没有目标集群的真实提交记录。

一个任务推进一份双电子波函数，主进程使用 Torch/Fortran 线程；生产输入另有 4 个 CPU 离子回放进程，从同一 CPU 配额中扣除。独立参数用数组并行。退出码 95 表示检查点尚未完成谱；保持配置、输出和 tag，使用原数值快照重提即可。数值快照功能支持无 `.git` 的解压目录。

## 交接范围

- 细化模型已完成：总谱 0.3/0.6 峰高比 4.01582；2p+1 符合谱峰高比 7.99412；绝对离子转移概率 99.85126%。
- Release/Debug 各 96 项、CUDA 专项 13 项，以及本地三项谱求积检查通过。机器可读证据位于 `results/preparation_20260921/validation/`。
- **完整生产收敛与后续长探测仍待执行。** 程序测试通过不代表全部空间、时间、边界和离子通道已收敛；延迟离子操作也不自动保证旧电子能谱劈裂。
- 主 NPZ/CSV 足够重画总谱和所有已记录离子通道的能谱。此最小包不含旧波函数检查点、全角复振幅和边界历史，不能直接接续原服务器的大计算；从头运行不需要那些文件。
- 导出元数据中的原服务器绝对路径、归档源码和历史检查脚本是追溯记录。当前运行入口不依赖那些路径；使用本页的 `handoff.sh`。
- 新任务默认不写原始 `flux.npy`，单文件默认上限 4 GB、硬上限十进制 5 GB。生产 16 项数组估算合计约 51.8 GB，建议预留 100 GB scratch。

`bundle_manifest.json` 记录来源提交和每个原始成员的 SHA-256；压缩包旁的 `.sha256` 文件用于传输校验。新增编译和运行文件不影响原包内容校验。
