# 独立最小交接代码包

## 当前版本：laqphysics-minimum-20260924（第三束探测 + 编码修复）

**[下载 laqphysics-minimum-20260924.tar.gz](laqphysics-minimum-20260924.tar.gz)**（11,832,131 字节，约 11.8 MB）· [SHA-256 校验文件](laqphysics-minimum-20260924.tar.gz.sha256)

在 20260921 版的基础上，本版新增：

- 修复超算报告阶段的 `UnicodeEncodeError: 'ascii'`。数值内核未改；已完成传播的任务无需重算：用新包原样重提，或执行 `PYTHONUTF8=1 python scripts/render_spectra.py --out-root <输出根>`。
- 第三束（双光子）探测的输入、生产计划、He+ 设计扫描、快速离子因子化路径及其验证证据。
- 新入口 `./handoff.sh probe-smoke [cpu|cuda:0]`：小网格上依次运行两脉冲 source、完整三脉冲 TDSE 和离子因子化。

```bash
sha256sum -c laqphysics-minimum-20260924.tar.gz.sha256
tar -xzf laqphysics-minimum-20260924.tar.gz
cd laqphysics-minimum-20260924
./handoff.sh verify
./handoff.sh test          # 编译 Release/Debug 并运行两套 107 项测试
./handoff.sh probe-smoke   # 三脉冲小算例
```

说明见包内 `docs/probe20260924/第三束探测.md`；三脉冲生产提交用 `HELIUM_PLAN=configs/probe_20260924/production/plan.json`。

在不含 `.git` 的独立解压目录中，外部环境设为 `LC_ALL=C PYTHONUTF8=0`（默认文本编码为 ASCII，与出错节点相同），依次实跑了以下 12 项，全部通过：

- 校验包；
- 编译并运行 Release/Debug 各 107 项测试；
- 两脉冲 CPU 小算例；
- 三脉冲小算例（完整 TDSE 与因子化），CPU 和 GPU 各一次；
- 重画全部已交付谱；
- 两脉冲与三脉冲的生产资源估算；
- 重新生成三脉冲输入，结果与包内逐字节相同；
- 无 Git 的代码快照；
- 用冻结快照运行真实 Slurm 作业体（只模拟 `srun`）；
- 运行后再次校验包，成员未被改动。

[机器可读验收](minimum_bundle_validation_20260924.json) · [实际日志](validation_logs_20260924)。打包可逐字节复现（两次构建 SHA-256 相同）。

源码版本：见包内 `bundle_manifest.json` 的 `source_commit`。完整校验值：

```text
3c82289b28ef93d0d48f724257fa8bd0f58e638900798c39c57f0b425ab540e0
```

完整生产收敛与三脉冲生产任务尚未运行；小算例只验证链路和方法，不是物理结论。

## 上一版：laqphysics-minimum-20260921

[laqphysics-minimum-20260921.tar.gz](laqphysics-minimum-20260921.tar.gz)（5,361,488 字节）· [SHA-256](laqphysics-minimum-20260921.tar.gz.sha256) · [验收](minimum_bundle_validation.json) · [日志](validation_logs)。源码版本 [ff81765](https://github.com/StephenQSstarThomas/laqphysics/commit/ff81765d7d2c0e3588338ab162af08ae691a1c1e)，校验值 `3cf8bf6e2bd29c5a1bcac5976f123aa7c4892b3b0a0e44250c5da2aa27febb37`。该版本在 ASCII locale 的节点上会在报告阶段失败；请改用 20260924 版。
