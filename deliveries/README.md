# 独立最小交接代码包

**[下载 laqphysics-minimum-20260921.tar.gz](laqphysics-minimum-20260921.tar.gz)**（5,361,488 字节，约 5.4 MB） · [SHA-256 校验文件](laqphysics-minimum-20260921.tar.gz.sha256)

包含源码、编译/依赖说明、全部 96 项测试、本轮 6 个完整算例的小型能谱和图、输入、算法说明、16 项生产配置及 CPU/GPU Slurm 入口。`handoff.sh` 提供中文菜单及命令行入口。

```bash
sha256sum -c laqphysics-minimum-20260921.tar.gz.sha256
tar -xzf laqphysics-minimum-20260921.tar.gz
cd laqphysics-minimum-20260921
./handoff.sh verify
# 依赖和编译器配置见包内 README.md；已有环境可直接：
./handoff.sh test
./handoff.sh smoke
./handoff.sh render
```

已在不含 `.git` 的独立解压目录中实际完成：

- 重新编译，Release/Debug 各 96 项通过。
- CPU 小算例从传播到总谱/符合谱、PNG/PDF/CSV 完整运行。
- 从包内数据重画全部 6 组已交付能谱。
- 生产资源估算、无 Git 的代码快照正常。
- 用模拟的 srun 启动真实冻结代码，Slurm 作业内的编译、快照及小算例运行通过；未宣称真实集群提交。
- 所有操作后，原包成员哈希仍一致。正式包的所有代码/数据成员与受测副本逐字节相同，正式版本的来源清单和无 Git 快照也重新核验。

[机器可读验收](minimum_bundle_validation.json) · [实际日志](validation_logs)

源码版本：[ff81765](https://github.com/StephenQSstarThomas/laqphysics/commit/ff81765d7d2c0e3588338ab162af08ae691a1c1e)。完整校验值：

```text
3cf8bf6e2bd29c5a1bcac5976f123aa7c4892b3b0a0e44250c5da2aa27febb37
```

自洽范围是源码、参数、小型结果和运行入口；Python 依赖、编译器和驱动由目标机提供。大波函数检查点、全角复振幅及边界历史未装入本包，可从头重算，不能直接续跑原服务器的历史。完整生产收敛与后续长探测仍待执行。
