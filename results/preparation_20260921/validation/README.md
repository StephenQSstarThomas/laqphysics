# 本轮核验记录

以下分开报告软件一致性、物理控制组和收敛范围。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| Release 完整测试组 | 96 通过 | [日志](tests_release_final.log) |
| Debug 完整测试组（Fortran 边界与浮点检查开启） | 96 通过 | [日志](tests_debug_final.log) |
| 启用 CUDA 的专项组，含 CPU 对照 | 13 通过 | [日志](tests_cuda_final.log) |
| GPU 投影历史重积分，对照独立 CPU 逆序积分；同时改能量/角度网格和时间步采样 | 逐复振幅通过 | [日志](reintegration_cuda_test.log) |
| 实际圆偏振 CUDA：新在线算法 vs 原全历史算法 | 波函数相对差 6.89×10⁻¹⁴，复谱振幅相对差 2.59×10⁻¹² | [数据](online_cuda_regression.json) |
| 完整圆偏振 CUDA：CPU vs GPU 复谱累计 | 相对差 1.64×10⁻¹⁶ | [数据](gpu_accumulator_regression.json) |
| 细网格第 3737 步：串行 LU/CPU 谱积分 vs 并行迭代回放/GPU 谱积分 | 波函数相对差 0，部分复谱相对差 8.21×10⁻¹¹ | [含两端完整元数据](fine_grid_execution_regression.json) |
| 完整细化模型的谱求积 | 角度、能量、时间采样三项全部通过；总谱及指定符合谱同时验收 | [数据](fine_spectral_quadrature.json) · [日志](fine_quadrature.log) |
| 命令行完整链路 | 中断退出 95；原命令续跑后退出 0，命名谱/PNG/PDF/CSV 齐备；测试上限 500,000 字节，实际最大 228,474 字节 | [数据](cli_resume_validation.json) · [日志](cli_resume_validation.log) |
| 旧结果版本校验 | 拒绝另一数值内核的缓存，同时保留先前完整结果 | [数据](stale_kernel_validation.json) |
| 收敛审计入口 | 已知谱夹具通过；不同内核被拒绝 | [数据](audit_workflow_validation.json) |
| Slurm 接口 | shell 语法、模拟调度参数、MPI 重复任务拒绝、提交前日志目录检查通过 | [数据](slurm_contract.json) |
| 原始输入保护 | 两份 Markdown、参考 PDF、用户 ZIP 及其中脚本均未改动 | [校验](input_integrity.json) |

CPU 迭代离子求解与稀疏 LU 的小段基准见 [benchmark](refined_ionic_backend_benchmark.json)。单脉冲控制组的物理解释核验见 [control_factorization_diagnostic.json](../control_factorization_diagnostic.json)。

`fine_grid_execution_regression.json` 比较的是同一时刻的检查点，单独的串行运行有意停在这里，**不计为完整最终能谱**。完整细化运行的后续输出另在结果索引中报告。上述相对差没有通过平移能轴或拟合谱幅度来减小。

实际碰到的报告 JSON 序列化失败，保留在 `parallel_gpu_accumulator_smoke.log`；修复后只重新发布结果的成功日志为 `parallel_gpu_accumulator_smoke_resume.log`。物理传播结果没有因该报告错误重新计算或改写。

`production_acceptance_pending.json` 如实列出超算生产与求积计划的待办。软件测试、单脉冲控制和一个细化模型，均不能替代全部独立参数收敛。这里没有实际 Slurm 控制器提交记录。

后处理日志中的 NVML 告警对应本机监控库/内核版本不一致；CUDA 重积分实际完成并通过数值比较，详见各 `fine_spectrum_*.npz.json` 中的设备记录。本轮未修改服务器驱动。
