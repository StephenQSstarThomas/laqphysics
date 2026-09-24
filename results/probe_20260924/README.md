# 第三束（双光子）探测：结果索引（2026-09-24）

- [说明文档](../../docs/probe20260924/第三束探测.md)：输入格式、物理设计、生产提交、快速离子因子化、编码修复与已完成任务的恢复方法。
- [ion_design](ion_design/summary.md)：制备好的 He+（2p+1）单独受探测脉冲作用的终态布居，覆盖 ω=5/36 与 0.14、σ±、场强/周期，以及 lmax 4–8、径向盒与库仑截断的收敛。
- [refined_preview](refined_preview)：已完成的两脉冲精细算例经收敛 He+ 探测映射得到的三脉冲谱（σ−、σ+），附与两脉冲 source 的对照。数值仅作预览，不是生产收敛结果。
- [validation](validation/README.md)：测试、编码复现与恢复、三脉冲链路、因子化与完整 TDSE 对照、CPU/GPU 对照、生产输入资源实测。

输入在 [configs/probe_20260924](../../configs/probe_20260924)：`production/`（生产与可选输入，以及 plan.json、资源估算）、`validation/`（因子化验证小模型）、`smoke/`（安装检查）、`ion_basis_converged.json`（快速路径所用的收敛 He+ 基）。
