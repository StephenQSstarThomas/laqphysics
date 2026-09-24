# 单电离谱结果索引

每行是一份独立输入；完整参数在 input.json，参数含义和结果说明在 RUN.md。
所有图采用绝对概率密度。同一配置重提会恢复检查点，不会新建一个伪重复结果。

| 模拟与输入 | 状态 | 主图 | 0.3/0.6 峰高比 | 能区产额比 | 旧能区 2p+1 份额 | 第三束探测 |
|---|---|---|---:|---:|---:|---|
| [probe](probe__7a4353cc24/RUN.md) · [输入](probe__7a4353cc24/input.json) | complete | [单电离谱](probe__7a4353cc24/figures/probe__7a4353cc24__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4smt230p6__SI.png) | 0.2614 | 0.37064 | 0.95697 | 旧能区极大值 1；[3, 2, 0] 0.0145 |
| [probe_sigma_plus](probe_sigma_plus__5ef68b4112/RUN.md) · [输入](probe_sigma_plus__5ef68b4112/input.json) | complete | [单电离谱](probe_sigma_plus__5ef68b4112/figures/probe_sigma_plus__5ef68b4112__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4spt230p6__SI.png) | 0.26113 | 0.37026 | 0.89043 | 旧能区极大值 1；[3, 2, 2] 0.085 |
| [source](source__64e48768c3/RUN.md) · [输入](source__64e48768c3/input.json) | complete | [单电离谱](source__64e48768c3/figures/source__64e48768c3__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70__SI.png) | 0.26147 | 0.37066 | 0.97428 | — |
