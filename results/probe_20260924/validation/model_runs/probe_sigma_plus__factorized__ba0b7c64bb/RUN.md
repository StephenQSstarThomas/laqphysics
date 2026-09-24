# probe_sigma_plus__factorized__ba0b7c64bb

[输入参数](input.json)（签名 `ba0b7c64bb6abb687afed16fef295b6f9bec7aff8b35dfc6f0ffff20387147b7`）。

[主谱数据 NPZ](probe_sigma_plus__factorized__ba0b7c64bb__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4spt230p6__SI.npz) · [CSV](probe_sigma_plus__factorized__ba0b7c64bb__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4spt230p6__SI.csv)。
[主图 PNG](figures/probe_sigma_plus__factorized__ba0b7c64bb__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4spt230p6__SI.png) · [PDF](figures/probe_sigma_plus__factorized__ba0b7c64bb__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4spt230p6__SI.pdf)。

主谱 NPZ 存能量、通道积分谱与总谱；完整复振幅和角分布保存在 spectrum.npz。

[归一化条件谱](figures/probe_sigma_plus__factorized__ba0b7c64bb__p1w1p2F0p08N12z_p2w1p5F0p1678N24spt70_p3w0p3F0p01N4spt230p6__SI__conditional.png)：各通道在计算能窗内积分归一，仅比较形状，不能据此判断峰的绝对强弱。

图中各曲线使用同一绝对概率密度刻度，未按各自峰高归一化。总谱仅求和已记录的束缚离子通道。

| 脉冲 | 作用 | 偏振 | ω / a.u. | F / a.u. | 周期数 | 起点 / a.u. | 支撑宽度 / fs |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | ionize | z | 1.2 | 0.08 | 12 | 0 | 1.5198 |
| 2 | prepare | sigma+ | 1.5 | 0.16779975 | 24 | 70 | 2.4317 |
| 3 | probe | sigma+ | 0.3 | 0.01 | 4 | 230.6 | 2.0264 |

`F` 为线偏振峰值电场；圆偏振两个分量各为 F/√2，具有相同周期平均强度。包络为 sin²。
`dt`、径向网格、lmax、total_Lmax 和离子通道均在 input.json；这些是待继续收敛的数值参数。

0.3/0.6 峰高比：0.26110539912899666；能区积分产额比：0.37015319376626415（None 表示该能区未计算）。
旧能区 2p+1 条件份额：0.8907273115959567。详细通道与准备态离子转移见 observables.json。
从已准备的离子 1s 到最终 2p+1 的绝对转移概率：0.8915952806031887；与上面的条件份额分别报告。
是否达到本次制备目标：None。该标志不等于完整数值收敛。

第三束（探测）脉冲：第 [3] 束；目标离子通道 [[3, 2, 2]]。制备目标对探测算例不适用。
旧能区各离子通道份额：{'[2, 1, 1]': 0.8907273115959567, '[3, 2, 2]': 0.08498912189308754, '[3, 1, 1]': 0.011998895075342117, '[1, 0, 0]': 0.010037222655724381, '[3, 0, 0]': 0.0013289213668081121, '[2, 0, 0]': 0.0003927679544880096}。
旧能区总谱显著极大值 1 个（判据：突出度≥窗口最大值 5%）；两最高峰间距：None。
从 1s 离子（ionic_transfer_times）到最终目标通道的绝对转移概率：{'[2, 1, 1]': 0.8915952806031887, '[3, 2, 2]': 0.08385883994116859}。
判断“劈裂”须与同一网格、同一结束时刻的无探测对照比较；电离电子离开后仅作用于离子的探测不改变无条件总谱（见 docs 中的说明）。

存储模式：derived；文件上限：4000000000 字节。
资源估算：resource_estimate.json；预计持久数组 unknown 字节。
