# selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837

[输入参数](input.json)（签名 `b9f521f837d932ea5d4e457154dda73063364859326bdd36ab5507d34c67523d`）。

[主谱数据 NPZ](selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837__p1w1p2F0p08N48z_p2w1p5F0p0200358N201spt360__SI.npz) · [CSV](selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837__p1w1p2F0p08N48z_p2w1p5F0p0200358N201spt360__SI.csv)。
[主图 PNG](figures/selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837__p1w1p2F0p08N48z_p2w1p5F0p0200358N201spt360__SI.png) · [PDF](figures/selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837__p1w1p2F0p08N48z_p2w1p5F0p0200358N201spt360__SI.pdf)。

主谱 NPZ 存能量、通道积分谱与总谱；完整复振幅和角分布保存在 spectrum.npz。

[归一化条件谱](figures/selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837__p1w1p2F0p08N48z_p2w1p5F0p0200358N201spt360__SI__conditional.png)：各通道在计算能窗内积分归一，仅比较形状，不能据此判断峰的绝对强弱。

图中各曲线使用同一绝对概率密度刻度，未按各自峰高归一化。总谱仅求和已记录的束缚离子通道。

| 脉冲 | 偏振 | ω / a.u. | F / a.u. | 周期数 | 起点 / a.u. | 支撑宽度 / fs |
|---|---|---:|---:|---:|---:|---:|
| 1 | z | 1.2 | 0.08 | 48 | 0 | 6.0793 |
| 2 | sigma+ | 1.5 | 0.020035791 | 201 | 360 | 20.3657 |

`F` 为线偏振峰值电场；圆偏振两个分量各为 F/√2，具有相同周期平均强度。包络为 sin²。
`dt`、径向网格、lmax、total_Lmax 和离子通道均在 input.json；这些是待继续收敛的数值参数。

0.3/0.6 峰高比：4.015816283860522；能区积分产额比：10.964459853079418（None 表示该能区未计算）。
旧能区 2p+1 条件份额：0.9998404635954417。详细通道与准备态离子转移见 observables.json。
从已准备的离子 1s 到最终 2p+1 的绝对转移概率：0.9985125609176596；与上面的条件份额分别报告。
是否达到本次制备目标：True。该标志不等于完整数值收敛。

存储模式：projected；文件上限：4000000000 字节。
资源估算：resource_estimate.json；预计持久数组 759179328 字节。

## 轻量交接范围

此目录保留可重画总谱与符合能谱的主 NPZ/CSV、图和完整输入。大数组、全角复振幅与恢复检查点仍在原计算目录：

`/playpen1/shiqiu/laqphysics-data/iteration20260921/runs/selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837`

原始文件清单、大小和数据摘要见 export.json；本目录不能直接恢复传播。
