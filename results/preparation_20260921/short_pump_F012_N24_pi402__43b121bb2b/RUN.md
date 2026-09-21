# short_pump_F012_N24_pi402__43b121bb2b

[输入参数](input.json)（签名 `43b121bb2bef6e5b93dc94170ae51b54ea7c0cc0d131868cb78f9dd5805c49fd`）。

[主谱数据 NPZ](short_pump_F012_N24_pi402__43b121bb2b__p1w1p2F0p12N24z_p2w1p5F0p0100179N402spt240__SI.npz) · [CSV](short_pump_F012_N24_pi402__43b121bb2b__p1w1p2F0p12N24z_p2w1p5F0p0100179N402spt240__SI.csv)。
[主图 PNG](figures/short_pump_F012_N24_pi402__43b121bb2b__p1w1p2F0p12N24z_p2w1p5F0p0100179N402spt240__SI.png) · [PDF](figures/short_pump_F012_N24_pi402__43b121bb2b__p1w1p2F0p12N24z_p2w1p5F0p0100179N402spt240__SI.pdf)。

主谱 NPZ 存能量、通道积分谱与总谱；完整复振幅和角分布保存在 spectrum.npz。

[归一化条件谱](figures/short_pump_F012_N24_pi402__43b121bb2b__p1w1p2F0p12N24z_p2w1p5F0p0100179N402spt240__SI__conditional.png)：各通道在计算能窗内积分归一，仅比较形状，不能据此判断峰的绝对强弱。

图中各曲线使用同一绝对概率密度刻度，未按各自峰高归一化。总谱仅求和已记录的束缚离子通道。

| 脉冲 | 偏振 | ω / a.u. | F / a.u. | 周期数 | 起点 / a.u. | 支撑宽度 / fs |
|---|---|---:|---:|---:|---:|---:|
| 1 | z | 1.2 | 0.12 | 24 | 0 | 3.0397 |
| 2 | sigma+ | 1.5 | 0.010017896 | 402 | 240 | 40.7314 |

`F` 为线偏振峰值电场；圆偏振两个分量各为 F/√2，具有相同周期平均强度。包络为 sin²。
`dt`、径向网格、lmax、total_Lmax 和离子通道均在 input.json；这些是待继续收敛的数值参数。

0.3/0.6 峰高比：2.296330008877113；能区积分产额比：25.153217695748868（None 表示该能区未计算）。
旧能区 2p+1 条件份额：0.9999515882781161。详细通道与准备态离子转移见 observables.json。
从已准备的离子 1s 到最终 2p+1 的绝对转移概率：0.9992817981710607；与上面的条件份额分别报告。
是否达到本次制备目标：False。该标志不等于完整数值收敛。

存储模式：spectrum；文件上限：4000000000 字节。
资源估算：resource_estimate.json；预计持久数组 93841872 字节。

## 轻量交接范围

此目录保留可重画总谱与符合能谱的主 NPZ/CSV、图和完整输入。大数组、全角复振幅与恢复检查点仍在原计算目录：

`/playpen1/shiqiu/laqphysics-data/iteration20260921/runs/short_pump_F012_N24_pi402__43b121bb2b`

原始文件清单、大小和数据摘要见 export.json；本目录不能直接恢复传播。
