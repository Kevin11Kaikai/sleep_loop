# T7 Spindle Envelope Burstiness (0501)

## 1. 讨论背景

本笔记总结了对 V7 中 T7 约束的讨论与统一理解。

T7 在代码中的目标是：
- 从 thalamic 信号中提取 spindle 频段振荡。
- 构造该振荡的包络（envelope）。
- 用包络变异系数衡量 burstiness（成簇程度）。

对应代码位置：
- models/s4_personalize_fig7_v7.py
- `compute_constraints_v7(...)` 内的 T7 部分。

---

## 2. T7 的信号处理流程

T7 实际执行的是标准窄带振荡分析链路：

1. `butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=fs, output='sos')`
2. `sosfiltfilt(sos, r_thal)`
3. `np.abs(hilbert(filtered))`
4. `sp_cv = envelope.std() / (envelope.mean() + 1e-12)`

可以写成：

- 原始时域信号：x(t) = r_thal(t)
- 窄带信号：x_sp(t) = BandPass{x(t)}
- 解析信号：z(t) = x_sp(t) + j * H[x_sp(t)]
- 包络：A(t) = |z(t)|
- 变异系数：CV = std(A) / mean(A)

---

## 3. 我们达成的一致理解

### 3.1 envelope 是不是 A(t)

是。在该语境下 envelope 就是包络函数 A(t) 的数值估计。

更精确地说：
- 不是原始全频信号的包络。
- 是先带通到 spindle 频段后，再通过 Hilbert 得到的瞬时振幅包络。
- 因而它代表的是 spindle 频段振荡强度随时间的外轮廓。

### 3.2 这一串操作是不是“为了得到 A(t)”

是。`butter -> sosfiltfilt -> hilbert -> abs` 的核心目的就是构造 A(t)，然后再用 A(t) 计算 burstiness 指标 CV。

### 3.3 这是否符合基本信号处理流程

是，完全符合。该流程属于经典的“窄带振荡包络分析”：
- 先限频（保证分析对象是 spindle 频段）。
- 再取解析信号幅值（得到 envelope）。
- 最后统计 envelope 的波动程度（CV）。

---

## 4. T7 的生理学与统计含义

T7 代码中判据：
- `t7 = sp_cv > SPINDLE_CV_MIN`
- 当前阈值 `SPINDLE_CV_MIN = 0.7`

解释：
- CV 小：包络较平，振幅变化小，像持续振荡，不够“成簇”。
- CV 大：包络起伏明显，出现高-低交替，更符合 spindle 的 burst 特征。

因此 T7 不只是检测“有无振荡”，而是检测“振荡是否以生理上合理的成簇方式出现”。

---

## 5. 与 T8/T12 的关系

- T7：看包络总体起伏（统计层面的 burstiness）。
- T8：看事件数量与持续时长（事件层面）。
- T12：看事件内频谱是否真有 sigma 主导峰（防假 spindle）。

三者互补：
- T7 防止“过平滑的连续振荡”。
- T8 防止“事件太少或持续时间不合理”。
- T12 防止“包络过阈值但事件内并非真实 sigma 振荡”。

---

## 6. 工程注意点（我们讨论中提到）

1. 边界效应
- `sosfiltfilt` 和 `hilbert` 在两端可能有边界伪影。
- 实践中常在后续分析中丢弃边缘样本（例如 PAC 部分已有类似处理）。

2. 带宽选择
- 带宽过宽会混入非 spindle 成分。
- 带宽过窄会让振幅估计对噪声更敏感。

3. 阈值可迁移性
- 0.7 是当前任务设定，不是所有数据集通用常数。
- 换数据、换模型、换采样条件时应重新校准。

---

## 7. 最终结论

T7 的本质是：
- 对 thalamic 信号提取 spindle 窄带振荡。
- 计算该窄带振荡的包络 A(t)。
- 用 CV 衡量 A(t) 的时间起伏是否呈现 burstiness。

所以我们讨论中的关键判断都是一致且正确的：
- envelope 在这里就是 A(t)（严格说是其数值估计）。
- 这串操作是经典、合理、可解释的信号处理流程。
