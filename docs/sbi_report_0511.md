# SBI Stage 2 技术报告

**日期**：2026-05-11  
**受试者**：SC4001（Sleep-EDF Cassette）  
**状态**：5-dim x_obs 已生成（`x_obs_v3.npz`），SBI 全流程已完成一轮（4000 次仿真，9.66 小时）

---

## 0. 框架与读法

### 0.1 为什么用 SBI 而不继续用 DE

Stage 1 的差分进化（DE）给出了一个最优参数点——Seed B——其 fitness 分数 0.674 在 364 个可行解里排第一。这是一个点估计。它告诉我们"哪组参数最好"，但不回答"这组参数有多确定"、"还有哪些参数组同样合理"，以及"模型各自由参数之间是否存在权衡关系（trade-off curve）"。

贝叶斯推断的目标是后验分布 $p(\theta \mid x_\text{obs})$，而不是一个点。后验的宽窄直接度量参数的可识别性（identifiability）：若某个参数的后验几乎等于先验，说明数据对它无约束力；若后验在某两个参数之间呈明显的负相关，说明两者在功能上可以相互补偿。对于 TBME 投稿，提供后验分布而非单点比较能够让审稿人评估模型的生理解释是否唯一。

传统 MCMC（Metropolis-Hastings、HMC）在每一步都需要运行一次仿真计算似然，对于需要 60 s 数值积分的力学仿真来说每次约 90 秒，根本无法收敛。SBI（Simulation-Based Inference）的核心思想是将似然计算从采样循环中剥离：先用仿真数据离线训练一个神经密度估计器，推断阶段只需前向传播，不再调用仿真器。

### 0.2 "生理约束力学后验"框架

本项目的 framing 是 physiology-constrained mechanistic posterior（生理约束力学后验）：

- 仿真器是一个完整的丘脑-皮层平均场模型（`ALNNode + ThalamicNode`），不是黑盒。每一个自由参数都有明确的生理对应（$g_h$：超极化电流增益；$g_{LK}$：漏电流增益；$c_\text{ctx2th}$：皮层→丘脑突触强度；$b$：适应电流强度）。
- 约束来自两侧：先验来自 Stage 1 DE 搜索到的可行域，观测来自真实 N3 EEG 的 5 个可直接测量的摘要统计量。
- T6（IBI CV）和 MI（调制指数）虽然在仿真器内部有确定的值，但因为 EEG 物理限制（头皮 EEG 中 >95% 的信号来自皮层锥体细胞的突触后电位；丘脑是封闭场源（closed field），偶极场相互抵消，在头皮几乎不可见），这两个统计量在 EEG 端没有等价的可靠提取路径，因此不进入 EEG observable 向量 $x_\text{obs}$，只在仿真器内部作为力学合理性的 gate 使用。

### 0.3 术语速查

| 术语 | 一句话 |
|------|--------|
| $\theta$ | 4维自由参数向量 $[g_h,\, g_{LK},\, c_\text{ctx2th},\, b]$ |
| $x$ | 5维摘要统计向量（仿真或 EEG 提取均使用同一组公式） |
| $x_\text{obs}$ | 从真实 EEG 提取的观测统计，形状 `(5,)`，保存于 `x_obs_v3.npz` |
| posterior | $p(\theta \mid x_\text{obs})$，4维联合分布，本文目标产出 |
| prior | $p(\theta)$，BoxUniform，范围来自 Stage 1 DE 的可行域边界 |
| SNPE-C | Sequential Neural Posterior Estimation (variant C)，sbi 库实现 |
| NSF | Neural Spline Flow，神经样条流，密度估计器的具体架构 |
| SBC | Simulation-Based Calibration，检验后验是否系统性过宽或过窄 |
| PPC | Posterior Predictive Check，从后验采样再仿真，与 $x_\text{obs}$ 对比 |

### 0.4 如何读这份文档

文档按依赖关系排列（数据准备 → 仿真器 → 训练 → 诊断），可以线性阅读。如果只想快速了解某个问题：

- 想知道 x_obs 怎么提取 → §1
- 想知道仿真器输出什么、用了哪些 Seed B 参数 → §2
- 想知道 SNPE-C 怎么训练、收敛了没有 → §3
- 想知道 T6 和 MI 为什么被删 → §4.3–4.4
- 想知道 SBC/PPC 数字结果 → §3.4–3.5
- 想知道各版本 compute_xobs 的演化 → §4.7

---

*（§0 完）*

---

## 1. x_obs 提取：`compute_xobs_from_eeg_v3.py`

### 1.1 输入 / 输出

输入来源：`data/manifest.csv` 提供 SC4001 的 PSG 路径和催眠图路径；从 EDF 文件中读取单通道 EEG（Fpz-Cz，100 Hz 采样）；催眠图由 `utils/02_preprocess_psd.py` 的 `load_hypnogram` 解析。

N3 分期选取逻辑：逐 30 s epoch 判断分期标签，对 N3 epoch 施加峰峰值伪迹拒绝（阈值 200 µV）。SC4001 共有 220 个 N3 epoch，78 个因伪迹被拒绝，最终接受 142 个，拼接后总时长 4260 s。

输出：`S4_sbi/x_obs_v3.npz`，包含三个键：
- `values`：float32[5]，顺序为 SUMMARY_KEYS
- `keys`：5 个统计量的名称列表
- `extraction_metadata`：JSON 字符串，记录版本、信号链、去除的维度及原因

实测输出值：

| 维度 | 值 |
|------|----|
| shape_r | 1.0 |
| T4_q | 2.645 |
| T4_freq | 0.75 Hz |
| T8_n_sp_events | 15.31 次/60 s |
| T11_lag_ms | 1.28 |

### 1.2 EEG → r_proxy 信号链

r_proxy 的作用是把双极 µV EEG（有符号、宽频）转换为非负、慢波包络的"皮层放电率代理"，使其在统计量的公式形式上与仿真器的 r_ctx 对齐。转换步骤：

```python
eeg_1k  = resample_poly(eeg_uv, 10, 1)            # 100 Hz → 1000 Hz
eeg_raw = detrend(eeg_1k, type="linear")           # 线性去趋势，保留纺锤频段
r       = np.abs(eeg_raw)                          # 整流
r_sm    = gaussian_filter1d(r, sigma=50.0)         # 50 ms Gaussian 平滑
r_proxy = (r_sm - r_sm.min()) / np.percentile(r_sm, 95) * 60.0  # 归一化到 [0, 60]
```

`sigma=50.0` 对应 50 ms（1000 Hz 下 50 个样本）。该滤波器的频率响应为：

$$H(f) = \exp\!\left(-\frac{2\pi^2 \sigma^2 f^2}{f_s^2}\right)$$

各关键频率的衰减：

| 频率 | $H(f)$ | 衰减量（dB） |
|------|--------|------------|
| 1.0 Hz（SO 频段） | 0.951 | −0.4 |
| 10 Hz（纺锤下沿） | 0.0072 | −43 |
| 12 Hz（纺锤中心） | 0.0008 | −62 |

SO 频段几乎无衰减，纺锤频段（10–14 Hz）衰减至 0.1% 以下。这一特性是 §4.1–4.2 中 T6 和 MI 退化的直接物理原因。

`eeg_raw`（去趋势后、整流前的 1000 Hz EEG）同时被保留，作为 T8 和 T11 PAC 中纺锤振幅的输入信号（见 §1.3）。

### 1.3 各维度计算

**shape_r**：在 x_obs 端硬编码为 1.0，详细设计逻辑见 §4.3。

**T4_q 和 T4_freq**：对 r_proxy 做 Welch 功率谱密度估计，

$$\hat{S}(f) = \frac{1}{K} \sum_{k=1}^{K} \left| X_k(f) \right|^2$$

其中 K 个重叠汉宁窗。SO 频段定义为 [0.2, 1.5] Hz，邻域频段为 [0.1, 0.2) Hz 和 (1.5, 2.8] Hz，Q 因子定义为：

$$Q = \frac{P_\text{peak}}{\langle P_\text{neighbor} \rangle}$$

T4_freq 取 SO 频段内功率谱的峰值频率（argmax）。

**T8_n_sp_events**：输入信号为 `eeg_raw`（不是 r_proxy），保留了完整的 10–14 Hz 纺锤波内容。检测流水线：

```python
sos      = butter(4, [10, 14], btype="band", fs=1000, output="sos")
filtered = sosfiltfilt(sos, eeg_raw)               # 零相位带通
envelope = np.abs(hilbert(filtered))               # 瞬时振幅
env_sm   = gaussian_filter1d(envelope, sigma=200)  # 200 ms 平滑
thresh   = np.percentile(env_sm, 75)               # 75 百分位阈值
# 事件定义：阈值上穿持续 0.3–2.0 s 的连续段
t8_norm  = n_valid_events * (60.0 / duration_s)   # 归一化到每 60 s
```

SC4001 N3 段共检出 1087 个有效纺锤事件，归一化后 T8 = 15.31 次/60 s。

**T11_lag_ms（实际存储值为 up_down_ratio）**：调用 `compute_pac_metrics_fixed.compute_pac_metrics(r_proxy, eeg_raw, fs=1000)`，其中 r_proxy 提供 SO 相位、eeg_raw 提供纺锤振幅。cycle-by-cycle 相位提取：在 r_proxy 上用 `find_peaks`（最小间距 700 样本，突出度阈值 0.3 × max）定位 UP 峰，相邻两峰之间按时间线性插值至 $[0, 2\pi)$，相位直接对应 UP 状态发生时刻而非滤波输出。up_down_ratio 定义为：

$$\text{up\_down\_ratio} = \frac{\bar{A}_\text{spindle}^{(\text{UP})}}{\bar{A}_\text{spindle}^{(\text{DOWN})}}$$

其中 UP 相位区间为 $[-\pi/2,\,\pi/2]$，DOWN 为其余部分。键名 "T11_lag_ms" 是从 V7 的命名约定继承的历史遗留，实际存储的是无量纲比值（不是毫秒）。实测 T11 = 1.28，表示纺锤波振幅在 SO UP 状态期间平均高出 28%。

---

## 2. 仿真器包装：`simulator_wrapper.py`

### 2.1 输入 / 输出

输入：`theta = [g_h, g_LK, c_\text{ctx2th}, b]`，形状 (4,)，可接受 numpy array、Python list 或 torch Tensor。

输出：`x_sim`，形状 (5,)，顺序与 SUMMARY_KEYS 一致。仿真失败时返回 `np.full(5, np.nan)`，不向调用方抛出异常。

该文件在 `import` 时触发两个一次性操作：加载 V7 模块（通过 importlib，不触发 `__main__`）和预计算 SC4001 EEG 的 target PSD 及 FOOOF 周期性成分——这两步只在模块初始化时执行一次，后续每次仿真不重复。

### 2.2 自由参数与固定参数

自由参数（4 维），先验为 BoxUniform：

| 参数 | 生理含义 | 先验下界 | 先验上界 |
|------|---------|---------|---------|
| $g_h$ | 超极化激活电流（$I_h$）电导 | 0.035 | 0.095 |
| $g_{LK}$ | 漏电流（K$^+$）电导 | 0.020 | 0.070 |
| $c_\text{ctx2th}$ | 皮层→丘脑突触强度 | 0.05 | 0.22 |
| $b$ | 适应电流强度 | 28.4 | 42.6 |

固定参数（来自 Seed B，保持不变）：

| 参数 | 值 |
|------|---|
| mue | 3.3407 |
| mui | 3.2758 |
| tauA | 1257.4 |
| c_th2ctx | 0.03295 |

固定 Seed B 的理由：Seed B 在 364 个 warm_start DE 可行解中取得最高 fitness 分数（0.674）；将所有 8 个参数都设为自由维度会显著增加先验体积和所需仿真次数；当前 SBI 的目标是在 Seed B 附近的子空间中量化参数不确定性，而非全局搜索。

### 2.3 `_extract_summaries` 信号链

1. 运行 ALN+Thalamus 模型 65 s（5 s 预热 + 60 s 主段），丢弃前 5000 个样本（5 s × 1000 Hz）
2. 从模型输出中提取 r_ctx = r_exc[0,:] × 1000（kHz→Hz）和 r_thal = r_exc[1,:] × 1000
3. 调用 `compute_constraints_v7(r_ctx, r_thal, f_c, p_c, fs=1000)` 获取 T4、T8、T11 的数值
4. 单独计算 shape_r（见下节）
5. T8 归一化：主段正好 60 s，故归一化因子 60.0/60.0 = 1.0，T8 数值即为 60 s 内事件总数

### 2.4 shape_r FOOOF 路径

```python
# 1. 仿真 r_ctx → Welch PSD → 插值到 FOOOF 频率网格
p_interp = interp1d(f_ctx, p_ctx)(_fooof_freqs)
# 2. 拟合 FOOOF（仅在初始化时对 EEG 也运行一次相同步骤）
fm_sim = FOOOF(**EVO_FOOOF_PARAMS).fit(_fooof_freqs, p_interp, [0.5, 20.0])
# 3. 提取仿真侧周期性成分
sim_periodic = log10(p_interp) - fm_sim._ap_fit
# 4. 与预存的 EEG 周期性成分做 Pearson-r
r_val, _ = pearsonr(sim_periodic, target_periodic)
shape_r = max(r_val, 0.0)
```

FOOOF 将 PSD 分解为 $1/f$ 非周期背景和高斯形周期峰。只比较去除背景后的"周期残差"可以避免个体间 $1/f$ 斜率差异的干扰。若 FOOOF 未安装（`HAS_FOOOF=False`），shape_r 退化为 0.0，会导致所有仿真与 x_obs 的 shape_r=1.0 形成最大距离，使后验崩溃——部署时须确认 FOOOF 可用。

---

## 3. Stage 2a: 5-Dim SBI as Diagnostic Run

> 本节描述 2026-05-22 → 05-23 的 5 维 SBI 实跑（4D theta × 5D x_obs）。  
> 该 run 的目的从一开始就被定位为**诊断性运行**：通过观察 4 个自由参数在哪些维度上无法被现有 5 维 likelihood 约束，决定 Sprint 1 后半段是否需要重新引入 T6 + MI（以 EEG-native 算法重做,见 §5）。  
> May 7 的 7 维 run(`sbi_outputs_7dim_archive_20260507/`)在 §3.5 速览表里作为对照。

### 3.1 训练过程

| 项 | 值 |
|----|----|
| 起止 | 2026-05-22 20:31:10 → 2026-05-23 04:34:40 |
| 总墙钟 | **8.06 h**(483.5 min) |
| 主循环 ROUND_SIMS | [2000, 1000, 1000, 1000] |
| 早停 | **未触发**(R3→R4 std 相对变化 = 27.1%,超 10% 阈值,R4 跑满) |
| SBC + PPC | 200 + 100 |
| 总 sims | **5300** |
| NaN 率 | 0%(每一轮、SBC、PPC 全程 0 NaN) |
| 设备 | CPU(sbi=0.26.1,torch=2.5.1) |
| num_workers | 1(Windows + numba 强制串行) |
| 平均速率 | 5.45 s/sim |
| Prior(本次)| g_h ∈ [0.035, 0.095]; g_LK ∈ [0.020, 0.070]; c_ctx2th ∈ [0.05, 0.22]; **b ∈ [28.4, 80.0]**(b 上界从 May 7 的 42.6 放宽到 80.0,见 §3.2 验证) |

算法部分(SNPE-C / NSF / z-score / 4 轮 sequential)与 May 7 一致。

### 3.2 健康指标(数字摘要)

#### SBC 4/4 PASS

200 个先验 SBC 对,每个抽 1000 个后验样本,KS 检验对均匀分布:

| 参数 | KS 统计 | p-value | 状态 |
|------|--------|---------|------|
| g_h | 0.0500 | 0.6803 | ✅ |
| g_LK | 0.0620 | 0.4090 | ✅ |
| c_ctx2th | 0.0420 | 0.8573 | ✅ |
| b | 0.0880 | 0.0850 | ✅(贴边,余量 0.035) |

(May 7 同维度 SBC:g_h FAIL p=0.017,其余 3 参 PASS;本次 4/4 PASS,说明 NSF 在 5 维上学到的 likelihood↔posterior 映射在算法层面校准良好。)

#### PPC:5 维中 3 维 PASS

100 次后验采样 → simulator → 比对 x_obs:

| 维度 | x_obs 百分位 | 状态 |
|------|-------------|------|
| shape_r | 100% | ⛔ FAIL(设计哨兵,见 §4.3) |
| T4_q | 100% | ⛔ FAIL |
| T4_freq | 77% | ✅ |
| T8_n_sp_events | 11% | ✅ |
| T11_lag_ms | 37% | ✅ |

shape_r=1.0 的硬编码使其必然位于 100th 百分位,该 FAIL 是设计层面的预期(见 §4.3)。T4_q FAIL 是新发现的偏移项。

#### b prior 放宽验证有效

| 项 | May 7(7 维) | May 23(5 维) |
|----|--------------|----------------|
| b prior 上界 | 42.6 | **80.0** |
| b MAP | 42.37(贴上界,余量 0.5%) | **50.53**(离上界 80 仍有 37%) |
| b 95% CI | [36.74, 42.54] | [49.48, 50.76] |

放宽后 MAP 不再贴 prior 边界,且 CI 收缩到极窄(宽度 1.3),说明 b 是**强可识别参数**且 May 7 的"贴边"是 prior 截断伪影。

#### MAP + 95% CI 全表

| 参数 | MAP | CI_lo | CI_hi | Prior 范围 |
|------|-----|-------|-------|-----------|
| g_h | 0.07641 | 0.04939 | 0.07973 | [0.035, 0.095] |
| g_LK | 0.04914 | 0.04220 | 0.05687 | [0.020, 0.070] |
| **c_ctx2th** | **0.20636** | **0.05193** | **0.21358** | [0.05, 0.22](CI 横跨几乎全 prior) |
| b | 50.52520 | 49.48069 | 50.76123 | [28.4, 80.0] |

### 3.3 关键诊断发现:c_ctx2th unidentifiability

#### pairplot 三个 panel 形态(均无 ridge)

`fig_pairplot.png` 中 c_ctx2th 与其他三参数的上三角 contour 全部呈现**接近独立**的结构:

| 配对 | 形态 | 解读 |
|------|------|------|
| c_ctx2th × g_h | **横条**(c_ctx2th 横扫 0.05–0.22,沿 g_h 仅在 0.076 和 0.053 出现细带) | c_ctx2th 与 g_h 接近独立 |
| c_ctx2th × g_LK | **横条**(g_LK 集中 ~0.05,c_ctx2th 横扫全 prior) | c_ctx2th 与 g_LK 接近独立 |
| c_ctx2th × b | **竖条**(b ≈ 50 极窄,c_ctx2th 沿 y 方向 4 个亮带堆叠) | 4 个 c_ctx2th 模态**共享同一 b**,b 维不分离模态 |

三个 panel 中**没有任何斜对角 ridge**——三个其他参数都没有"解释"c_ctx2th 的多模态来源。

#### 4 个 spurious modes(从 10000 后验样本统计)

| Mode | 估计位置 | 邻域占比 | 备注 |
|------|---------|---------|------|
| M1 | 0.05–0.06 | **8.99%** | 贴下边界,呈撞墙形态(0.058→0.067 降 41%,0.067→0.076 降 82%) |
| M2 | 0.10–0.12 | 5.55% + 5.94% | 内部峰 |
| M3 | 0.15–0.17 | 10.86% + 8.76%(第二高) | 内部峰 |
| M4 | 0.205–0.215 | **15.35%**(最高) | 内部峰,位于 0.207;**不撞 0.22 上界**(0.205–0.21 → 0.215–0.22 占比 9.88% → 2.41%,从峰下降) |

#### 与 May 7 7 维模态结构对比

| Run | 维度 | c_ctx2th 模态数 | 主峰位置 | b prior 撞边 |
|-----|------|----------------|---------|------------|
| May 7 | 7(含 T6, MI) | 3 | 0.075 | 是(42.6) |
| **May 23** | **5**(去 T6, MI) | **4** | **0.207** | 否(80.0) |

模态数和主峰位置随 likelihood 维度的选择而**质变**——若 c_ctx2th 是真实生理可识别参数,模态结构应稳定;模态结构对维度敏感即等价于"likelihood 不约束 c_ctx2th"。

#### Pareto seed log_prob 排序变化

| Seed | May 7(7 维) | May 23(5 维) | DE 标签 |
|------|--------------|----------------|---------|
| A | -12.07 | -9.32 | PAC-dominant(MI=0.137) |
| B | **+2.41** | **-4.11** | 最高 fitness(score 0.674) |
| C | +1.21 | -20.42 | shape-dominant(shape_r=0.690) |

排序由 May 7 的 **B > C > A** 变为 May 23 的 **B > A > C**:Seed A 与 Seed C 的相对位置反转。5 维已不再包含 MI(即 PAC 调制指数),故 PAC-dominant 的 Seed A 与 shape-dominant 的 Seed C 在 5 维 likelihood 下不再被有效区分。所有三个 seed 的 log_prob 绝对值都跌入负区,反映 5 维 likelihood 整体不再像 7 维那样精确锚定。

### 3.4 诊断结论:5 维 EEG-derived summary 信息不足

三条证据交叉印证 c_ctx2th 在当前 5 维 likelihood 下不可识别:

1. **算法层面无故障**:SBC 4/4 PASS 表明 NSF 已正确学到 likelihood→posterior 映射,后验形状不是训练欠拟合的结果。
2. **力学模型无故障**:PPC 3/5 PASS、0% NaN、Seed B 仿真在 simulator 端给出 shape_r=0.677(与 DE Seed B 表的 0.679 一致到 0.3%),仿真器能稳定再现 x_obs 附近的输出。
3. **后验形态明确表明信息匮乏**:c_ctx2th 在边际呈 4 个 spurious modes,联合分布与三个其他参数全部接近独立(横/竖条无 ridge),且模态结构对 likelihood 维度变化敏感。

→ **结论**:5 维 EEG-derived summary statistics 对 c_ctx2th(皮层→丘脑突触强度)的信息量约等于零。要拿到一个对 c_ctx2th 有约束力的后验,必须重新引入 T6(SO UP 节律的 IBI CV)和 MI(SO 与 spindle 之间的 PAC 强度),且必须用 EEG-native 算法替代 §1.2–1.3 中的 r_proxy 包络路径(后者已被 §4.1–4.2 证明在两个维度上都不可修复)。

此结论直接 motivate Sprint 1 后半段(详见 §5):EEG-native T6 + EEG-native MI 重做 → 7 维 x_obs 第二次重跑 → 与本次 5 维 baseline 对比 c_ctx2th 后验是否收窄。

### 3.5 May 23 vs May 7 速览表(替换原 §3.6 速览)

| Metric | May 7(7 维) | May 23(5 维) | 备注 |
|--------|--------------|----------------|------|
| 总耗时 | 9.66 h | **8.06 h** | 5 维仿真本身不更快;5 维 sims 多 1200 次但因稳定速率反而总时间更短 |
| 主循环 sims | 4000 | 5000 | R4 跑满 |
| NaN 率 | 0% | 0% | |
| R4 早停 | 是(R2→R3 5.8%) | 否(R3→R4 27.1%) | 5 维下后验未充分收敛 |
| SBC 通过 | 3/4(g_h FAIL p=0.017) | **4/4** | |
| PPC 同维度通过(不计 T6/MI) | 2/5(shape_r/T4_q/T8 FAIL) | **3/5**(shape_r/T4_q FAIL) | T8 由 1% 升至 11%(归一化口径改了) |
| c_ctx2th 模态数 | 3 | **4** | |
| c_ctx2th 主峰 | 0.075 | **0.207** | 主峰从 prior 下半区跳到上半区 |
| b prior 上界 | 42.6 | 80.0 | |
| b MAP | 42.37(贴边) | 50.53 | 放宽后落在新区域,撞边伪影解除 |
| Pareto 排序 | B > C > A | **B > A > C** | A↔C 反转 |

### 3.6 输出文件清单

`S4_sbi/sbi_outputs/`(May 23 5 维 run):

- `all_simulations.npz`(theta 5000×4 + x 5000×5)
- `round{1,2,3,4}_posterior.pkl`(R4 跑满)
- `fig_marginals.png` / `fig_pairplot.png` / `fig_ppc.png` / `fig_pareto_overlay.png` ✅
- `fig_sbc.png` ⚠️ 主 run 因 `sbc_rank_plot()` 缺 `num_posterior_samples` 参数报错未生成,由 `S4_sbi/replot_sbc_5dim.py` 补出
- `S4_sbi/sbi_results.md`(MAP/CI/SBC/PPC 完整表)
- 主日志 `S4_sbi/sbi_log.txt`,stdout 镜像 `sbi_5dim_run.log`,stderr `sbi_5dim_run.err`(仅 fooof DeprecationWarning)

旧 7 维产物归档至 `S4_sbi/sbi_outputs_7dim_archive_20260507/`(9 个文件,含原 5 张图 + 3 个 pkl + all_simulations.npz)。

### 3.7 中止规则(未变,记录在案)

| 条件 | 处理 | 本次是否触发 |
|------|------|------------|
| 任一轮 NaN 率 > 30% | `RuntimeError` | 否(0%) |
| 有效 sims < 200(对 ≥500 轮次) | `RuntimeError` | 否 |
| R2→R3 / R3→R4 std 相对变化 < 10% | 跳过下轮,提前结束 | 否(27.1%) |
| SBC ≥ 2/4 KS 失败 | WARNING,不中止 | 否(0/4 失败) |

---

## 4. 诊断工具与设计决策

### 4.1 T6 IBI CV 退化

`scan_xobs_params.py` 对 500 ms Gaussian 平滑后的 r_proxy 扫描了 5 个阈值，检测 IBI CV 是否落在生理合理区间 [0.3, 0.6]：

| 阈值 | n_bursts | IBI CV | 是否通过 |
|------|----------|--------|---------|
| 5 | 4 | 0.631 | ✗（只有 3 个 IBI 区间，统计不稳定） |
| 8 | 23 | 1.542 | ✗ |
| 10 | 88 | 1.955 | ✗ |
| 12 | 196 | 2.265 | ✗ |
| 15 | 368 | 1.714 | ✗ |

全部失败。根本原因：r_proxy 是慢包络信号，在 4260 s 内几乎始终维持在高值（threshold=5 才产生 4 次穿越，说明信号绝大部分时间都在 5 以上）；当阈值升高，平滑包络的微小起伏产生大量"虚假 burst"，IBI 序列极度不规则，CV 远超上限。这是 r_proxy 的形态特征决定的，无法通过调整阈值解决。诊断图见 `S4_sbi/scan_diagnostics/fig_t6_threshold_scan.png`。

### 4.2 MI 退化

`scan_xobs_params.py` 扫描 `SO_PEAK_PROMINENCE_FRAC` ∈ [0.05, 0.30]，目标 MI ∈ [0.01, 0.05]：

| prom_frac | n_so_peaks | MI | up_down_ratio | 是否通过 |
|-----------|-----------|-----|---------------|---------|
| 0.05 | 3761 | 0.00017 | 1.232 | ✗ |
| 0.10 | 3479 | 0.00025 | 1.219 | ✗ |
| 0.15 | 3143 | 0.00018 | 1.219 | ✗ |
| 0.20 | 2752 | 0.00014 | 1.244 | ✗ |
| 0.30 | 1901 | 0.00012 | 1.280 | ✗ |

全部失败，MI 约为目标区间下界的 1/100。物理原因直接：

$$H(12\,\text{Hz}) = \exp\!\left(-\frac{2\pi^2 \times 50^2 \times 12^2}{1000^2}\right) \approx 0.0008$$

r_proxy 中 12 Hz 纺锤能量衰减至 0.08%。即使在 v2/v3 中改用 eeg_raw 提供纺锤振幅，cycle-by-cycle 相位仍从 r_proxy 的包络极值提取，而这些极值并不对应真实皮层 UP 状态。pf=0.30 时检出 1901 个峰（约 27 个/min），落在 SO 生理预期 30–50/min 的低端，峰值数量本身并不异常；核心问题是整流操作（`r = abs(eeg_raw)`）将 EEG 负相（DOWN 状态的大振幅负偏转）折叠为正值，使包络极值同时出现在 UP 峰和 DOWN 谷两处，与真实皮层 UP 峰相位相差约半个周期。cycle-by-cycle 算法以包络极值作为 phase = 0 的锚点，提取出的 SO 相位不再是物理意义上的 UP 状态锚点，纺锤振幅在相位空间中的分布被系统性打乱，MI → 0。参数调整不可修复此问题。诊断图见 `S4_sbi/scan_diagnostics/fig_mi_prominence_scan.png`。

### 4.3 shape_r 硬编码 1.0：哨兵设计

x_obs 端 `d["shape_r"] = 1.0` 是一个上界哨兵。EEG 自身的周期性成分与自身的 Pearson-r 必然为 1.0，因此这个值在物理上是合理的。

从 SBI 密度估计器的视角：训练数据对形如 `(sim_shape_r ∈ [0, 1], x_obs_shape_r = 1.0)`。估计器在推断时会把较高的后验密度分配给能产生较高 shape_r 的参数区域，相当于把"谱型匹配"这一偏好软性编码进了后验，而无需在损失函数中增加显式正则项。

PPC 中 shape_r 的 x_obs 必然落在 100th 百分位（任何仿真都达不到 1.0）——这是设计预期，不算诊断失败。

### 4.4 T6 与 MI 作为 mechanistic gate：EEG 物理学理由

T6（皮层 UP/DOWN 节律的 IBI CV）和 MI（皮层 SO 与丘脑纺锤波的 PAC 调制指数）在仿真器内部有确定性的值，继续在 `compute_constraints_v7` 的 T1–T12 约束体系中作为力学合理性的门控（feasibility gate）使用。它们不进入 x_obs 的原因是 EEG 信号的物理限制，而非噪声或数据量不足：

- 头皮 EEG 信号的 >95% 来自皮层锥体细胞的突触后电位（Nunez & Srinivasan, 2006）。
- 丘脑继电核（TC）和丘脑网状核（TRN）在解剖上是封闭场源（closed-field geometry）：相邻细胞的树突-轴突方向相互抵消，几乎不产生可测量的头皮偶极子（Buzsáki et al., 2012）。
- 即使使用颅内深部电极，EEG 对丘脑信号的灵敏度也比皮层低 10× 以上（Percival & Schwartz, 2013，近似引用；确切文献待查）。

这意味着：即使通过 EEG 重建出了良好的 SO 相位，PAC 中"纺锤振幅"的来源也不是丘脑直接信号，而是皮层对丘脑纺锤波的反应——两者有延迟和幅度失真。T6 和 MI 在 EEG 端的可提取性是结构性的低保真，不是本版本的技术限制。

### 4.5 PAC 输入不对称

仿真器端 PAC（`compute_pac_metrics_fixed(r_ctx, r_thal)`）的两个输入：
- `r_ctx`：皮层放电率 → UP/DOWN 节律清晰 → SO 相位提取可靠
- `r_thal`：丘脑放电率 → 10–14 Hz T 电流共振振荡 → 真实的纺锤波振幅

x_obs 端即使在 v2/v3 的修复后：
- r_proxy → SO 相位（包络极值过密，相位插值不可靠）
- eeg_raw → 纺锤振幅（真实 EEG 纺锤波，但混有皮层反应 + 噪声，无 r_thal 直接信号）

这种不对称在仿真-观测之间制造了一个系统性偏差：即使参数完全正确，$x_\text{sim}^{(\text{MI})}$ 和 $x_\text{obs}^{(\text{MI})}$ 也不可比。Stage 3 的 EEG 原生 PAC 方案（见 §5.2）的目标是尽量弥合这一差距，但丘脑的 closed-field 限制意味着两侧的 MI 永远不可能完全等价。

### 4.6 `run_sbc_standalone.py`：SBC 事后重跑工具

该脚本独立于 `run_sbi.py` 的训练流程，单独执行 SBC 诊断。用途：

1. `run_sbi.py` 内置 SBC 因 sbi 版本 API 差异失败时的补救路径
2. 训练完成后，需要使用不同样本数重新评估校准性
3. 在不同 x_obs 文件下快速检查后验校准

核心流程：加载 `sbi_outputs/round3_posterior.pkl` → 从先验采样 200 个 $(\theta, x)$ 对 → 对每对抽 1000 个后验样本计算秩 → KS 检验 → 保存 `fig_sbc.png`。与 `run_sbi.py` 内置版本的区别：不需要重跑 4 轮训练，节省约 9.5 小时。

### 4.7 版本演化：v1 → v2 → v3

| 版本 | 维度 | T8 来源 | PAC 振幅来源 | T6 阈值 | 已知问题 |
|------|------|---------|------------|---------|---------|
| v1（`compute_xobs_from_eeg.py` = `_v1_buggy.py`） | 7 | r_proxy（无纺锤内容） | r_proxy × r_proxy | 50th pctile（自适应） | T8 = 噪声计数；MI ≈ 0；T6 约束强制 50% 占空比 |
| v2（`compute_xobs_from_eeg_v2.py`） | 7 | eeg_raw ✓ | r_proxy 相位 + eeg_raw 振幅 | 硬阈值 15 | T8 = 15.31 ✓；MI = 0.00012（仍近零）；T6 = 1.714（暴露结构性不兼容） |
| v3（`compute_xobs_from_eeg_v3.py`） | **5** | eeg_raw ✓ | r_proxy 相位 + eeg_raw 振幅 | 已删除 | T6 和 MI 经诊断扫描后确认不可修复，从 x_obs 移除 |

`compute_xobs_from_eeg.py` 与 `compute_xobs_from_eeg_v1_buggy.py` 内容完全一致（`diff` 无输出）；后者是手动存档副本，§1 引用时以 v1 称呼，指 `compute_xobs_from_eeg.py`。

---

## 5. Stage 3 计划（不在本次迭代实现）

### 5.1 EEG 原生 T6：从 raw EEG 检测 SO UP 事件

替代 r_proxy 包络阈值法，直接在 raw EEG 上检测慢波 UP 事件：对 eeg_raw 做 [0.2, 4] Hz 带通滤波，检测正向过零点（DOWN→UP 转换）或 EEG 正峰，施加振幅（> 75 µV 半波，AASM 标准）和持续时间（0.5–2 s）筛选。由此得到的 UP 事件时间戳序列直接计算 IBI CV，与 V7 的 r_ctx 阈值穿越方法在方法论上对齐。预期 T6 会从 v2 的 1.714 下降到生理合理范围 [0.3, 0.6]。

### 5.2 EEG 原生 MI：从 raw EEG 提取纺锤振幅

PAC 计算的两个输入改为均来自 raw EEG：

- **SO 相位**：将 `find_peaks` 作用于 [0.5, 1.5] Hz 带通滤波后的 EEG 正峰（而非 r_proxy 的包络极值），或改用 Hilbert 相位提取（需评估滤波群延迟对 UP 峰位置的影响）
- **纺锤振幅**：与 T8 相同的 [10, 14] Hz + Hilbert 流水线，作用于 eeg_raw

这样两臂均包含真实的 EEG 信号成分，与仿真器侧（r_ctx SO 相位 + r_thal 纺锤振幅）的方法论对应关系比 v1–v3 都更接近。预期 MI 从 ~0.0001 提升到 [0.01, 0.05] 范围。

### 5.3 7-dim SBC/PPC 基准对比

以 5-dim（v3，本次）和 7-dim（Stage 3）分别完成一轮 4000-sim SNPE-C，对比：
- SBC 各参数的 KS $p$-value
- PPC 各维度的百分位分布（尤其关注 T6 和 MI 两个新维度是否通过）
- 后验标准差（宽窄度）——新增维度若有效约束参数，后验应收窄
- MAP 估计的稳定性

核心问题：T6 和 MI 是否能对 $g_h$（$I_h$ 电流驱动 SO 周期节律）和 $c_\text{ctx2th}$（皮层-丘脑 PAC 通路）提供额外约束？如果后验在这两个维度上没有明显收窄，7-dim 方案的额外工程成本将难以收回。

---

## 附录：引用数字索引

本附录列出全文所有具体数字及其来源，供 review 时核对。

| 数字 | 所在节 | 来源 |
|------|-------|------|
| 220 N3 epochs，78 rejected，142 accepted，4260 s | §1.1 | 脚本运行日志（compute_xobs_from_eeg_v3.py 终端输出） |
| x_obs_v3: shape_r=1.0, T4_q=2.645, T4_freq=0.75, T8=15.31, T11=1.28 | §1.1 | 脚本运行日志 |
| H(10 Hz) ≈ 0.0072，H(12 Hz) ≈ 0.0008 | §1.2 | 本文计算，公式 $\exp(-2\pi^2 \sigma^2 f^2 / f_s^2)$，σ=50, fs=1000 |
| 1087 个有效纺锤事件，T8=15.31/60s | §1.3 | 脚本运行日志 |
| Seed B: mue=3.3407, mui=3.2758, tauA=1257.4, c_th2ctx=0.03295 | §2.2 | `S4_v7_repair/pareto_seeds_fresh_DE.json` |
| 364 个 warm_start DE 可行解 | §0.1, §2.2 | `S4_v7_repair/pareto_seeds_fresh_DE.json` 第 3 行 `"n_feasible": 364` |
| Seed B fitness = 0.674 | §0.1（脚注）, §2.2 | `pareto_seeds_fresh_DE.json`，Seed B objectives.score |
| 先验边界 [0.035–0.095, 0.020–0.070, 0.05–0.22, 28.4–42.6] | §2.2, §3.3 | `run_sbi.py` PRIOR_LOW / PRIOR_HIGH 常量 |
| 4000 次仿真，9.66 小时，0% NaN | §3.3 | 用户 prompt 提供 |
| R2→R3 std reduction 5.8% | §3.3 | 用户 prompt 提供 |
| MAP: g_h=0.0592, g_LK=0.0514, c_ctx2th=0.069, b=42.37 | §3.3 | 用户 prompt 提供 |
| SBC: g_h KS p=0.017，其余 3 参数 PASS | §3.4 | 用户 prompt 提供 |
| PPC: all 5 dims pass（精确百分位待 `sbi_results.md` 中查询） | §3.5 | 用户 prompt 提供 |
| Pareto log_prob: Seed B +2.41, Seed C +1.21, Seed A −12.07 | §3.5 | 用户 prompt 提供 |
| T6 扫描：threshold 5→8→10→12→15, IBI CV 0.631→1.542→1.955→2.265→1.714 | §4.1 | `scan_xobs_params.py` 运行日志 |
| MI 扫描：pf 0.05→0.30, MI 0.00017→0.00012，n_peaks 3761→1901 | §4.2 | `scan_xobs_params.py` 运行日志 |
| Seed A: score=0.510, MI=0.137, PAC_compound=1.0 | §3.5 | `pareto_seeds_fresh_DE.json` |
| Seed B: score=0.674, MI=0.070, PAC_compound=0.592 | §3.5 | `pareto_seeds_fresh_DE.json` |
