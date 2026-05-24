# Sprint 1 Phase 2 — EEG-native T6 + MI 技术规划 v2

**前置依据**:`sbi_report_0511.md` §3.4 — 5 维 EEG-derived summary 信息不足,需重新引入 T6 + MI,EEG 端用 AASM-style 算法替代 r_proxy 包络路径。

**v2 变化点(相对 v1)**:
- **Rev 1+3 合并**:r_ctx 端不做 bandpass+find_peaks,**保留 V7 现行 T6 算法**(已使用绝对 Hz 阈值,无 σ 自适应 loophole)
- **Rev 2 verify 结论**:V7 spindle band 当前就是 10–14 Hz,与 EEG-native plan 一致,无须改 V7
- 工作量从 3 工作日上修至 **4–5 工作日**(加 V7 NaN 哨兵处理 + 2 轮 Seed B 调参 + 7 维 SBI 可能需要 prior 微调)

---

## 1. EEG-native T6:从 raw EEG 检 SO UP 事件(AASM)

### 算法(仅 EEG 端使用)

输入:`eeg_raw`(1000 Hz,detrended,µV),来自 `compute_xobs_from_eeg_v3.py` 的 step 2 输出。

```python
from scipy.signal import butter, sosfiltfilt, find_peaks

SO_BAND        = (0.2, 4.0)   # Hz, 放宽到 4 Hz 抑制群延迟
HALF_WAVE_UV   = 75.0         # AASM 2017 §IV.D.2: peak-to-trough ≥ 75 µV
DUR_LO_S       = 0.5
DUR_HI_S       = 2.0

sos        = butter(4, SO_BAND, btype="band", fs=1000, output="sos")
eeg_so     = sosfiltfilt(sos, eeg_raw)
neg_peaks, _ = find_peaks(-eeg_so, distance=int(0.4*1000))   # DOWN
pos_peaks, _ = find_peaks( eeg_so, distance=int(0.4*1000))   # UP
```

对相邻 DOWN(t_dn)→ UP(t_up)对,若 `t_up - t_dn ∈ [0.5, 2.0] s` **且** `eeg_so[t_up] - eeg_so[t_dn] ≥ 75 µV`,该 UP 事件入选。

得到 `up_times` 序列,IBI = `diff(up_times)`,**T6 = std(IBI) / mean(IBI)**。

### r_ctx 端:保留 V7 算法,不动

V7 现行 `compute_constraints_v7` 中(`models/s4_personalize_fig7_v7.py:495-514`):

```python
# T3 已用 run-length encoding 从 r_ctx > 阈值(绝对 Hz)检出 UP 事件
# T6 直接复用 T3 的 `starts` 序列
ibi_cv = 999.0
if len(starts) >= 3:
    intervals = np.diff(starts) / fs
    ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
details["T6_ibi_cv"] = round(ibi_cv, 3)   # ← 已暴露,可直接接到 SBI x
```

**绝对阈值**(V7 经验值在 r_ctx 上 ≥ 15 Hz),不随仿真信号自身振幅变化,避免 V7 当年要防的 fitness loophole。`T6_ibi_cv` 字段在 `compute_constraints_v7` 已经返回,simulator_wrapper 把它读出来即可,无需 V7 代码改动。

### 工程论证:两端不同算法,同一物理量

EEG 端 AASM bandpass+find_peaks 和 r_ctx 端 run-length encoding 在数值实现上不同,但**两者都在测同一物理量:皮层 SO UP 事件的发生时刻序列 → IBI 序列 → CV**。

这种"两端不同算法,同一物理量"的模式在当前 pipeline 里已有先例:

| 维度 | EEG 端 | simulator 端 | 共同物理量 |
|------|--------|--------------|-----------|
| T8 | raw EEG bandpass 10–14 Hz + Hilbert envelope → 阈值过线计数 | r_thal envelope → 阈值过线 + σ-band power 验证 | spindle event count |
| **T6**(新) | AASM bandpass + DOWN-UP pairing(75 µV / 0.5–2 s) | r_ctx > 15 Hz run-length encoding | SO UP IBI CV |

SBI 密度估计器对 x 做 `z_score_x='independent'` 归一化,各维度独立缩放后再训练,**两端数值绝对值差异不会污染推断**——只要 sim/obs 同维度内部的相对结构一致即可。

### V7 NaN 哨兵处理

V7 在 `n_bursts < 3` 时返回 `T6_ibi_cv = 999.0`,这会污染 SBI 训练。simulator_wrapper 端需要拦截:

```python
t6_raw = float(con.get("T6_ibi_cv", 999.0))
if t6_raw > 5.0:          # 实际生理 CV 上限 ~1.5;> 5 视为"未检出 UP"
    return np.full(len(SUMMARY_KEYS), np.nan)
```

让该次 sim 整体被 SBI 当 NaN 过滤掉,而不是让 999 误导 NSF 学习。

---

## 2. EEG-native MI:Tort 2010 全 EEG 版本

### 频段已对齐 V7

- V7 `SPINDLE_LO=10.0`, `SPINDLE_HI=14.0`(`s4_personalize_fig7_v7.py:292-293`)
- V7 `compute_pac_metrics_fixed(...)` 默认 `SPINDLE_LO=10.0, SPINDLE_HI=14.0`(`S4_v7_repair/compute_pac_metrics_fixed.py:81`)
- EEG-native plan:`AMP_BAND = (10, 14) Hz`

**结论:三处均使用 10–14 Hz,无须改动。** SC4001 用 Fpz-Cz/Pz-Oz,Pz-Oz 对 fast spindle (12–15 Hz) 敏感,10–14 Hz 是合理选择且不依赖于 raw EEG channel 选取。

### 算法(EEG 端)

```python
PHASE_BAND   = (0.5, 1.5)    # Hz
AMP_BAND     = (10.0, 14.0)  # Hz, 三处对齐
N_PHASE_BINS = 18

sos_p = butter(4, PHASE_BAND, btype="band", fs=1000, output="sos")
sos_a = butter(4, AMP_BAND,   btype="band", fs=1000, output="sos")
phase = np.angle(hilbert(sosfiltfilt(sos_p, eeg_raw)))
amp   = np.abs (hilbert(sosfiltfilt(sos_a, eeg_raw)))

bins             = np.linspace(-np.pi, np.pi, N_PHASE_BINS + 1)
mean_amp_per_bin = np.array([
    amp[(phase >= bins[k]) & (phase < bins[k+1])].mean()
    for k in range(N_PHASE_BINS)
])
p   = mean_amp_per_bin / mean_amp_per_bin.sum()
H   = -np.sum(p * np.log(p + 1e-12))
MI  = (np.log(N_PHASE_BINS) - H) / np.log(N_PHASE_BINS)
```

### simulator 端

`r_ctx` 提供 SO 相位(同 [0.5, 1.5] Hz bandpass + Hilbert),`r_thal` 提供 spindle 振幅([10, 14] Hz bandpass + Hilbert)。

r_ctx 是非负 firing rate,做 bandpass **不会**引入 DOWN-UP pairing 偏差(MI 只取 Hilbert 相位,不依赖正负极性的 peak/trough 对齐)。**这就是为什么 MI 可以在两端用相同算法,而 T6 必须分家**:

| 操作 | 对正负极性敏感? |
|------|----------------|
| Hilbert phase | 否(只取角度) |
| Hilbert amplitude | 否(取模) |
| find_peaks DOWN-UP pairing | **是**(r_ctx 中心化后偏置进 bandpass,DOWN 谷不真实) |

---

## 3. 代码改动清单

| 文件 | 改动 |
|------|------|
| `S4_v7_repair/compute_pac_metrics_eeg_native.py`(新) | 实现 `compute_t6_eeg_aasm(eeg_raw, fs)` 与 `compute_mi_eeg_native(eeg_raw, fs)`;**不含 r_ctx 路径**,r_ctx 端走 V7 |
| `S4_sbi/compute_xobs_from_eeg_v4.py`(新) | 复用 v3 的 `load_n3_eeg + build_rate_proxy`,把 T6/MI 调用换到新模块;SUMMARY_KEYS 重回 7 维 `[shape_r, T4_q, T4_freq, T6_ibi_cv, T8_n_sp_events, T11_lag_ms, MI]` |
| `models/s4_personalize_fig7_v7.py` | **不改 T6 算法**(保留 run-length);只在 `compute_constraints_v7` 末尾补一条:把 PAC MI 切换到新算法(原 `compute_pac_metrics_fixed` 留作 fallback,用 flag 控制) |
| `S4_sbi/simulator_wrapper.py` | SUMMARY_KEYS 由 5 改回 7,补 T6_ibi_cv 与 MI;**加 V7 NaN 哨兵**(T6_ibi_cv > 5 → 整次 sim 返回 NaN);默认 x_obs 路径同步改为 `x_obs_v4.npz` |
| `S4_sbi/run_sbi.py` | 默认 `--x-obs` 改 `x_obs_v4.npz`;ROUND_SIMS 保持 `[2000,1000,1000,1000]`;b 上界保持 80;新归档目录命名 `sbi_outputs_5dim_archive_20260523/` |

---

## 4. 验证步骤(三层)

### 4.1 单信号合成测试(`valid_scripts/validate_t6_mi_eeg_native.py`)

构造 5 个合成 EEG 信号:
1. 全耦合 SO+spindle(MI 预期 > 0.03)
2. 纯 SO,无 spindle(MI 预期 < 0.001,T6 应可算出)
3. 纯 spindle,无 SO(T6 应给 NaN/n_bursts<3,MI 退化)
4. 纯白噪(T6 应给 999,被哨兵抓住)
5. SO 与 spindle 反相(MI 仍非零但 amp 集中在 DOWN 相位)

### 4.2 Seed B 单次仿真验证

跑 `simulator_wrapper.simulator(theta_B)`,7 维输出预期:

| 维度 | 预期 | 容差 | 阻塞? |
|------|------|------|------|
| shape_r | 0.67–0.68 | ±0.01 | 是 |
| T4_q | 2.3–2.4 | ±0.1 | 否 |
| T4_freq | 1.0–1.3 Hz | ±0.2 | 否 |
| **T6_ibi_cv** | **0.30–0.60** | hard window | **是**(超范围回去 debug V7 阈值) |
| T8_n_sp_events | 15–25 | ±5 | 否 |
| T11_lag_ms | 3.0–4.0 | ±0.5 | 否 |
| **MI** | **0.02–0.05** | 与 x_obs_v4 实测 MI **同量级** | **是** |

注:MI 的"同量级"判据替代了原 "Seed B pareto 表 0.07" 参考。Pareto 表 0.07 是 V7 原算法(r_ctx 提供 phase,r_thal 提供 amp)的 **cross-channel** PAC 数值;EEG-native MI 是 **single-channel** PAC(同一 raw EEG 同时提供 phase 和 amp),信噪比天然低于 cross-channel,实测大概率落在 0.02–0.05。**真正的两端可比性 = simulator 输出 MI 与 x_obs_v4 EEG 实测 MI 在同一数量级**,不要求与 V7 历史数值一致。

**两轮调参预算**:
- 第 1 轮若 T6 或 MI 偏离,先调 V7 r_ctx threshold(可能从 15 Hz 提到 18 Hz)或 EEG 端 75 µV 阈值(可能调到 70 µV 适配 SC4001 信噪比)
- 第 2 轮微调 + 最终签字

### 4.3 SC4001 EEG 端验证(`x_obs_v4.npz`)

| 维度 | 预期范围 |
|------|---------|
| T6_ibi_cv | 0.40–0.55(健康 N3) |
| MI | 0.02–0.04(SC4001 中等 PAC) |

**若任一项落在生理范围外,先 debug 算法,不进 SBI。**

---

## 5. 工作量预估(v2 上修)

| 任务 | v1 估计 | v2 估计 | 备注 |
|------|---------|---------|------|
| 新算法模块 + 单元测试 | 4 h | 4 h | r_ctx 路径取消,工时持平 |
| v4 x_obs 抽取 + SC4001 调通 | 2 h | 3 h | 加 SC4001 实测范围核对 |
| simulator NaN 哨兵 + Seed B 验证(第 1 轮)| 4 h | 5 h | 加 V7 T6 行为剖析 |
| Seed B 第 2 轮调参(必备预算)| — | 4 h | v1 漏算 |
| MI 切换到新算法 + V7 fallback flag | 2 h | 3 h | 加 V7 回归测试,确认不影响现有 DE pipeline |
| 合成信号验证(5 个 case) | 2 h | 3 h | 多一个反相 case |
| run_sbi.py 配置调整 + dry-run | 1 h | 2 h | dry-run 验证 7 维输入维度 + sbc_rank_plot 已修补 |
| **代码 + debug 合计** | **15 h** | **24 h(~3 个工作日)** | |
| 7 维 SBI 主 run | 9–10 h | **10–12 h** | 7 维 NSF 训练略重 + 可能 R4 跑满(May 23 已经跑满,5300 sims;7 维可能再加 PPC/SBC 时间) |
| 后处理 + 5→7 维对比 + 报告 | 3 h | 6 h | 加 c_ctx2th 后验收窄量化、Pareto 排序回归比较、报告新增 §3b |
| 7 维 SBI 若需 prior 微调重跑(20% 概率)| — | 10 h | 备用预算,实际触发取决于第一次 SBI 结果 |
| **总计** | **~3 个工作日 + 1 晚算力** | **~4–5 工作日 + 1–2 晚算力** | |

---

## 6. v2 决策汇总

| 修订 | v2 决策 | 引用 |
|------|---------|------|
| Rev 1(r_ctx 阈值绝对化) | 自动满足(V7 现有算法已用绝对 Hz) | §1 「r_ctx 端」 |
| Rev 2(MI spindle 频段一致性) | 10–14 Hz 三处已对齐,无须改动 | §2 第一段 |
| Rev 3(r_ctx T6 保留 V7 算法) | 接受,工程论证见 §1 末尾「两端不同算法同一物理量」表 | §1 |

完成后产出:7 维 `sbi_outputs/` + `sbi_report_0511.md` 新增 §3b(7 维结果) + c_ctx2th 边际后验在 5→7 维之间的收窄量化对比 + Pareto 排序回归至 B>C>A 的验证。
