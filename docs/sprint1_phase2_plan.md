# Sprint 1 Phase 2 — EEG-native T6 + MI 技术规划

**前置依据**:`sbi_report_0511.md` §3.4(5 维 EEG-derived summary 对 c_ctx2th 信息不足,需重新引入 T6 + MI,但必须用 EEG-native 算法替代 r_proxy 包络路径)。

本文档只写 **HOW**(具体算法、代码改动、验证步骤、工作量),不写 WHY。

---

## 1. EEG-native T6:从 raw EEG 检 SO UP 事件

### 算法

输入:`eeg_raw`(1000 Hz,detrended,µV),来自 `compute_xobs_from_eeg_v3.py` 的 step 2 输出(已存在,无需重新生成)。

```python
from scipy.signal import butter, sosfiltfilt, find_peaks

SO_BAND       = (0.2, 4.0)   # Hz, AASM 0.5–2.0 Hz half-wave SO,带宽放宽至 4 Hz 防群延迟
HALF_WAVE_UV  = 75.0         # peak-to-trough amplitude (AASM 2017 §IV.D.2)
DUR_LO_S      = 0.5
DUR_HI_S      = 2.0

sos       = butter(4, SO_BAND, btype="band", fs=1000, output="sos")
eeg_so    = sosfiltfilt(sos, eeg_raw)            # 零相位带通,保留 UP 时刻
neg_peaks, _ = find_peaks(-eeg_so, distance=int(0.4*1000))   # DOWN 极值
pos_peaks, _ = find_peaks( eeg_so, distance=int(0.4*1000))   # UP 极值
```

对每个相邻 DOWN→UP 对(t_dn → t_up,t_up - t_dn ∈ [DUR_LO_S, DUR_HI_S]),计算 peak-to-trough 幅度 `A = eeg_so[t_up] - eeg_so[t_dn]`;通过 `A ≥ 75` µV 的 UP 事件入选。

得到 UP 时间戳序列 `up_times`(秒),IBI 序列 `ibis = diff(up_times)`,**T6 = std(ibis) / mean(ibis)**。

### simulator 端对应改动

`r_ctx`(皮层放电率,Hz)是非负、单极性信号,无法直接套同一带通+幅度阈值流水线。统一策略:**对 r_ctx 做相同形态变换后再走相同算法**:

```python
r_ctx_zm = r_ctx - r_ctx.mean()        # 中心化
r_ctx_so = sosfiltfilt(sos, r_ctx_zm)  # 同一 0.2–4 Hz 带通
# 同样 find_peaks + 幅度筛选,幅度阈值用 r_ctx_so 标准差的 σ_thresh × std
# 经验值 σ_thresh = 1.5(待在 Seed B 上标定)
```

这样保证两侧算法在"形态域(滤波后零均值时间序列)"上等价。MAP r_ctx 的 std 经验值 ~3 Hz,阈值 ~4.5 Hz,与 V7 现行的 6 Hz 硬阈值同量级。

---

## 2. EEG-native MI:Tort 2010 全 EEG 版本

### 算法

```python
PHASE_BAND  = (0.5, 1.5)    # Hz
AMP_BAND    = (10.0, 14.0)  # Hz
N_PHASE_BINS = 18

sos_p = butter(4, PHASE_BAND, btype="band", fs=1000, output="sos")
sos_a = butter(4, AMP_BAND,   btype="band", fs=1000, output="sos")
phase = np.angle(hilbert(sosfiltfilt(sos_p, eeg_raw)))     # ∈ [-π, π]
amp   = np.abs (hilbert(sosfiltfilt(sos_a, eeg_raw)))      # ≥ 0

bins  = np.linspace(-np.pi, np.pi, N_PHASE_BINS + 1)
mean_amp_per_bin = np.array([
    amp[(phase >= bins[k]) & (phase < bins[k+1])].mean()
    for k in range(N_PHASE_BINS)
])
p = mean_amp_per_bin / mean_amp_per_bin.sum()
MI = (np.log(N_PHASE_BINS) - (-np.sum(p * np.log(p + 1e-12)))) / np.log(N_PHASE_BINS)
```

MI ∈ [0, 1],生理预期 0.01–0.05。**SO 相位与 spindle 振幅都来自同一 raw EEG**,不再混用 r_proxy。

### simulator 端对应改动

`r_ctx` 提供 SO 相位(同 §1 的 `r_ctx_so` 经 Hilbert 取 angle),`r_thal` 提供 spindle 振幅(走 [10,14] Hz 带通 + Hilbert)。`r_thal` 是丘脑放电率,与皮层 spindle 在物理上对应一致,且非负不影响 Hilbert。

---

## 3. 代码改动清单

| 文件 | 改动 |
|------|------|
| `S4_v7_repair/compute_pac_metrics_eeg_native.py`(新) | 实现 `compute_t6_eeg_native(signal, fs, mode='eeg'|'r_ctx')` 与 `compute_mi_eeg_native(sig_for_phase, sig_for_amp, fs)` 两个独立函数 |
| `S4_sbi/compute_xobs_from_eeg_v4.py`(新) | 复用 v3 的 `load_n3_eeg` + `build_rate_proxy`(只为 eeg_raw),调上面新模块算 T6 / MI;写入 `x_obs_v4.npz`,SUMMARY_KEYS 重回 7 维:`[shape_r, T4_q, T4_freq, T6_ibi_cv, T8_n_sp_events, T11_lag_ms, MI]` |
| `models/s4_personalize_fig7_v7.py` | 在 `compute_constraints_v7` 中,T6 改用新模块的 r_ctx 路径(替换现有阈值穿越);PAC 计算(T9_mi)切换到新模块;**保留旧函数作 fallback,用 flag 控制** |
| `S4_sbi/simulator_wrapper.py` | `SUMMARY_KEYS` 由 5 改回 7,补 T6_ibi_cv 与 MI;默认 x_obs 路径同步改为 `x_obs_v4.npz` |
| `S4_sbi/run_sbi.py` | 默认 `--x-obs` 改 `x_obs_v4.npz`;`ROUND_SIMS` 保持 `[2000,1000,1000,1000]`;`b` 上界保持 80;**重命名输出归档:`sbi_outputs_5dim_archive_20260523/`** |

---

## 4. 验证步骤(三层)

### 4.1 单信号合成测试

构造 5 个合成信号(SO+spindle 全耦合、纯 SO、纯 spindle、纯白噪、SO 与 spindle 反相),验证 T6 在 [0.3, 0.6]、MI 在 [0.01, 0.05] 范围内对全耦合信号给出最高值。脚本路径:`valid_scripts/validate_t6_mi_eeg_native.py`。

### 4.2 Seed B 单次仿真验证

用 Seed B params 跑 `simulator_wrapper.simulator(theta_B)`,逐项检查:

| 维度 | 预期 | 容差 |
|------|------|------|
| shape_r | 0.67–0.68 | ±0.01 |
| T4_q | 2.3–2.4 | ±0.1 |
| T4_freq | 1.0–1.3 Hz | ±0.2 |
| **T6_ibi_cv** | **0.30–0.60** | hard window |
| T8_n_sp_events | 15–25 | ±5 |
| T11_lag_ms | 3.0–4.0 | ±0.5 |
| **MI** | **0.05–0.10** | Seed B 表 0.07,容差 ±0.03 |

T6 落在 [0.3, 0.6] **必须满足**,否则算法本身有问题。

### 4.3 SC4001 EEG 端验证

跑 `compute_xobs_from_eeg_v4.py`,核对新 x_obs 的 T6 和 MI:

| 维度 | 预期范围 | 备注 |
|------|---------|------|
| T6_ibi_cv | 0.40–0.55 | 健康 N3 IBI 变异 |
| MI | 0.02–0.04 | SC4001 在 EEG 数据库中属于中等 PAC 强度 |

若任一项落在生理范围外,**先 debug 算法,不要进 SBI**。

---

## 5. 工作量预估

| 任务 | 工时 |
|------|------|
| 新增 2 个算法文件 + 单元测试 | 4 h |
| v4 x_obs 抽取 + 调通 SC4001 | 2 h |
| simulator 端 r_ctx 路径改造 + Seed B 验证 | 4 h |
| `compute_constraints_v7` T6/MI 嵌入 + flag 兼容 | 2 h |
| 合成信号验证(`validate_t6_mi_eeg_native.py`) | 2 h |
| run_sbi.py 配置调整 + dry-run 检查 | 1 h |
| **代码 + debug 合计** | **~15 h(2 个工作日)** |
| 7 维 SBI 主 run | ~9–10 h(同 May 23 节奏) |
| 后处理 + 与 5 维 baseline 对比 + 报告 | 3 h |
| **总计** | **~3 个工作日 + 1 晚算力** |

完成后产出:`sbi_outputs/`(新 7 维)+ `sbi_report_0511.md` 新增 §3a(7 维结果) + c_ctx2th 后验在 5→7 维之间的收窄量化对比。
