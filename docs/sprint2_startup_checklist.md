# Sprint 2 Startup Checklist

**前置事实**(2026-05-23 closeout):
- Stage 2 SBI 最终基线已确定:5 维 May 23 run(8.06 h,SBC 4/4 PASS,PPC 3/5 PASS)
- Sprint 1 Phase 2(EEG-native T6 + MI)已 closeout,c_ctx2th unidentifiability 转为 publication-worthy negative result
- Sprint 2 范围:**4 个 post-hoc validation** 在现有 5 维后验 + Seed B simulator 之上进行,不再扩 likelihood 维度
- 主线代码冻结;Sprint 2 全部新增脚本独立目录,不动 `S4_sbi/` 主管线

---

## 4 个 validation 概览

| # | 名称 | 目的 | 主要工具 | 已有 vs 新写 |
|---|------|------|---------|------------|
| 1 | noise-off | 仿真器在零噪声下能否仍生成 SO+spindle,确认结构是内源性而非噪声-induced | V7 build_model + 关 OU 噪声 + 7 个 T1–T12 检查 | **小量新写** |
| 2 | Rayleigh | SO phase 在 spindle peak 时刻的分布是否单峰集中,而非均匀 | scipy.stats.rayleigh + 现有 PAC 流水线 | **大部分已有** |
| 3 | PAC surrogate | 仿真器侧 PAC 是否在 z-score 检验下显著(我们已有 z=+42 单点,需多个后验样本扫一遍) | **`compute_mi_zscore` 已实现** | **几乎全已有** |
| 4 | spindle morphology | 仿真生成的 spindle 是否在 carrier 频率、envelope 形状、duration 上 EEG-realistic | scipy bandpass + envelope + Gaussian fit | **中等量新写** |

---

## 各 validation 拆解

### 1. noise-off(推荐首发)

**核心问题**:V7 simulator 用 Ornstein-Uhlenbeck 输入注入随机驱动;若把 OU 噪声 std → 0,Seed B 参数还能不能自发产生 SO+spindle 结构?

**已有工具**:
- `models/s4_personalize_fig7_v7.py:build_model` 接收 `sigma_ou`(可调到 0)
- `compute_constraints_v7` 给出 T1–T12 完整诊断
- `valid_scripts/test_surrogate_zscore_pac.py` 提供 v7 importlib + 单次仿真的样板代码

**需要新写**(估 3-4 h):
- `valid_scripts/validate_noise_off.py`:
  - 用 Seed B params 各跑一次:`sigma_ou=0`、`sigma_ou=0.5 × default`、`sigma_ou=default`、`sigma_ou=2 × default`
  - 对每次输出运行 `compute_constraints_v7` 拿 T1-T12
  - 输出表格:四档噪声下 T4_freq、T6_ibi_cv、T8_n_sp_events、T11_lag_ms、MI 各自的值
- 预期:噪声 = 0 时 T8 不应跌到 0(否则 spindle 是噪声 forced),T6 应仍在生理范围(否则 SO 节律靠噪声)

**通过判据**:噪声 0× → T6 < 0.6 且 T8 > 5;噪声从 0 升到 default,各指标的变化应 < 20%。

---

### 2. Rayleigh

**核心问题**:5 维 SBI 后验 mean 处的参数下,仿真器输出的 spindle 振幅极值时刻,SO 相位是否集中在一个单一角度?

**已有工具**:
- `S4_v7_repair/compute_pac_metrics_fixed.py:compute_pac_metrics` 已给 `phase_argmax`、`phase_concentration`、`bimodality_flag`
- `scipy.stats.rayleigh.test` 或 `scipy.stats.circstd`(circular Rayleigh test 需手写,约 20 行)

**需要新写**(估 2-3 h):
- `valid_scripts/validate_rayleigh.py`:
  - 从 5 维后验抽 50 个 theta 样本
  - 每个样本跑 simulator 拿 r_ctx
  - 检测 spindle envelope peaks,对每个 peak 取该时刻的 SO 相位
  - 对 50 × N_peaks 的 phase 集合做 Rayleigh test 输出 z + p
- 预期:p < 0.001 表示 phase 集中(PAC 真实);p > 0.05 表示 phase 均匀(PAC 是 artifact)

**通过判据**:p_Rayleigh < 0.001 且 Z > 5 → "spindle 显著 SO-phase 锁定"

---

### 3. PAC surrogate(工具最完整,可秒杀)

**核心问题**:5 维后验抽样下,simulator 端 PAC z-score 分布是否仍稳定 >> 2?(目前只在 Seed B 单点测过 z=+42)

**已有工具**:
- `S4_v7_repair/compute_pac_metrics_eeg_native.py:compute_mi_zscore` — **算法完全就绪**
- `valid_scripts/test_surrogate_zscore_pac.py` — Seed B 端到端样板已跑通

**需要新写**(估 1-2 h):
- `valid_scripts/validate_pac_surrogate.py`:
  - 从 5 维后验抽 50 个 theta(reuse `sbi_outputs/round4_posterior.pkl`)
  - 每个 theta 跑一次 simulator → 拿 r_ctx + r_thal
  - 调 `compute_mi_zscore(r_ctx, fs, sig_amp=r_thal, n_surrogates=100)` → z
  - 画 50 个 z 的直方图 + 报告 median/min/max
- 预期:median z > 10,min z > 2

**通过判据**:50 个 z 中 >= 95% (= 47/50) 满足 z > 2

---

### 4. spindle morphology(工作量最大)

**核心问题**:simulator 生成的 spindle 与 EEG 实测的 spindle 在形态学(carrier freq、duration、envelope shape)上是否可比?

**已有工具**:
- `S4_sbi/compute_xobs_from_eeg_v4.py` 中 T8 spindle 检测流水线([10,14]Hz bandpass + Hilbert + 0.3-2 s duration)
- V7 `compute_constraints_v7` 中 T7(spindle envelope CV)、T12(verified spindle 计数)

**需要新写**(估 4-6 h):
- `valid_scripts/validate_spindle_morphology.py`:
  - 从 5 维后验抽 30 个 theta
  - 每个跑 simulator → 拿 r_ctx → 检测 spindle events
  - 对每个 spindle event 提取:
    - carrier dominant freq(FFT)
    - duration(envelope 阈值过线)
    - envelope shape(归一化 + Gaussian fit,返回 sigma)
    - amplitude
  - 同样流水线作用在 SC4001 EEG 上拿 reference 分布
  - KS test 对比 sim vs EEG 各属性分布
- 预期:carrier freq KS p > 0.05(同分布);duration 中位数差 < 30%

**通过判据**:carrier freq 与 duration 两项 KS p 均 > 0.01

---

## 推荐执行顺序

1. **noise-off**(3-4 h)— 你倾向首发,且最容易给出二元 PASS/FAIL,适合启动节奏
2. **PAC surrogate**(1-2 h)— 工具最完整,可作为"小巧"validation 紧跟 noise-off,快速积累 2/4 的进度感
3. **Rayleigh**(2-3 h)— 与 PAC surrogate 互补,phase concentration 是 PAC 强度之外的第二种 PAC 表征
4. **spindle morphology**(4-6 h)— 最重的工作量,放最后

---

## 工作量重估(基于 5 维基线已定)

| 阶段 | 估时 | 关键产出 |
|------|------|---------|
| validation 1 noise-off | 3-4 h | T1-T12 噪声依赖表 |
| validation 2 PAC surrogate | 1-2 h | 50 个后验样本的 z 分布 |
| validation 3 Rayleigh | 2-3 h | Rayleigh Z + p |
| validation 4 spindle morphology | 4-6 h | sim vs EEG 形态学 KS |
| 跨 validation 汇总报告 | 2 h | `docs/stage2_validation_report.md` |
| **总计** | **12-17 h(≈ 2 工作日)** | |

Sprint 1 Phase 2 原本估 4-5 工作日,现在 Sprint 2 在主线不变 + 工具复用前提下 2 工作日完工,**比 Phase 2 节省 ~60%**。

---

## 启动前先确认的事

1. 5 维后验文件路径:`S4_sbi/sbi_outputs/round4_posterior.pkl` ✅(8.06h run 产物)
2. simulator_wrapper 当前是 5 维输出 — 4 个 validation 都需要直接取 r_ctx / r_thal,而非走 5 维 summary。需要一个 helper:`def run_seed_to_signals(theta) -> (r_ctx, r_thal)`,从 `test_surrogate_zscore_pac.py` 的 Part 2 复用即可
3. v4 / surrogate / phase 2 相关代码当前未 commit。Sprint 2 启动前是否需要先归档?(取决于你 1.2 的 git 操作决定)
4. Sprint 2 启动前是否需要写一份 Stage 2 closeout 给 TBME 草稿?(可能是 1.3 markdown)
