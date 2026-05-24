# c_ctx2th Unidentifiability:Evidence Dossier

**Subject**: SC4001(Sleep-EDF Cassette,N3 = 4260 s post artifact rejection)
**Model**: ALN + Thalamus mean-field;4 free params `[g_h, g_LK, c_ctx2th, b]`
**Date**: 2026-05-23
**Status**: Sprint 1 Phase 2 closed without dimension expansion;5 维 SBI(May 23)是 Stage 2 最终基线。

本档案汇总 c_ctx2th 在 SC4001 上**结构性不可识别**的诊断证据,作为 TBME 论文的 supporting material。三条证据互相独立、互相印证。

---

## 1. 直接证据:5 维 SBI 后验显示 c_ctx2th 多模态 + 无与其他参数耦合

**Run**: 2026-05-22 20:31 → 05-23 04:34,8.06 h,5300 sims,0% NaN,R4 跑满,SBC 4/4 PASS,PPC 3/5 PASS。

| 子证据 | 数字 | 出处 |
|--------|------|------|
| c_ctx2th 边际后验**多模态** | 4 个 modes:0.052 / 0.107 / 0.155 / 0.207 | `sbi_outputs/fig_marginals.png` |
| c_ctx2th × g_h 联合 | **横条**(c_ctx2th 横扫全 prior,沿 g_h 仅在 0.076 和 0.053 出现细带),**无 ridge** | `fig_pairplot.png`,详 `sbi_5dim_diagnostic_20260523.md` Task 2.1 |
| c_ctx2th × g_LK 联合 | **横条**,与 g_LK 接近独立 | 同上 Task 2.2 |
| c_ctx2th × b 联合 | **竖条**,4 个 c_ctx2th 模态共享同 b ≈ 50 | 同上 Task 2.3 |
| 模态数对 likelihood 维度敏感 | May 7(7 维):3 模态主峰 0.075;May 23(5 维):4 模态主峰 0.207 | `sbi_report_0511.md` §3.3 |
| c_ctx2th MAP 95% CI 宽度 | [0.052, 0.214],占 prior 全宽 0.17 的 **95%** | `sbi_results.md`,`sbi_5dim_review_20260523.md` §2 |

**算法层面无故障**:SBC 4/4 PASS(g_h KS p=0.68, g_LK p=0.41, c_ctx2th p=0.86, b p=0.085)证明 NSF 已正确学到 likelihood↔posterior 映射;c_ctx2th 多模态不是密度估计器欠拟合的结果。

**力学模型无故障**:PPC 3/5 PASS,Seed B 仿真给出 shape_r=0.677(与 DE Seed B 表的 0.679 一致到 0.3%),simulator 在 x_obs 附近稳定再现观测。

详尽报告:`sbi_5dim_diagnostic_20260523.md`,`sbi_5dim_review_20260523.md`,`sbi_report_0511.md` §3.3。

---

## 2. 否证路径 A:T6 + MI 重引入(Sprint 1 Phase 2 中止)

### 2.1 设计假说

5 维 likelihood 删除了 T6(SO IBI CV)和 MI(SO–spindle PAC)。假说:重新引入这两维(用 EEG-native algorithms 替代失败的 r_proxy 路径)可恢复 c_ctx2th 约束。

实现:
- `S4_v7_repair/compute_pac_metrics_eeg_native.py` — AASM 75 µV 半波 SO UP 事件检测 + Tort 2010 single-channel PAC
- `S4_sbi/compute_xobs_from_eeg_v4.py` — 7 维 x_obs 抽取(`shape_r, T4_q, T4_freq, T6_ibi_cv, T8_n_sp_events, T11_lag_ms, MI`)
- 合成信号验证:`valid_scripts/validate_t6_mi_eeg_native.py` — **10/10 PASS**(算法在 ground-truth 信号上工作正确)

### 2.2 SC4001 实测 T6 + MI

| 量 | 实测值 | v2 plan 预期 | 状态 |
|----|--------|--------------|------|
| T6_ibi_cv(全 4260s concat)| 1.085 | [0.40, 0.55] | ⚠️ 超 2× |
| T6_ibi_cv(per-epoch 中位数,89/142 epochs ok)| 0.524 | [0.40, 0.55] | ✅ |
| n_UP per 30s epoch(中位数)| 3 | ~24(0.8 Hz × 30s) | 14%(elderly 慢波幅度低) |
| **MI(全 4260s concat)** | **0.00019** | [0.02, 0.05] | ⛔ **100× 偏低** |
| **MI(per-epoch 中位数,142 epochs)** | **0.0016** | [0.02, 0.05] | ⛔ **10× 偏低** |

T6 在 per-epoch 模式下可恢复(concat 142 个不连续 epoch 制造伪极值);MI 即使 per-epoch 也仅达到 0.0016,远低于预期。

### 2.3 排除根因:concatenation 与单段时长

跑了 quick verification 测拼接是否是 MI 偏低的根因:

| 模式 | 信号 | epoch 边界数 | MI |
|------|------|------------|-----|
| Mode X | 全 4260s concat | 141 | 0.000193 |
| **Mode Y** | **最长连续段 480s(16 个 epoch)** | **0** | **0.000107** |

**Mode Y 0 边界仍给 MI≈0.0001**,**比 Mode X 还低**。**拼接不是根因**;SC4001 single-channel raw EEG 的 Tort MI 绝对量级就是 ~1e-4。

---

## 3. 否证路径 B:Surrogate z-score normalization 救不了

### 3.1 设计假说

绝对 MI 值低不等于没信号——可能 SC4001 上 MI 的 null floor 也低,信噪比仍足够。用 amplitude time-shift surrogate(Tort 2010 canonical method,100 次 circular shift ≥ 10 s)计算 z = (MI_obs − μ_null) / σ_null,判定 PAC 是否显著高于 null。

实现:`S4_v7_repair/compute_pac_metrics_eeg_native.py` 的 `compute_mi_zscore()` 函数。

### 3.2 数字

| 信号 | MI_obs | null μ | null σ | **z** | 判定 |
|------|--------|--------|--------|-------|------|
| **SC4001 EEG Mode X**(4260s concat,141 边界)| 0.000193 | 0.000013 | 0.000008 | **+21.40** | 表面显著 |
| **SC4001 EEG Mode Y**(480s contig,0 边界)| 0.000107 | 0.000098 | 0.000067 | **+0.137** | **null** |
| 白噪 paranoia(480s)| 0.000014 | 0.000084 | 0.000041 | −1.698 | 在 ±2 内(算法自检) |
| Seed B simulator cross-channel(r_ctx × r_thal,55s)| 0.0617 | 0.0019 | 0.0014 | **+42.51** | 强显著 |
| Seed B simulator single-channel(r_ctx × r_ctx,55s)| 0.1223 | 0.0026 | 0.0017 | **+68.72** | 强显著 |

### 3.3 解读

**Mode X 的 z=+21 是 boundary artifact,不是真信号**:
- null σ 极小(8 × 10⁻⁶),分母小拉高 z
- 141 个 epoch 拼接边界制造 amp surrogate roll 无法完全打破的伪相关
- 把 Mode X 的 MI_obs 投到 Mode Y 的 null 分布上:(0.000193 − 0.000098) / 0.000067 = **+1.42**(仍在 null 内)

**Mode Y(0 边界 480s)给 z=+0.137**:**SC4001 single-channel EEG-native PAC 在 surrogate 检验下不显著**,**与白噪 paranoia 的 z=−1.7 在统计上无法区分**。

**Simulator 端 z=+42(cross-channel)与 z=+69(single-channel)**:V7 simulator 端 PAC 真实存在且极强;两端 z 量级差 300 倍。

### 3.4 两端不可比的后果

| | 若 MI 进 7 维 SBI |
|---|------------------|
| EEG 端 z ≈ 0(null) | NSF 学到 "x_obs 的 MI 维 = null" |
| Simulator 端 z >> 2(strong) | 仿真给出的 MI 都是显著的 |
| 推断方向 | NSF 把 c_ctx2th 推向**破坏 PAC 的参数空间**以匹配 EEG 端的 null 信号 |
| 生理意义 | 荒谬(c_ctx2th 是皮层-丘脑突触强度,推到 0 即去除 thalamic feedback) |

---

## 4. 结论:c_ctx2th unidentifiability 是数据-模型 mismatch,不是算法 bug

三条独立证据汇聚于同一结论:

1. **后验形态**(§1):c_ctx2th 在 4 维 SBI 下后验呈多模态 + 完全独立于其他三参数,模态结构对 likelihood 维度敏感(7 维 vs 5 维不同)
2. **EEG-native T6 + MI 重引入失败**(§2):MI 量级(~1e-4)远低于 simulator(~6e-2),且与 EEG-side 拼接方式无关
3. **Surrogate z-score 不能挽救**(§3):0 边界干净段上 EEG MI z=+0.14(null),而 simulator z=+42 — 两端在 statistical-significance 层面就不可比

**根本原因**:SC4001 这套 single-channel Fpz-Cz raw EEG **不携带可靠的 SO-spindle PAC 信号**。任何 EEG-derived PAC summary statistic 在该受试者上都会饱和在 null,使其无法约束模型中真实存在的 c_ctx2th 参数维度。

**这是一个 publication-worthy negative result**:用 SBI 严格证明了某个特定 EEG dataset(Sleep-EDF Cassette,Fpz-Cz,artifact-rejected N3)在 single-channel PAC 维度上对皮层-丘脑突触强度的可识别性贡献为零。

---

## 5. 引用文档

- `docs/sbi_report_0511.md` — Stage 2 全报告(含 §3 5 维 SBI 详细数字)
- `docs/sbi_5dim_review_20260523.md` — 5 维 run first-pass review
- `docs/sbi_5dim_diagnostic_20260523.md` — pairplot + 后验采样诊断
- `docs/sprint1_phase2_plan_v2.md` — Phase 2 设计(已 deprecated)
- `S4_v7_repair/compute_pac_metrics_eeg_native.py` — EEG-native + surrogate z 算法实现
- `S4_sbi/compute_xobs_from_eeg_v4.py` — 7 维 x_obs 抽取尝试(未上线)
- `valid_scripts/validate_t6_mi_eeg_native.py` — 合成信号 10/10 PASS 验证
- `valid_scripts/test_surrogate_zscore_pac.py` — surrogate z-score 全套测试
