# `compute_target_periodic()` 说明（`s4_personalize_fig7_v3.py`）

本文档汇总对 **`compute_target_periodic`** 的讨论：FOOOF 如何拟合 **目标 N3 PSD**，如何得到 **`target_periodic` / `fooof_freqs`**，以及 **峰参数、非周期指数、窄带过剩** 等概念与常见误解。

**代码位置**：`models/s4_personalize_fig7_v3.py` 中 **Part 2**（约第 313–329 行）。

**前置输入**：由 **`load_target_psd()`** 得到的 **`target_psd`**（线性 PSD）、**`target_freqs`**（Hz）。详见 `docs/load_target_psd_notes.md`。

---

## 1. 函数在整体流程中的角色

- 在 **差分进化开始前调用一次**（若安装 FOOOF；否则 **`HAS_FOOOF`** 为假时返回 **`None, None`**）。  
- 对 **经验目标谱** 做一次 FOOOF 拟合，并构造 **log 域「去背景」残差** **`target_periodic`**，供 **`compute_fitness_v3`** 中 **`shape_r`** 与仿真谱对齐比较（仿真侧在相同 **`fooof_freqs`** 网格上重复同一套流程）。  
- **含义**：把 **平滑非周期背景** 从 **log₁₀(PSD)** 里减掉，得到 **压平后的残差谱**（突出 **窄带过剩**；**不是** 线性标度下的「周期分量 PSD」）。

---

## 2. 逐步在做什么？

| 步骤 | 代码意图 |
|------|-----------|
| **`FOOOF(...)`** | 设定峰宽范围、最多峰数、峰高阈值、非周期模式（见下表）。 |
| **`fm.fit(target_freqs, target_psd, [F_LO, F_HI])`** | 在 **`[F_LO, F_HI]`** 内，用 **线性功率** **`target_psd`** 与频率 **`target_freqs`** 联合拟合：**非周期背景 + 至多 4 个窄带峰**（在 **log 功率域** 建模）。 |
| **`fooof_freqs = fm.freqs`** | FOOOF **内部** 使用的频率轴（Hz），与后续 **`fm._ap_fit`**、**`target_periodic`** 长度一致。 |
| **`target_log`** | **`np.log10(target_psd + 1e-30)`**：经验 PSD 的 **log₁₀**（避免 log(0)）。 |
| **`target_periodic`** | **`target_log[:len(fm._ap_fit)] - fm._ap_fit`**：在 **log 域** 减去 **拟合的非周期曲线** → **残差**（窄带鼓包 + 背景拟合误差等）。 |
| **打印** | 遍历 **`fm.peak_params_`** 打印各峰；打印 **`fm.aperiodic_params_[1]`** 作为 **非周期指数**。 |

**MNE / Welch**：本函数 **不** 读原始 EEG；只消费已算好的 **`target_psd` / `target_freqs`**。

---

## 3. FOOOF 构造参数与 `fit` 入参（物理 / 数据含义）

| 符号 | 含义 |
|------|------|
| **`target_freqs`** | 与 **`target_psd`** 逐点对齐的频率（Hz），来自 Welch 掩码后的网格。 |
| **`target_psd`** | **线性 PSD**（约 **V²/Hz** 量级），N3 多 epoch 平均后的目标谱。 |
| **`F_LO`, `F_HI`** | 传给 **`fit(..., [F_LO, F_HI])`** 的 **拟合频段**；与 Config 中 **0.5–20 Hz** 一致。 |
| **`peak_width_limits=[1.0, 8.0]`** | 每个峰 **FWHM（Hz）** 允许落在 **1–8 Hz**，约束峰过窄/过宽。 |
| **`max_n_peaks=4`** | 最多 **4** 个加性窄带峰。 |
| **`min_peak_height=0.05`** | 峰的最小高度阈值（**log₁₀ 域**，相对背景），抑制弱假峰。 |
| **`aperiodic_mode="fixed"`** | 非周期项 **无膝点**：在拟合频段内用 **单一幂律型背景**（offset + exponent），而非带 **knee** 的分段模型。 |

---

## 4. `fm._ap_fit` 是什么？如何得到？

- **`fm._ap_fit`** 是 **拟合得到的非周期分量** 在 **`fm.freqs`** 上 **逐点取值** 的向量，与 **`target_log`** 一样在 **log₁₀ 功率** 域。  
- **不是** 单独一个标量（如指数）；**非周期指数** 是 **`fm.aperiodic_params_`** 的一部分，库用其 **重建整条背景曲线** 再得到 **`_ap_fit`**。  
- **来源**：**`fm.fit`** 完成数值优化后，用估好的 **aperiodic 参数** 在内部频率网格上 **求值** 得到（实现细节见 `fooof` / `specparam` 源码；下划线表示偏内部 API）。

---

## 5. 返回值：`target_periodic` 与 `fooof_freqs`

| 返回值 | 含义 | 常见误解纠正 |
|--------|------|----------------|
| **`target_periodic`** | **log₁₀(经验 PSD) − 非周期拟合**，即 **log 域残差 / 压平谱**。 | **不是** 线性 **PSD（V²/Hz）**；**不是** 对时间序列再 Welch 得到的「纯周期 PSD」。残差里含 **窄带过剩**，也含 **背景拟合误差**。 |
| **`fooof_freqs`** | 与 **`target_periodic`** **一一对应** 的频率数组（Hz）。 | 与 Welch 原始 **`target_freqs`** 可能 **长度/网格略有不同**（FOOOF 重采样到内部网格）；故 **`shape_r`** 路径显式在 **`fooof_freqs`** 上对齐仿真。 |

**措辞建议**：称 **「去非周期背景后的 log 域残差」** 比 **「经验周期信号 PSD」** 更严谨。

---

## 6. 打印输出：「FOOOF target peaks」与 `Aperiodic exponent`

- **「target peaks」**：指在 **`target_psd`** 这条 **目标谱** 上，FOOOF **拟合出的窄带峰**（最多 4 个），不是事先给定的目标列表。  
- 对 **`peak_params_` 每一行**（顺序以所用库版本文档为准，常见为 **CF、power、width**）：  
  - **Hz**：峰 **中心频率**。  
  - **power**：**log₁₀ 域** 与峰幅度相关的量（**不是** 直接等于线性 µV²/Hz）。  
  - **width**：通常 **FWHM（Hz）**，峰在频率上的宽度。  

**`Aperiodic exponent: …`**（代码第 327 行）打印的是 **`fm.aperiodic_params_[1]`**，即 **`fixed`** 模式下的 **非周期指数（常记 χ）**，描述 **背景随频率的陡缓**（与 **1/f^χ** 类幂律在功率域的图像一致，具体常数项见库公式）。

- **不是** 对四个峰做 **线性回归** 算出来的；**指数与全部峰参数是 `fit` 时联合优化** 的同一次解。

---

## 7. 窄带过剩与「周期 / 非周期」

- **窄带过剩**：在 PSD 上，相对 **平滑 1/f 型背景**，某一 **窄频带** 内 **多出来的功率**（谱形 **局部鼓起**）。FOOOF 的 **峰** 就是在 log 域刻画这种过剩。  
- **准周期节律**（纺锤、δ 等）在谱上常表现为 **窄带过剩**；但 **鼓包** 也可能混有 **泄漏、谐波、伪迹** 等，**不能** 与「严格周期信号」一一等同。  
- **非周期（aperiodic）**：在本流程里主要指 **FOOOF 拟合的那条平滑背景**；减背景 **不是** 减「噪声」一词意义上的随机项，而是减 **模型化的非周期谱分量**。

---

## 8. 下游衔接（便于对照）

- **`compute_fitness_v3(..., target_periodic, fooof_freqs)`**：对仿真皮层率 **Welch** 后，将 **线性功率** 插值到 **`fooof_freqs`**，再 FOOOF 得 **`fm_sim._ap_fit`**，算 **`sim_periodic = log10(p_interp) - ap_fit`**，与 **`target_periodic`** 算 Pearson **`shape_r`**。顺序与网格与 `load_target_psd` / 本函数约定一致；详见 `docs/s4_personalize_fig7_v3_compute_fitness_v3.md` 及 `s4_personalize_fig7_v3.py` 中 **Part 3** 注释。

---

## 9. 相关配置常量（同文件 Config）

| 常量 | 典型值 | 与本函数关系 |
|------|--------|----------------|
| `F_LO`, `F_HI` | 0.5, 20 Hz | **`fit`** 的频率范围；与目标 Welch 频段一致。 |

FOOOF 超参数（`peak_width_limits`、`max_n_peaks` 等）写在本函数体内，修改即改变拟合行为。

---

## 10. 参考与依赖

- 库：**FOOOF**（或迁移后的 **specparam**）；参数名与 **`peak_params_` 列顺序** 以当前安装版本文档为准。  
- 前置文档：**`docs/load_target_psd_notes.md`**。

---

*文档根据对话整理；若 `compute_target_periodic` 实现变更，请以 `models/s4_personalize_fig7_v3.py` 为准。*
