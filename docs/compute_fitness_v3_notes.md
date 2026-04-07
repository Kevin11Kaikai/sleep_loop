# `compute_fitness_v3()` 说明（`s4_personalize_fig7_v3.py`）

本文档汇总对 **`compute_fitness_v3`** 的讨论与阅读要点：仿真对象 **`m`**、发放率 **`r_exc` / `r_ctx`**、**Welch 长度**、**插值到 `fooof_freqs`**、**与 `compute_target_periodic` 同构的 shape_r 流程**、**适应度加权与返回值**。

**代码位置**：`models/s4_personalize_fig7_v3.py` 中 **Part 3**（`compute_fitness_v3` 及前后全局变量）。

**更细的逐段代码 walkthrough**（含英文代码块对照）：`docs/s4_personalize_fig7_v3_compute_fitness_v3.md`。

**相关**：目标 PSD → `docs/load_target_psd_notes.md`；目标 FOOOF 残差 → `docs/compute_target_periodic_notes.md`。

---

## 1. 函数在整体流程中的角色

- **差分进化（DE）** 每尝试一组 **8 维参数** 调用 **一次**；成功时返回 **`-fitness`**（`scipy.optimize.differential_evolution` **最小化**该值 ⇒ **最大化 `fitness`**）；仿真失败、burn-in 后皮层过弱、或部分异常路径返回 **`0.0`**（视为最差）。  
- **输入**：当前参数向量；启动阶段算好的 **`target_psd` / `target_freqs`**；**`target_periodic` / `fooof_freqs`**（来自 **`compute_target_periodic`**）。  
- **输出（副作用）**：更新 **`_eval_count`、`_records`、`_best_score` / `_best_params`**。

---

## 2. `m = build_model(...); m.run()` 里的 `m` 是什么？

- **`m`** 为 **neurolib `MultiModel`**，内部是脚本定义的 **`ThalamoCorticalNetwork`**：**ALN 皮层节点 + 丘脑节点**，耦合 **`c_th2ctx` / `c_ctx2th`**。  
- **`m.run()`** 按 **`SIM_DUR_MS`**、**`dt` / `sampling_dt`** 等积分，得到各输出时间序列（如 **`r_mean_{EXC}`**）。

---

## 3. `r_exc`、`r_ctx`、`r_thal` 与 `×1000`

- **`r_exc = m[f"r_mean_{EXC}"]`**：兴奋群体 **平均发放率**；**`EXC`** 为 neurolib 常量，键名形如 **`r_mean_E`**。  
- **形状**：双节点时常见 **`(2, T)`**——**第 0 维**为 **节点**（**0=皮层、1=丘脑**），**第 1 维**为 **时间**（如 30 s×1000 Hz = 30000；burn-in 后常 **25000**）。  
- **`if r_exc.ndim == 2 and shape[0] >= 2`**：分别取两行 → **`r_ctx` / `r_thal`**；**否则** 仅一条作皮层，**`r_thal` 全零**，保证下游始终有两条 1D 序列。  
- **`×1000`**：**neurolib 常用 kHz 标度** → 脚本统一为 **Hz**，与 **`welch(..., fs=FS_SIM)`**、**`UP_THRESH_HZ`** 等一致。

---

## 4. Burn-in 后长度（为何常是 25000？）

- 总长 **30 s × `FS_SIM`（1000）= 30000** 个采样点；**丢掉前 5 s** ⇒ **30000 − 5000 = 25000**。  
- 这是 **时间序列长度**，**不是** Welch 之后的频率点数。

---

## 5. Welch：`r_ctx` 很长，为何 `f_ctx` / `p_ctx` mask 后约 196？

- **Welch** 输出的是 **「每个频率一个 PSD 值」**；**频率个数**主要由 **`nperseg`** 决定（约 **`nperseg/2+1`** 覆盖 0～Nyquist），**不**等于 **`len(r_ctx)`**。  
- 本脚本 **`nperseg = min(10·FS_SIM, len(r_ctx))`**，默认常 **10000** ⇒ **Δf ≈ fs/nperseg = 0.1 Hz**。  
- 再 **`mask` 到 `[F_LO, F_HI]`（0.5～20 Hz）** 后，频点约 **0.5, 0.6, …, 20.0** ⇒ **(20−0.5)/0.1+1 = 196**（参数变化时此数会变）。

---

## 6. `interp1d(f_ctx, p_ctx)(fooof_freqs)` 是否多余？

- **默认配置**下，**mask 后的 `f_ctx`** 与 **`fooof_freqs`**（目标 EEG FOOOF 的频率轴）**常对齐**，插值在节点上取值 ⇒ **近似恒等**。  
- **仍保留插值**的原因：**目标 `fs_eeg`、仿真剩余长度、`nperseg`、FOOOF 内部网格** 任一变化时，**Welch 与 `fooof_freqs` 可能不同步**；插值保证 **`fm_sim.fit(fooof_freqs, p_interp, …)`** 与 **`target_periodic`** **共用同一频率轴**。  
- **`p_interp`** 长度恒为 **`len(fooof_freqs)`**；**`p_ctx`** 为 **`len(f_ctx)`**——二者仅在网格一致时长度相同。

---

## 7. `shape_r` 路径与 `compute_target_periodic` 的关系

- **构造 log 域残差**的配方 **一致**：**`log10(线性 PSD + ε)[:len(_ap_fit)] − fm._ap_fit`**（峰 **不** 单独减掉，只减非周期背景）。  
- **区别**：**目标侧**用 **`target_psd`** 与 **`compute_target_periodic`** 里拟合的 **`fm`**；**仿真侧**用 **插值后的 `p_interp`** 与 **新实例 `fm_sim`**——**各自独立拟合**，**`_ap_fit` 不可混用**。  
- 之后 **`pearsonr(sim_periodic, target_periodic)`** 比 **形状**；**`r < 0`** 截为 **0**，避免 **`0.35*shape_r`** 为负。

**顺序敏感**：须 **先** 把仿真线性功率插值到 **`fooof_freqs` 再 FOOOF**，**不能** 与「先在原生 Welch 网格 FOOOF 再插残差」混用（与 JSON 中 **`shape_r`** 不一致）；详见 `s4_personalize_fig7_v3_compute_fitness_v3.md` 末段。

---

## 8. FOOOF 不可用时的 fallback

- 在 **`target_freqs`** 上对仿真 PSD 插值，与 **`target_psd`** 比较 **加权 log 差**（**χ² 型**），映射为 **0～1** 标量代替 **`shape_r`**。

---

## 9. `so_power` / `spindle_power`

- 再次对 **`fooof_freqs`** 插值并 **FOOOF 拟合**，读 **`peak_params_`**：峰中心落在 **慢波带 [0.3, 1.5] Hz** 或 **纺锤带 [8, 16] Hz** 时，取 **log 域峰高** 的最大值作为分项（实现上与 shape_r 路径略重复，便于读峰表）。

---

## 10. `dynamics_score` 与总适应度

- **`compute_dynamics_score_v3(r_ctx, r_thal)`**：时域 **T1–T5**（DOWN/UP/持续、SO 峰频、纺锤 FWHM 等），返回 **[0, 1]**。  
- **`fitness = 0.35*shape_r + 0.15*so_power + 0.15*spindle_power + 0.35*dynamics_score`**。

| 项 | 权重 | 含义（通俗） |
|----|------|----------------|
| `shape_r` | 0.35 | 去 1/f 后频谱形状与目标 EEG 的相关 |
| `so_power` | 0.15 | FOOOF 慢波带峰强度 |
| `spindle_power` | 0.15 | FOOOF 纺锤带峰强度 |
| `dynamics_score` | 0.35 | 时域是否像 N3 慢波–纺锤动力学 |

---

## 11. 源码中的中文注释

`compute_fitness_v3` 内已按上述要点在对应代码块旁增加 **`# 中文：...`** 注释，便于与本文档对照阅读。

---

*文档根据对话整理；若 `compute_fitness_v3` 实现变更，请以 `models/s4_personalize_fig7_v3.py` 为准。*
