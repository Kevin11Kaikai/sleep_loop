# SBI Stage 2：观测臂 vs 仿真臂对照说明

**文档版本**：2026-05-23  
**受试者**：SC4001（Sleep-EDF Cassette，通道 Fpz-Cz）  
**相关代码**：

| 角色 | 脚本 |
|------|------|
| 观测臂 | `S4_sbi/compute_xobs_from_eeg_v3.py`（5 维，当前 5 维 SBI 主线）、`S4_sbi/compute_xobs_from_eeg_v4.py`（7 维，Sprint 1 Phase 2） |
| 仿真臂 | `S4_sbi/simulator_wrapper.py`（调用 `models/s4_personalize_fig7_v7.py`） |
| 训练 | `S4_sbi/run_sbi.py`（SNPE-C） |
| PSD 对比图 | `S4_sbi/plot_concat_vs_mean_psd.py` → `S4_sbi/scan_diagnostics/fig_concat_vs_mean_psd_*.png` |

---

## 1. 术语（避免混用）

| 中文 | 英文 | 本仓库指什么 |
|------|------|----------------|
| **观测臂** | observation / `x_obs` | 从**真实 EEG** 提取的摘要向量，推断时**固定不变** |
| **仿真臂** | simulation / `x_sim` | 给定参数 **θ**，跑**力学模型**得到的摘要向量，每次仿真不同 |
| **V7 个性化** | Stage 1 DE | `s4_personalize_fig7_v7.py` 的差分进化；**不是** SBI 单次 `simulator(θ)` 调用 |

注意：日常说的「仿真器」= `simulator_wrapper`；**不是**把 142 个 epoch 的 EEG 喂进模型。

---

## 2. 总数据流

```text
                    SC4001 真实 N3 EEG
                           |
           +---------------+---------------+
           |                               |
           v                               v
   [观测臂] compute_xobs_from_eeg_*.py     [仿真臂] simulator_wrapper(θ)
           |                               |
           | 142 epoch 拼接 ~4260 s         | 每次 60 s 数值积分
           | r_proxy + eeg_raw              | r_ctx + r_thal
           v                               v
        x_obs  (5 或 7 维)                  x_sim  (5 维；7 维为 Phase 2 目标)
           |                               |
           +---------------+---------------+
                           |
                           v
                  run_sbi.py  (SNPE-C)
                  学 p(θ | x_obs)
```

**SNPE 在学什么**：在**固定的** `x_obs` 下，哪些 **θ** 能让 **`x_sim` 在统计上靠近 `x_obs`**（各维独立 z-score 后训练）。

---

## 3. 观测臂：输入 → 过程 → 输出

### 3.1 输入

| 来源 | 内容 |
|------|------|
| `data/manifest.csv` | SC4001 的 PSG、催眠图路径 |
| Sleep-EDF EDF | 单通道 `EEG Fpz-Cz`，原生约 100 Hz |
| 分期 | 仅 **N3** epoch |
| 质控 | 峰峰值 > 200 µV 的 epoch 剔除 |

典型规模：约 220 个 N3 epoch，拒绝约 78 个，**接受约 142 个**，拼接后约 **4260 s**。

### 3.2 中间信号（Step 2，v3/v4 相同）

```text
eeg_uv (100 Hz, 拼接)
    → resample 1000 Hz
    → eeg_raw = detrend(线性)          # 有正负号，保留纺锤频段
    → r_proxy = |eeg_raw| → 50 ms 高斯平滑 → 按 95% 分位缩放到 [0, 60]
```

| 信号 | 用途 |
|------|------|
| `eeg_raw` | T6(v4)、T8、MI、T11(v4) |
| `r_proxy` | T4_q、T4_freq |

### 3.3 输出：`x_obs.npz`

| 键 | 说明 |
|----|------|
| `values` | `float32` 向量，顺序 = `SUMMARY_KEYS` |
| `keys` | 名称列表 |
| `extraction_metadata` | JSON：版本、方法、诊断字段 |

### 3.4 五维（v3，`x_obs_v3.npz`）— 当前与 `simulator_wrapper` 对齐

| # | 键 | 观测端计算要点 |
|---|-----|----------------|
| 0 | `shape_r` | 固定 **1.0**（哨兵，见 `sbi_report_0511.md` §4.3） |
| 1 | `T4_q` | `r_proxy` 的 Welch PSD，SO 段 0.2–1.5 Hz |
| 2 | `T4_freq` | 同上，峰频率 |
| 3 | `T8_n_sp_events` | `eeg_raw` 10–14 Hz 事件数 / 60 s |
| 4 | `T11_lag_ms` | `compute_pac_metrics_fixed(r_proxy, eeg_raw)` 的 **up_down_ratio**（名存实比值） |

**SC4001 实测（v3）**：`[1.0, 2.645, 0.75, 15.31, 1.28]`

**刻意不含**：`T6_ibi_cv`、`MI`（r_proxy 路径上不可修复，见扫描图 `scan_diagnostics/fig_t6_*.png`、`fig_mi_*.png`）。

### 3.5 七维（v4，`x_obs_v4.npz`）— Phase 2 观测目标

在 v3 基础上 **加回** T6、MI，并改算法（`S4_v7_repair/compute_pac_metrics_eeg_native.py`）：

| # | 键 | v4 相对 v3 |
|---|-----|------------|
| 3 | `T6_ibi_cv` | **新增**：`eeg_raw` 上 AASM 式 SO UP 检测 → IBI CV |
| 6 | `MI` | **新增**：单通道 EEG-native Tort PAC（0.5–1.5 Hz 相位 × 10–14 Hz 振幅） |
| 4 | `T11_lag_ms` | 改为与 MI **同一次** `compute_mi_eeg_native` 的 up_down_ratio（不再用 r_proxy 相位） |

运行：`python S4_sbi/compute_xobs_from_eeg_v4.py` → `S4_sbi/x_obs_v4.npz`

---

## 4. 仿真臂：输入 → 过程 → 输出

### 4.1 模块加载时（一次）

```text
import simulator_wrapper
    → importlib 加载 s4_personalize_fig7_v7.py（不跑 DE main）
    → v7.load_target_psd()           # N3 epoch → 逐条 PSD → 平均
    → v7.compute_target_periodic()   # FOOOF → _target_periodic（仅 shape_r 用）
```

此处 EEG 用法与观测臂 **不同**：**平均 PSD**，不是 4260 s 拼接波形。

### 4.2 每次 `simulator(θ)` 调用

**输入**

| 类型 | 内容 |
|------|------|
| 自由参数 θ (4,) | `[g_h, g_LK, c_ctx2th, b]` |
| 固定 Seed B (4,) | `mue, mui, tauA, c_th2ctx`（写死在 wrapper） |

**过程**

```text
v7.build_model(8 参) → seed_numba(42) → m.run()  # 60 s @ 1000 Hz
    → r_ctx  = 皮层兴奋性发放率 [Hz]
    → r_thal = 丘脑发放率 [Hz]
    → 去掉前 5 s burn-in（约 55 s 用于统计）
    → v7.compute_constraints_v7(r_ctx, r_thal)  # 内部 T1–T12
    → _extract_summaries → 5 维 x_sim
```

**输出**：`x_sim (5,)`，`float64`；失败为全 `NaN`。

### 4.3 五维（当前 `simulator_wrapper.SUMMARY_KEYS`）

| # | 键 | 仿真端来源 |
|---|-----|------------|
| 0 | `shape_r` | 本次 `r_ctx` PSD 的 FOOOF 周期成分 vs `_target_periodic`（Pearson r，∈[0,1]） |
| 1–2 | `T4_q`, `T4_freq` | `con` ← **`r_ctx`** PSD（与 V7 T4 相同公式） |
| 3 | `T8_n_sp_events` | `con` ← **`r_thal`** 纺锤事件，归一化到 /60 s |
| 4 | `T11_lag_ms` | `con` ← **`compute_pac_metrics_fixed(r_ctx, r_thal)`** 的 up_down_ratio |

**内部仍算、但不输出**：`T6_ibi_cv`、`T9_mi`（在 `con` 字典里）。Phase 2 计划从 `con` 读回并做 NaN 哨兵（`T6>5` 或 `999` → 整行 NaN）。

**Seed B smoke test 参考**：`shape_r ≈ 0.68`，全约束下 `MI ≈ 0.07`（见 wrapper `__main__` 注释）。

---

## 5. 逐维对照表（观测 vs 仿真）

### 5.1 当前 5 维 SBI（v3 × 现 wrapper）

| 维 | 观测 (v3) | 仿真 (wrapper) | 信号/算法是否同构 | 备注 |
|----|-----------|----------------|-------------------|------|
| shape_r | 1.0 固定 | FOOOF(r_ctx) vs 平均 EEG 目标谱 | 否（故意） | PPC 常在 100% 分位，设计预期 |
| T4_q, T4_freq | `r_proxy` PSD | `r_ctx` PSD | 公式同，**信号不同** | 慢波频率常仍可比 |
| T8 | `eeg_raw` 事件 | `r_thal` 事件 | 检测链类似，**源不同** | 头皮 vs 丘脑模型 |
| T11 | r_proxy 相位 × eeg_raw 幅 | r_ctx 相位 × r_thal 幅 | MI 公式同族，**双通道 vs 混合** | v3 无 MI 维 |

### 5.2 计划 7 维（v4 × Phase 2 wrapper）

| 维 | 观测 (v4) | 仿真（计划） | 备注 |
|----|-----------|--------------|------|
| T6 | `eeg_raw` AASM UP → IBI CV | `r_ctx` >15 Hz → IBI CV（V7 T6） | 算法不同，物理量同类 |
| MI | 单通道 Hilbert PAC | `r_ctx`×`r_thal` cycle-by-cycle PAC | 公式同为 Tort 18 bins，**通道数不同** |

---

## 6. 长拼接 Welch PSD vs 逐 epoch 平均 PSD

问题：观测 T4 用 **整条拼接 `r_proxy` 的一次 Welch**；V7 `load_target_psd` 用 **每 30 s epoch 的 PSD 再平均**。能否观测也全用平均 PSD？

### 6.1 三种谱（SC4001，142 epoch，脚本实测）

| 方法 | T4_freq [Hz] | T4_q | 用途 |
|------|--------------|------|------|
| 长拼接 `r_proxy` Welch（**v4/v3 T4**） | **0.750** | **2.653** | `compute_xobs` |
| 逐 epoch `r_proxy` PSD 再平均 | 0.750 | 2.654 | 与上几乎重合 |
| 逐 epoch **原始 EEG** PSD 再平均（**V7 目标**） | **0.500** | **4.869** | `load_target_psd` / `shape_r` |

结论：

- 在 **r_proxy** 上，长拼接 vs epoch 平均对 **T4 几乎无差别**。
- **原始 EEG 平均谱**与 **r_proxy 谱**形状差很大——故 V7 的 shape 目标与 v4 的 T4 本来就不是同一条信号上的量。
- **T6 / T8 / MI** 依赖时间轴上的事件与相位，**无法**从一条平均 PSD 得到；这是观测不用「全平均 PSD」的主因。

### 6.2 示意图

见 `S4_sbi/scan_diagnostics/`：

| 图 | 文件 |
|----|------|
| 0–20 Hz 三曲线 + epoch 灰线 | `fig_concat_vs_mean_psd_overview.png` |
| SO 段放大 + T4 表 | `fig_concat_vs_mean_psd_so_zoom.png` |
| 拼接 vs 平均概念 + r_proxy 片段 | `fig_concat_vs_mean_psd_timeline.png` |

复现：`python S4_sbi/plot_concat_vs_mean_psd.py`

---

## 7. 与 V7 全脚本 `compute_fitness_v7` 的关系

| | V7 DE（Stage 1） | SBI 仿真臂 |
|--|------------------|------------|
| 参数 | 8 维全搜索 | 4 维自由 + 4 维固定 |
| 每次评估 | `build_model` + `m.run()` + **fitness** + 12 约束 | 同仿真核心 + 只输出 **5 维摘要** |
| EEG | 平均 PSD → shape_r 等奖励 | 同 `_target_periodic` |
| 优化 | 最大化 fitness | 学 **后验** $p(\theta\mid x_\text{obs})$ |

仿真子流水线（`build_model` → `r_ctx,r_thal` → `compute_constraints_v7`）与 V7 **同源**；SBI 不跑 DE，不把 12 条约束变成似然，除非将来把约束标量也放进 `x_sim`。

---

## 8. 工程状态（2026-05-23）

| 组件 | 维度 | 状态 |
|------|------|------|
| `x_obs_v3.npz` | 5 | 已生成；5 维 SBI 已跑完（见 `docs/sbi_5dim_review_20260523.md`） |
| `simulator_wrapper` | 5 | 与 v3 对齐 |
| `x_obs_v4.npz` | 7 | 需运行 `compute_xobs_from_eeg_v4.py` |
| `simulator_wrapper` Phase 2 | 7 | 计划加 T6/MI 导出（见 `docs/sprint1_phase2_plan_v2.md`） |
| 7 维旧 run | 7 | 归档 `S4_sbi/sbi_outputs_7dim_archive_20260507/`（r_proxy T6/MI 不可比） |

---

## 9. 推荐阅读顺序

1. 本文（观测 vs 仿真总览）  
2. `S4_sbi/STAGE2_ARCHITECTURE.md`（三文件职责）  
3. `docs/sbi_report_0511.md`（5 维/7 维实验结论与 Sprint 计划）  
4. `docs/sprint1_phase2_plan_v2.md`（v4 EEG-native T6/MI）  
5. HTML 版（同内容 + 内嵌图）：`docs/sbi_obs_vs_sim_comparison.html`

---

*生成说明：对照表依据 `compute_xobs_from_eeg_v3/v4.py`、`simulator_wrapper.py`、`s4_personalize_fig7_v7.py` 及 `plot_concat_vs_mean_psd.py` 在 SC4001 上的运行结果。*
