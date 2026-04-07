# Sleep-Loop 项目进度报告 — 2026-04-04

> 运行环境：`conda activate neurolib`（Python 3.x，NumPy 2.x，需 `fooof` 用于 Fig.7 谱拟合）  
> 项目根目录：`D:\Year3_Mao_Projects\sleep_loop\`  
> 下文脚本均从项目根目录执行：`python models/<script>.py` 或 `python plot_scripts/<script>.py`

本报告在 [`docs/0315_Progress.md`](0315_Progress.md) 的 Session 1–3 主线（数据管线、s4 个体化、s5 分岔）之上，汇总 **Fig.7 风格丘脑–皮层个体化拟合** 的代码进展：`s4_personalize_fig7.py`（v1）、`s4_personalize_fig7_v2.py`（v2）、以及配套绘图 `plot_fig7.py`、`plot_fig7_residuals.py`。

---

## 脚本总览（Fig.7 相关）

| 脚本 | 功能一句话 | 主要输出 |
|------|-----------|---------|
| `models/s4_personalize_fig7.py` | SC4001 N3 个体目标 PSD + FOOOF 去 1/f；8 维 DE；适应度 **0.5·shape_r + 0.25·SO + 0.25·spindle** | `data/patient_params_fig7_SC4001.json`，`outputs/evolution_fig7_records.csv`，脚本内嵌 Fig.7 预览 |
| `models/s4_personalize_fig7_v2.py` | 同上数据与 FOOOF 思路，**收紧 BOUNDS** + **时域动力学三项约束** + **FOOOF 失败时 1/f 加权 χ² 回退** + 更大 DE | `data/patient_params_fig7_v2_SC4001.json`，`outputs/evolution_fig7_v2_records.csv` |
| `plot_scripts/plot_fig7.py` | 读最优参数，60 s 仿真，生成论文式 **(a)–(d)**：网络示意、皮层/丘脑参数空间与分岔扫描缓存、时序、FOOOF 谱 | `outputs/fig7_personalized.png`，分岔缓存 `.npz` |
| `plot_scripts/plot_fig7_residuals.py` | 目标 EEG 与仿真皮层 PSD 的 **FOOOF 周期残差** 叠图，复算 Pearson *r* | `outputs/fig7_residuals_overlay.png` |

---

## `s4_personalize_fig7.py` — Fig.7 基线（v1）

### 相对 `s4_personalize.py`（见 0315）的要点

- **目标频段**：`F_LO, F_HI = 0.5, 20.0` Hz（与论文 Fig.7 / 仿真采样一致；勿与 s4 常用 0.5–30 Hz 混用）。
- **FOOOF**：对目标 N3 PSD 与仿真皮层发放率 PSD 均在 log 域做非周期拟合，用 **periodic 残差**（`log10(PSD) − aperiodic`）比形状；未安装 `fooof` 时退化为归一化 log 谱。
- **8 个自由参数**：在 s3/s4 基础上增加 **`b`, `tauA`, `c_th2ctx`, `c_ctx2th`**，丘脑–皮层耦合可搜。
- **适应度**：`shape_r` = 目标与仿真的 periodic 残差 Pearson *r*；`so_power` / `spindle_power` 分别为残差在 **0.2–1.5 Hz** 与 **10–14 Hz** 的峰值；经 `(·+5)/7` clip 到 [0,1] 后与 `shape_r` 加权（**0.5 / 0.25 / 0.25**）。
- **优化器**：`scipy.optimize.differential_evolution`（规避 neurolib `Evolution` 与 NumPy 2 的 pypet 问题，与 0315 一致）。
- **规模**：`DE_POPSIZE=15`，`N_GEN=20`，单次评估 `SIM_DUR_MS=30_000` ms；丢弃前 5 s 瞬态后做 Welch。

### 搜索空间 `BOUNDS`（v1，较宽）

| 参数 | 范围 | 备注 |
|------|------|------|
| `mue`, `mui` | [2.5, 4.5] / [2.5, 5.0] | 皮层 E/I 背景 |
| `b`, `tauA` | [0, 50] / [500, 5000] ms | 适应项宽范围 |
| `g_LK`, `g_h` | [0.02, 0.20] | TCR 离子通道 |
| `c_th2ctx`, `c_ctx2th` | [0.001, **0.30**] | 双向耦合均可较大 |

### 已知现象（v1 动机）

仅频域匹配时，优化器易找到 **频谱形状尚可但动力学不像 N3** 的解（例如 E–I 极限环：主峰频率不在慢波带、缺少明显 DOWN、丘脑谱峰过窄等）。这直接推动 v2 的 **物理约束搜索空间** 与 **dynamics_score**。

---

## `s4_personalize_fig7_v2.py` — 物理约束 + 动力学适应度（v2）

### 三处相对 v1 的升级（与脚本头部说明一致）

1. **收紧 `BOUNDS`**：例如 `b∈[10,40]`，`tauA∈[800,2000]`，`g_LK/g_h∈[0.03,0.15]`，`c_th2ctx∈[0.05,0.25]`；关键为 **`c_ctx2th` 上界 0.30 → 0.05**，偏向「皮层主导慢波、丘脑弱耦合」的 N3 共识参数区。
2. **`compute_fitness_v2`**：在保留 FOOOF 形状项与 SO/纺锤残差峰的前提下，增加 **`dynamics_score`** = 三个二值检验的均值（各 0 或 1）  
   - (a) **DOWN**：`min(r_E) < 1` Hz；  
   - (b) **慢波主峰**：Welch 在 0.1–30 Hz 内全局主峰落在 **0.3–1.5 Hz**；  
   - (c) **宽纺锤**：丘脑 PSD 在 8–16 Hz 带内峰之 **半高宽 ≥ 2 Hz**（区分 waxing-waning 与窄带极限环）。  
   **综合得分**：`0.35*shape_r + 0.15*so_norm + 0.15*sp_norm + 0.35*dynamics_score`。
3. **搜索更充分 + 鲁棒性**：`DE_POPSIZE=20`，`N_GEN=30`；若单次仿真侧 FOOOF 拟合失败，使用 **1/f 加权** 的 log 谱 χ² 将误差映射为 `shape_r` 代理，避免整次评估报废。

### 输出与字段

- JSON 除 8 维参数与 `score`、`shape_r`、`so_power`、`spindle_power` 外，增加 **`dynamics_score`**、`version: "v2"` 等。
- 进化记录 CSV 含 `dynamics_score` 列，便于与 v1 对比。

### 与 v1 共用的事实

- 仍用 SC4001 manifest → N3 epoch → 个人平均 PSD；仍用 **皮层 + 丘脑** 双节点 `ThalamoCorticalNetwork`；仍 `workers=1` 避免 numba 多进程问题。

---

## `plot_scripts/plot_fig7.py` — 论文式 Fig.7 成图

- **输入**：默认 `data/patient_params_fig7_SC4001.json`、`outputs/evolution_fig7_records.csv`（用于参数云绿色点筛选等）。
- **流程**：加载最优参数 → 与拟合脚本同构的网络与 `build_model` → **60 s** 仿真（长于拟合评估的 30 s，图更平滑）→ 皮层/丘脑时序、EEG 与仿真的 Welch + FOOOF 1/f 曲线、**皮层 mue–mui** 与 **丘脑 g_LK–g_h** 子图，并对慢振荡 / 纺锤频段做 **分岔扫描**（结果缓存 `.npz`，重复运行秒开）。
- **输出**：`outputs/fig7_personalized.png`（dpi 150）。

**注意**：`s4_personalize_fig7_v2.py` 末尾提示：生成 **v2** 图时需将脚本内 **`PARAMS_PATH` / `RECORDS_PATH`** 改为 `*_fig7_v2_*` 文件；标题中 `shape_r`、SO/spindle 等字段需与 JSON 一致（v2 含 `dynamics_score` 时可酌情改 `suptitle` 展示）。

---

## `plot_scripts/plot_fig7_residuals.py` — 残差叠图与 *r* 校验

- **依赖**：必须安装 `fooof`。
- **输入**：默认 `data/patient_params_fig7_SC4001.json`（与 v1 拟合输出一致）。
- **流程**：复用与 `s4_personalize_fig7` 相同的 N3 目标 PSD 与 FOOOF 目标残差；用最优参数跑 **30 s** 仿真（与进化评估时长一致），对皮层 PSD 做 FOOOF 得 `sim_periodic`；绘制目标 vs 仿真 **log 域周期残差** 叠图，并 **重算 Pearson *r***，可与 JSON 中 `shape_r` 对照。
- **输出**：`outputs/fig7_residuals_overlay.png`。

使用 **v2 参数** 时，应将 `PARAMS_PATH` 指向 `patient_params_fig7_v2_SC4001.json`（或复制脚本改路径）。

---

## 与 `0315_Progress.md` 的衔接

| 0315 内容 | Fig.7 分支的延续 |
|-----------|------------------|
| s4 个体化 PSD（SC4001、归一化与 delta/sigma 诊断） | Fig.7 改为 **FOOOF 残差形状** + 频段峰，并扩展到 8 维与丘脑–皮层耦合 |
| `differential_evolution` 替代 neurolib Evolution | Fig.7 v1/v2 沿用同一选择 |
| 残余问题「谐波峰、适应度可再改进」 | v1→v2 用 **动力学约束 + 收紧边界** 直接回应「频对但动力学错」 |

---

## 数据流（Fig.7 子管线）

```
data/manifest.csv + Sleep-EDF
        │
        ├─► s4_personalize_fig7.py  ──► data/patient_params_fig7_SC4001.json
        │                              outputs/evolution_fig7_records.csv
        │
        ├─► s4_personalize_fig7_v2.py ──► data/patient_params_fig7_v2_SC4001.json
        │                                 outputs/evolution_fig7_v2_records.csv
        │
        ├─► plot_scripts/plot_fig7.py ──► outputs/fig7_personalized.png
        │     (按需改 PARAMS_PATH → v1 或 v2)
        │
        └─► plot_scripts/plot_fig7_residuals.py ──► outputs/fig7_residuals_overlay.png
              (按需改 PARAMS_PATH → v1 或 v2)
```

---

## 待办与使用提示

1. **v2 成图**：运行 `plot_fig7.py` 前确认 `PARAMS_PATH`、`RECORDS_PATH` 与 `outputs/fig7_personalized.png` 是否要与 v2 对齐，避免仍显示 v1 最优解。
2. **残差脚本**：分析 v2 时同步更新 `PARAMS_PATH`，否则叠图仍基于 v1 参数。
3. **计算成本**：v2 单次进化评估次数约为 `DE_POPSIZE × 8 × (N_GEN+1)` 量级，明显高于 v1；适合在参数可行域缩小后换更好解，而非快速试探。

---

## 相关文件路径速查

| 路径 | 说明 |
|------|------|
| `models/s4_personalize_fig7.py` | Fig.7 v1 拟合主脚本 |
| `models/s4_personalize_fig7_v2.py` | Fig.7 v2 拟合主脚本 |
| `plot_scripts/plot_fig7.py` | Fig.7 (a)–(d) 成图 |
| `plot_scripts/plot_fig7_residuals.py` | FOOOF 残差叠图 |
| `data/patient_params_fig7_SC4001.json` | v1 最优参数 |
| `data/patient_params_fig7_v2_SC4001.json` | v2 最优参数 |
| `outputs/evolution_fig7_records.csv` | v1 进化历史 |
| `outputs/evolution_fig7_v2_records.csv` | v2 进化历史 |
| `outputs/fig7_personalized.png` | 论文式总图 |
| `outputs/fig7_residuals_overlay.png` | 残差对比图 |
