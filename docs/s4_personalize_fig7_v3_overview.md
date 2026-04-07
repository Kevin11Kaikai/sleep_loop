# `s4_personalize_fig7_v3.py` 分 Part 导读

面向：**有神经科学/建模背景、信号处理与机器学习公式接触较少**的博士生读者。  
目标：说明本脚本**在干什么**、**数据往哪流**、**v3 相对 v2 改了什么**。

配套细读：**[`s4_personalize_fig7_v3_compute_fitness_v3.md`](s4_personalize_fig7_v3_compute_fitness_v3.md)**（专门讲 `compute_fitness_v3`）。

---

## 脚本在整体上做什么？

1. 从睡眠脑电（Sleep-EDF）里取出 **一名受试者（默认 SC4001）的 N3 慢波睡眠** 段，算出一条 **「目标功率谱」**（真实数据长什么样）。  
2. 用 **丘脑–皮层神经质量模型** 仿真皮层与丘脑的发放率时间序列。  
3. 用 **差分进化** 在 **8 个参数** 的盒子里搜索，使 **仿真既像目标的频谱形状，又满足 N3 常见的时域动力学**（慢波 UP/DOWN、纺锤等）。  
4. 把 **最优参数** 和 **全部尝试记录** 存成 JSON / CSV。

**不**需要事先懂 DE：把它想成 **「带随机性的智能网格搜索」**——每次试一组参数，算一个标量分数，分高的留下。

---

## Part A — 文件头与依赖（约第 1–65 行）

- **文档字符串**：v3 相对 v2 的核心变化——**动力学分数**从 3 项改为 **5 个子检验（T1–T5）**，并增加 **皮层 UP 态** 要求（避免「一直 DOWN」也能骗过高分）。  
- **NumPy 补丁**：部分旧版 **neurolib** 依赖已弃用的 `np.int` 等，在导入 neurolib 前补回别名。  
- **库**：`numpy` / `pandas` 做数组与表；`scipy.signal.welch` 估计功率谱；`interp1d` 在频率轴上插值；`pearsonr` 算形状相关；`differential_evolution` 做优化；`mne` 读 EDF；`neurolib` 跑多节点模型；**`fooof`**（可选）做 1/f 分解。  
- **`HAS_FOOOF`**：未安装时 **`shape_r`** 走简化替代指标，结果与论文式 FOOOF 不完全一致。

---

## Part B — 配置与参数边界（约第 66–94 行）

| 符号 | 通俗含义 |
|------|----------|
| `SUBJECT_ID` | 用哪个受试者的睡眠数据。 |
| `EEG_CHANNELS` | 用哪几个头皮电极（两路平均）。 |
| `F_LO`, `F_HI` | 比较谱时用的频段（0.5–20 Hz）。 |
| `FS_SIM` | 仿真输出采样率（1000 Hz，与 1 ms 步长一致）。 |
| `SIM_DUR_MS` | 每次评估跑 **30 s** 仿真（与 v2 可比）。 |
| `DE_POPSIZE`, `N_GEN` | 进化规模：约 **160 个体/代 × 31** 代量级，总评估次数约几千。 |
| `BOUNDS` | 八个参数各自的 **最小–最大**；**`c_ctx2th` 上界 0.05** 表示 **皮层→丘脑耦合较弱**（偏生理上的 N3 设定）。 |

**`PARAM_NAMES`**：与 `BOUNDS` 顺序一致，用于写 CSV 列名。

---

## Part C — 网络与 `build_model`（约第 96–156 行）

- **`ThalamoCorticalNetwork`**：**两个节点**——皮层（ALN，兴奋/抑制）与丘脑（TCR/TRN）；连接矩阵里可调 **丘脑→皮层**、**皮层→丘脑** 强度。  
- **`set_params_glob`**：neurolib 参数键名很长，用 **通配符** 批量赋值。  
- **`build_model`**：给定 8 个数，建好 **MultiModel**、积分设置、噪声与离子通道参数，返回可 **`run()`** 的模型。

---

## Part D — V3 动力学：`compute_dynamics_score_v3`（约第 159–266 行）

在 **burn-in 之后** 的 **`r_ctx`（皮层 Hz）** 和 **`r_thal`（丘脑 Hz）** 上打 **5 个勾选题**，按权重加分，得到 **`[0, 1]`** 的 **`dynamics_score`**。

| 检验 | 大意 | 权重 |
|------|------|------|
| **T1** | 是否存在 **DOWN**（发放率曾低于约 1 Hz） | 0.20 |
| **T2** | 是否存在 **UP**（峰值超过约 15 Hz） | 0.25 |
| **T3** | **UP 是否持续足够久**（避免单个尖峰骗过 T2） | 0.25 |
| **T4** | 皮层 Welch 谱在慢波带内的 **主峰频率** 是否在约 0.3–1.5 Hz | 0.15 |
| **T5** | 丘脑谱在纺锤带内的峰 **不太窄**（半高宽 > 2 Hz，偏「真实纺锤」而非极限环尖峰） | 0.15 |

**v2 的问题（脚本注释里写的）**：只有 T1 时，模型可以 **几乎一直待在 DOWN**，仍可能在谱上「凑」出看似合理的峰；**T2+T3** 强迫出现 **足够高、足够长的 UP**，更接近文献里慢波交替。

---

## Part E — 目标 EEG：`load_target_psd`（约第 269–307 行）

1. 从 **`manifest.csv`** 找到该受试者 PSG 与睡眠分期文件。  
2. **MNE** 读 EDF，按 **N3（含原 N4 标签）** 切 **30 s epoch**。  
3. **伪迹剔除**：任一路电压峰峰值超过 **200 µV** 的 epoch 丢弃。  
4. 两通道 **平均** → **Welch** → 取 **0.5–20 Hz** → 所有合格 epoch **平均** → **`target_psd`, `target_freqs`**。

这是 **「这名受试者 N3 的平均谱」**，作为拟合目标。

---

## Part F — 目标 FOOOF：`compute_target_periodic`（约第 310–329 行）

- 对 **`target_psd`** 做一次 **FOOOF**：在 log 域拟合 **1/f 背景 + 峰**。  
- 输出：  
  - **`fooof_freqs`**：FOOOF 使用的频率轴（后面仿真侧要对齐这条轴）。  
  - **`target_periodic`**：**log₁₀(功率) − 拟合的 1/f 线**，即 **「去掉背景后的振荡形状」**。  

进化里 **成千上万次** 评估都 **复用** 这两个量，只对 **仿真** 重算。

---

## Part G — 适应度函数与全局变量（约第 332–523 行）

- **`_eval_count`, `_best_score`, `_best_params`, `_records`**：记录评估次数、当前最优、完整历史。  
- **`compute_fitness_v3`**：核心；逐步骤说明见 **[`s4_personalize_fig7_v3_compute_fitness_v3.md`](s4_personalize_fig7_v3_compute_fitness_v3.md)**。  
- 概要输出四个标量再加权：  
  **`0.35×shape_r + 0.15×so_power + 0.15×spindle_power + 0.35×dynamics_score`**。  
- 末尾 **`return -fitness`** 供 DE **最小化**。

---

## Part H — 进化回调 `_callback`（约第 526–546 行）

每 **一代** 结束后打印：当前 **最优总分**、**shape_r**、**dynamics**、**T1–T5 是否通过**、**最大皮层发放率**、累计 **评估次数**、**耗时**。便于在长时间运行中 **肉眼看进度**。

---

## Part I — `main`：串联全流程（约第 549–649 行）

1. 打印配置摘要。  
2. **`load_target_psd`** → **`compute_target_periodic`**。  
3. **`differential_evolution(compute_fitness_v3, bounds=BOUNDS, args=(...))`**：把 **目标数据** 通过 **`args`** 传进每次适应度调用。  
4. 结束後：**最优参数字典** 写入 **`data/patient_params_fig7_v3_SC4001.json`**（路径由 `PARAMS_PATH` 决定）。  
5. **`_records`** 写入 **`outputs/evolution_fig7_v3_records.csv`**。  
6. 控制台打印 **验证摘要**（分数与各子检验）。  
7. 提示用 **绘图脚本** 可视化；若用 v3 结果，需把绘图脚本里的 **`PARAMS_PATH`** 指到 **`_v3`** 文件。

---

## 数据流简图

```
Sleep-EDF (SC4001)
       │
       ▼
load_target_psd ──► target_psd, target_freqs
       │
       ▼
compute_target_periodic ──► target_periodic, fooof_freqs
       │
       │      ┌──────────────────────────────────────┐
       │      │  DE loop: many parameter vectors      │
       │      │    compute_fitness_v3(params, ...)      │
       │      │      → simulate → Welch → FOOOF       │
       │      │      → shape_r, SO/spindle, dynamics  │
       │      │      → return -fitness                 │
       │      └──────────────────────────────────────┘
       ▼
Best JSON + CSV records
```

---

## v2 与 v3 一眼区别

| 方面 | v2 | v3 |
|------|----|----|
| 动力学 | 3 个二值检验 | **5 个**（含 UP 存在与持续） |
| 输出文件名 | `*_fig7_v2_*` | `*_fig7_v3_*` |
| **`shape_r` / FOOOF 主路径** | 与 v3 **相同**（先插值功率到 `fooof_freqs` 再拟合） | 同左 |

---

## 延伸阅读（仓库内）

- **`plot_fig7_v2_fast.py`**：快速出图；若重算 **`shape_r`** 与 JSON 对照，须与 **`compute_fitness_v3` 同一套 PSD+FOOOF 顺序**（见 compute_fitness 专文末节）。  
- **`docs/0315_Progress.md`**、**`docs/0404_Progress.md`**：项目更早的进度与 Fig.7 相关脚本列表。

---

*文档随 `models/s4_personalize_fig7_v3.py` 结构整理；若源码变更，请以实际代码为准。*
