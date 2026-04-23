# Sleep-Loop 项目进度报告 — 2026-04-22

> 运行环境：`conda activate neurolib`（Python 3.x，NumPy 2.x，需 `fooof`）
> 项目根目录：`D:\Year3_Mao_Projects\sleep_loop\`
> 下文脚本均从项目根目录执行。

本次会话主要做了 4 件事：

1. 为仓库新建 `CLAUDE.md`（给未来 Claude Code 实例用的项目导航）
2. 讲解 `models/s4_personalize_fig7_v4_improve_v2.py` 的 `compute_fitness_v4`
3. 讲解"连续惩罚 [-10, -2]"相比"悬崖式 -1e6"的好处
4. 讲解 `plot_scripts/plot_fig7_v3_fast.py` 的输入 / 过程 / 输出

目标读者：**接触过 Python 编程的高中生**，风格为分步骤 + 比喻。

---

## 1. 新建 `CLAUDE.md`

位置：仓库根目录 `CLAUDE.md`。

写入内容（概要）：

- **项目目的**：用真实 Sleep-EDF N3 脑电反推丘脑-皮层模型的 8 个参数。
- **常用命令**：V3 主脚本、单次适应度测试、快速画图脚本，都必须从仓库根目录执行。
- **数据前置**：`data/manifest.csv` + `data/sleep-edfx-cassette/` 在 `.gitignore` 内，需要本地准备。
- **架构要点**（需要读多文件才能理解的"隐藏知识"）：
  - `s4_personalize_fig7_v3.py` 是当前主线；`v1/v2/v4*` 保留作对比。
  - FOOOF 缺失时 `shape_r` 会静默退化到加权 χ² —— 两条路径结果不等价。
  - `utils/02_preprocess_psd.py` 以数字开头，只能用 `importlib` 加载。
  - `neurolib` 依赖 `np.int` 等旧别名 —— 脚本顶部有 NumPy 补丁，不能删。
  - 模块级全局变量（`_eval_count`、`_best_params`、`_records` 等）在 DE 过程中被 mutate；新驱动脚本需在前后清零。
  - 画图脚本若要重算 `shape_r`，必须完全复现 "Welch → 插值到 FOOOF 频轴 → FOOOF → log 残差相关" 的顺序。
- **输出命名规则**：`data/patient_params_fig7_{version}_{SUBJECT_ID}.json`、`outputs/evolution_fig7_{version}_records.csv`、`outputs/fig7_{version}_*.png`。
- **文档索引**：指向 `docs/` 下 V3 overview、compute_fitness_v3 详解、load_target_psd / compute_target_periodic 笔记等。

---

## 2. `compute_fitness_v4` 讲解（V4-improve-v2）

### 定位

**差分进化（DE）的裁判函数**：拿到一组参数 → 跑仿真 → 返回一个标量分数。DE 想最小化，所以实际 `return -fitness`。

### 输入

| 参数 | 含义 | 每次调用是否变化 |
|---|---|---|
| `params_vec` | 长度 8 的参数向量 `[mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th]` | **变** |
| `target_psd`, `target_freqs` | 真实 N3 EEG 的平均功率谱及对应频率 | 不变 |
| `target_periodic`, `fooof_freqs` | 目标去 1/f 后的振荡形状及 FOOOF 频轴 | 不变 |

### 过程

1. `_eval_count += 1`；解包 8 个参数。
2. `build_model + m.run()`：跑 30 s 丘脑-皮层仿真；若 numba 后端失败，切换 `jitcdde`；再失败返回 `BAD_OBJECTIVE`。
3. 取皮层 / 丘脑发放率（乘 1000 转 Hz），**丢弃前 5 s 热身**。
4. Welch 算仿真 PSD，截取 `0.5–20 Hz`。
5. 调用 `compute_constraints_v4` 做 **8 条硬性检验 T1–T8**：

   | 编号 | 检验内容 |
   |---|---|
   | T1 | 皮层出现过 DOWN（< 1 Hz） |
   | T2 | 皮层出现过 UP（> 15 Hz） |
   | T3 | UP 持续时间 ≥ 阈值（防瞬时尖峰骗 T2） |
   | T4+ | SO 峰在 0.3–1.5 Hz 且 Q-factor 足够尖 |
   | T5 | 丘脑纺锤峰 FWHM 足够宽 |
   | T6 | 慢波间隔 CV 小（节律规整） |
   | T7 | 纺锤包络 CV 大（burst 而非连续振荡） |
   | T8 | 纺锤事件数 ≥ 阈值且时长合理 |

6. 根据可行性分支打分：

   - **8 条全过（feasible）**：`fitness = 0.50·shape_r + 0.25·so_power + 0.25·spindle_power`，范围 `[0, ~1]`。
     - `shape_r` 用 V3 相同流水线（插值 → FOOOF → log 残差 → Pearson r → `max(·, 0)`）。
     - `so_power` / `spindle_power` 来自 FOOOF 峰参数在 SO / 纺锤频段的最大值。
   - **有未通过（infeasible）**：`fitness = -10 + 8 · mean(c_scores)`，范围 `[-10, -2]`。

7. 追加一行到 `_records`（记录 8 个参数、fitness、每条 T 详情）；若刷新最优则更新 `_best_score` / `_best_params`。

### 输出

- **返回值**：`-fitness`（交给 `differential_evolution` 最小化）。
- **副作用**：`_records` / `_best_params` / `_eval_count` 模块级全局更新；最终在 `main()` 里写入 JSON 与 CSV。

---

## 3. 连续惩罚 vs. 悬崖式 -1e6

### 核心好处：**给 DE 一个可爬的坡度**

DE 通过对比个体得分决定变异方向。悬崖式惩罚把所有"不可行"压平成 `-1e6`，DE 无法区分 "差一条 T 就过" 与 "八条全崩"，只能随机乱跳直到偶然 feasible。

### 对比

| 情景 | -1e6 方案 | 连续方案 `[-10, -2]` |
|---|---|---|
| 过 7/8 条 | -1,000,000 | ≈ -2.4 |
| 过 4/8 条 | -1,000,000 | ≈ -6.0 |
| 全崩 | -1,000,000 | ≈ -10 |
| 可行（得分 0.6） | +0.6 | +0.6 |

连续方案：**feasible > infeasible** 依然严格成立（可行最低 0，不可行最高 -2），但 DE 能"嗅到"哪些不可行个体更接近可行区，从而把变异引向正确方向。

### 比喻

- 悬崖式 = 考试不到 60 分全判 0，58 分和 10 分看起来一样糟。
- 连续式 = 不到 60 分按实际分给，老师能看出谁只差一点。

### `-10 + 8 · mean` 映射的目的

- `mean(c_scores)` 范围 `[0, 1]`。
- 乘 8 + 偏移 -10 ⇒ 映射到 `[-10, -2]`。
- **留出 `-2 ~ 0` 缺口**，保证任何 infeasible 分严格低于任何 feasible 分。

类比：RL 里"撞墙只是让你停下，不是杀死你" —— 连续负反馈比一记闷棍更能指导搜索方向。

---

## 4. `plot_fig7_v3_fast.py` 讲解

### 定位

**训练后的"成绩单"脚本**：读取最优参数 JSON → 再跑一次仿真 → 生成 3 张 PNG。**不做优化，仅可视化**。

### 输入

命令行参数（全部可选）：

- `--params-path`：默认 `data/patient_params_fig7_v3_SC4001.json`
- `--sim-dur-ms`：`30000`（与训练对齐）或 `60000`（更长时序观察），默认 30 s
- `--out-timeseries` / `--out-spectra` / `--out-residuals`：3 张图各自的输出路径

硬编码：`SUBJECT_ID=SC4001`，频段 `0.5–20 Hz`，展示窗口 `8–24 s`。

### 过程

1. **切根目录 + NumPy 补丁**（相对路径 & 旧 neurolib 兼容）。
2. **读 JSON**：拿到 8 参数 `bp` 字典，外加训练时记录的 `score`、`shape_r`、`dynamics_score`。
3. **建模 + 跑仿真**：`build_model(bp) → m.run()`；numba 失败 → jitcdde。
4. **取时间序列**：`rE_cortex`、`rE_thalamus`（×1000 转 Hz）。
5. **丢前 5 s 热身 + Welch PSD**（截 0.5–20 Hz）。
6. **加载目标 EEG**：`load_target_psd_v3_aligned` 与 `compute_target_periodic`，流程与 V3 训练完全一致（读 EDF → 切 N3 epoch → 伪迹剔除 → Welch → FOOOF 分离 1/f 与周期）。
7. **复算 `shape_r`**：`recompute_shape_and_peaks_v3_order` 严格复现训练顺序（先插值仿真 PSD 到 FOOOF 频轴，再 FOOOF，再相关），得到 `shape_r_raw` 与 `shape_r_clipped`，以及重算的 `so_power` / `spindle_power`。
8. **绘制 3 张图**：
   - `fig7_v3_timeseries.png`（Fig. 7c 风格）：皮层 / 丘脑发放率时序，1 Hz 与 15 Hz 阈值虚线，标题含 JSON 分数与 8 参数。
   - `fig7_v3_spectra.png`（Fig. 7d 风格）：上子图目标 EEG PSD + 1/f 线；下子图仿真 PSD + 1/f 线；SO / 纺锤频带橙 / 绿底色；右下角列出 JSON 分与重算分。
   - `fig7_v3_residuals.png`：目标与仿真的 FOOOF 去背景残差叠图，标题含 `r_raw` / `r_clipped` / JSON 三值对比。
9. **控制台总结**：打印 JSON 分、重算分、皮层 min/mean 发放率。

### 输出

- `outputs/fig7_v3_timeseries.png`
- `outputs/fig7_v3_spectra.png`
- `outputs/fig7_v3_residuals.png`
- 控制台打印分数对比

### 关键对齐点（易错）

1. **目标 PSD 流水线**必须与训练完全一致。
2. **`shape_r` 计算顺序**必须"先插值再 FOOOF"，任何顺序变化都会导致重算值与 JSON 记录不匹配。

---

## 与既有文档的衔接

- V3 主线 overview：[`s4_personalize_fig7_v3_overview.md`](s4_personalize_fig7_v3_overview.md)
- `compute_fitness_v3` 深读：[`s4_personalize_fig7_v3_compute_fitness_v3.md`](s4_personalize_fig7_v3_compute_fitness_v3.md)
- V3 相关函数笔记：[`load_target_psd_notes.md`](load_target_psd_notes.md)、[`compute_target_periodic_notes.md`](compute_target_periodic_notes.md)、[`compute_fitness_v3_notes.md`](compute_fitness_v3_notes.md)
- V3 → V4 过渡：本次 V4-improve-v2 将 T1–T5 扩展为 T1–T8 硬约束，并把 -1M 悬崖替换为连续惩罚 `[-10, -2]`，配合 ±20% 收紧的 BOUNDS 与 `best1bin + workers=1` 稳定搜索设置。
