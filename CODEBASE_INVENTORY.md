# CODEBASE_INVENTORY — sleep_loop 仓库事实清单

> 只读盘点，不含论文框架/贡献/标题/摘要建议。所有"已存在产物"均按磁盘实测列出（路径、大小来自实际文件系统扫描）。
> 标注约定：**(a)** = 磁盘上已存在的保存产物；**(b)** = 代码可生成但**未在磁盘找到对应输出**；`需确认` = 无法仅凭文件名/头部注释完全确证其内容。
> 生成日期：2026-06-20。

---

## 1. 仓库结构

### 1.1 顶层目录

| 目录 | 作用（依据代码/文档） |
|------|------|
| `models/` | 各阶段主脚本 s1–s8；s4 系列为个体化拟合主线 |
| `S4_sbi/` | Stage 2 SBI（SNPE-C）管线 + 观测特征提取 + 诊断绘图；含 `legacy/`、`sbi_outputs/`、`sbi_outputs_7dim_archive_20260507/`、`scan_diagnostics/` |
| `S4_v7_repair/` | V7 PAC 相位 bug 修复、cycle-by-cycle PAC、warm-start DE、Pareto 种子选取 |
| `utils/` | PSD 预处理（`02_preprocess_psd.py`，按路径导入）、FOOOF 分析 |
| `plot_scripts/` | Fig.7 系列绘图、V7/V8 对比、Seed A/B 对比 |
| `tests/` | 手写检查脚本（非 pytest）：fitness 单次评估、种子复现性、spindle |
| `valid_scripts/` | PAC 修复的 9 项合成信号验证、surrogate z-score、T6/MI EEG-native 验证 |
| `data/` | 参数 JSON、manifest、PSD/epochs 中间数据（Sleep-EDF 原始数据 gitignored） |
| `outputs/` | 进化 CSV、诊断 PNG、日志、缓存（整体 gitignored） |
| `docs/` | 过程文档、分函数说明、阶段报告（md + pdf + html） |
| `warm_start_de/`, `reevaluate_v7/`, `results/` | warm-start 产物、V7 重评估结果、PSD 结果缓存 |
| `sbi-logs/` | sbi 框架自动写的 TensorBoard event 日志（NPE_C/时间戳） |

### 1.2 主脚本（精确文件指向）

| 组件 | 文件 | 关键入口 |
|------|------|----------|
| Stage 1 个体化主线（**当前 mainline**） | `models/s4_personalize_fig7_v7.py` | `compute_constraints_v7` (≈398–688)、`compute_feasibility_score` (≈691–790)、`compute_fitness_v7` (≈868–1038)、`build_model`、`load_target_psd`、`compute_target_periodic`、`seed_numba` |
| Stage 1 基线（保留） | `models/s4_personalize_fig7_v3.py` | `compute_fitness_v3` |
| 历史版本 | `models/s4_personalize_fig7_{v2,v4,v4_improve,v4_improve_v2,v5,v6}.py` | — |
| PAC 修复实现 | `S4_v7_repair/compute_pac_metrics_fixed.py`（镜像于 `valid_scripts/`） | `up_down_ratio`（≈246）、返回 `mi/preferred_phase/phase_argmax/phase_concentration/up_down_ratio/...` |
| EEG-native PAC | `S4_v7_repair/compute_pac_metrics_eeg_native.py` | `compute_mi_eeg_native`（v4 x_obs 使用） |

---

## 2. 数据处理管线（文件/函数指向）

| 阶段 | 文件 | 说明 |
|------|------|------|
| 假设/PSD 预处理 | `utils/02_preprocess_psd.py` | `load_hypnogram`、`compute_epoch_psd`、`EPOCH_LEN_S`（按路径 `importlib` 导入，不能作模块名导入） |
| FOOOF 分析 | `utils/03_fooof_analysis.py` | 周期成分分离 |
| 目标 PSD 载入 | `s4_personalize_fig7_v7.py: load_target_psd` / `compute_target_periodic` | 从 manifest 指向的 Sleep-EDF 提取 N3，Welch→FOOOF |
| Stage 2 观测特征提取（EEG 侧） | `S4_sbi/compute_xobs_from_eeg_v3.py`（5 维，**基线在用**）；`compute_xobs_from_eeg_v4.py`（7 维，重引入 T6_ibi_cv+MI） | r_proxy: detrend→abs→50ms 平滑→rescale；spindle/PAC 用 eeg_raw |
| Stage 2 仿真 wrapper | `S4_sbi/simulator_wrapper.py` | `simulator(theta_4d)→np.ndarray`；4 自由参 + Seed B 固定参；调用 `v7.compute_constraints_v7` |

**数据前置依赖（无则主线无法运行）**：`data/manifest.csv`（已存在，15030 B）、`data/sleep-edfx-cassette/`（gitignored）、目标 `SUBJECT_ID=SC4001`。

---

## 3. 模型 / 算法组件

| 组件 | 文件/函数 |
|------|-----------|
| 丘脑-皮层均场模型 | `s4_personalize_fig7_v7.py: build_model`（neurolib MultiModel：ALNNode + ThalamicNode） |
| 12 约束 T1–T12 | `s4_personalize_fig7_v7.py: compute_constraints_v7` |
| 连续可行性松弛 | `s4_personalize_fig7_v7.py: compute_feasibility_score`（仅 `compute_fitness_v7` 不可行分支调用，≈999） |
| 适应度 | `compute_fitness_v7`：可行=0.50·shape_r+0.25·so+0.25·spindle；不可行=-10+10·Σfeas/12 |
| Stage 1 优化器 | `scipy.optimize.differential_evolution`（8 参） |
| warm-start DE | `S4_v7_repair/warm_start_de_with_fixed_pac.py`（固定 PAC，以 V7 top-K 为种子） |
| Stage 2 推断 | `S4_sbi/run_sbi.py`：SNPE-C，sbi NSF，4 轮≈5000 sim，逐轮 checkpoint；SBC/PPC/marginals/pairplot/Pareto overlay |
| Numba 确定性播种 | `seed_numba(seed)` |

**Stage 2 自由参数**：`[g_h, g_LK, c_ctx2th, b]`（4 维）。
**固定参数（Seed B，来自 `S4_v7_repair/pareto_seeds_fresh_DE.json`）**：mue=3.3406859…, mui=3.2758268…, tauA=1257.409…, c_th2ctx=0.0329531…。
**基线 SUMMARY_KEYS（5 维）**：`shape_r, T4_q, T4_freq, T8_n_sp_events, T11_lag_ms`（`simulator_wrapper.py:88–91`）。

---

## 4. 评估 / 诊断脚本

| 脚本 | 作用 |
|------|------|
| `tests/test_compute_fitness_v3.py` | 单次 fitness 评估 + 可选出图 |
| `tests/test_seed_reproducibility.py` | 种子复现性 + T1–T12 约束诊断 |
| `tests/test_spindles.py` | spindle 检测检查 |
| `valid_scripts/validate_compute_pac_metrics_fixed.py` | 9 项合成信号 PAC 验证 |
| `valid_scripts/validate_t6_mi_eeg_native.py`、`test_surrogate_zscore_pac.py` | T6/MI EEG-native、surrogate 验证 |
| `S4_v7_repair/diagnose_v7_phase.py` | V7 相位 bug 4 层交叉验证 |
| `S4_v7_repair/{verify_pareto_seeds,Pareto_Analysis,Sobol10_feasibility_check,diagnose_t6_ibi_cv}.py` | Pareto 验证 / 可行性扫描 |
| `S4_sbi/replot_sbc_5dim.py` | 重生成 fig_sbc.png（修 `sbc_rank_plot` 签名，不重训） |
| `S4_sbi/plot_concat_vs_mean_psd.py` | concat vs per-epoch PSD 对比图 |
| `plot_scripts/plot_fig7_compare_v7_vs_v8.py`、`plot_seed_a_vs_b.py` 等 | 对比绘图 |

---

## 5. 已产出实验资产（重点）—— (a) 磁盘已存在的保存产物

### 5.1 Stage 2 SBI — 5 维基线（被接受的主结果）`S4_sbi/sbi_outputs/`

| 文件 | 大小 | 内容 | 产出脚本 |
|------|------|------|----------|
| `round1–4_posterior.pkl` | 各 337 KB | 4 轮 SNPE-C 后验（pickle） | `run_sbi.py` |
| `all_simulations.npz` | 181 KB | 全部 (theta, x) 仿真对 | `run_sbi.py` |
| `fig_marginals.png` | 68 KB | 4 参数后验边缘分布 | `run_sbi.py` |
| `fig_pairplot.png` | 87 KB | 后验两两联合分布 | `run_sbi.py` |
| `fig_ppc.png` | 89 KB | 5 维 PPC | `run_sbi.py` |
| `fig_pareto_overlay.png` | 240 KB | Pareto 种子在后验上的叠加 | `run_sbi.py` |
| `fig_sbc.png` | 36 KB | SBC 秩直方图 | `replot_sbc_5dim.py`（注：主 run 当时绘图失败，此图为事后补出 — `需确认`其与日志 KS 数值一致） |
| `S4_sbi/sbi_results.md` | 1112 B | x_obs 值、MAP+CI、PPC 百分位、墙钟 | `run_sbi.py` |
| `S4_sbi/sbi_log.txt` | 8581 B | 主 run 完整日志 | `run_sbi.py` |
| `S4_sbi/x_obs_v3.npz` | 4288 B | 5 维观测向量（基线输入） | `compute_xobs_from_eeg_v3.py` |
| `sbi_5dim_run.log` / `sbi_5dim_run.err` | 20931 / 4386 B | 5 维 run stdout/stderr | run 重定向 |

**基线关键数字**（见 `docs/sbi_5dim_review_20260523.md`，2026-05-22→23，8.06 h，未早停）：MAP g_h=0.07641 / g_LK=0.04914 / c_ctx2th=0.20636 / b=50.52520；SBC 4/4 PASS；PPC 3/5 PASS（shape_r、T4_q FAIL）；Pareto log_prob B>A>C。

### 5.2 Stage 2 SBI — 7 维归档 `S4_sbi/sbi_outputs_7dim_archive_20260507/`

| 文件 | 大小 | 说明 |
|------|------|------|
| `round1–3_posterior.pkl` | 各 344 KB | 仅 3 轮（早停，跳 R4） |
| `all_simulations.npz` | 178 KB | 7 维 run 仿真对 |
| `fig_{marginals,pairplot,ppc,pareto_overlay,sbc}.png` | 38–303 KB | 7 维 run 全套图（含 fig_sbc.png 37 KB） |

注：7 维 run（May 7）= 9.66 h，2/5 PASS（保留维度）。SBC 数值当时来自 `legacy/run_sbc_standalone.py` 单独运行。

### 5.3 Stage 2 诊断图 `S4_sbi/scan_diagnostics/`

| 文件 | 大小 | 内容 | 产出脚本 |
|------|------|------|----------|
| `fig_concat_vs_mean_psd_overview.png` | 130 KB | concat vs mean PSD 0–20 Hz | `plot_concat_vs_mean_psd.py` |
| `fig_concat_vs_mean_psd_so_zoom.png` | 92 KB | SO 带放大 + T4 指标表 | 同上 |
| `fig_concat_vs_mean_psd_timeline.png` | 152 KB | 示意 + r_proxy 短迹 | 同上 |
| `fig_mi_prominence_scan.png` | 157 KB | MI prominence 扫描 | `需确认`（疑 `legacy/scan_xobs_params.py`） |
| `fig_t6_threshold_scan.png` | 55 KB | T6 阈值扫描 | `需确认` |

### 5.4 Stage 1 DE / V7 修复产物

| 文件 | 大小 | 内容 | 产出脚本 |
|------|------|------|----------|
| `S4_v7_repair/pareto_seeds_fresh_DE.json` | 2383 B | 3 个 Pareto 种子 A/B/C（参数+目标+PAC） | warm-start DE |
| `S4_v7_repair/pareto_seeds.json` | 3942 B | 早期 Pareto 种子集 | `需确认` |
| `S4_v7_repair/seeds_verification_metrics.csv` | 529 B | 种子验证指标 | `verify_pareto_seeds.py` |
| `S4_v7_repair/fig_pareto_fresh_DE.png` | 266 KB | fresh DE Pareto 前沿 | `Pareto_Analysis.py`（`需确认`） |
| `S4_v7_repair/fig_pareto_2d.png` | 306 KB | 2D Pareto | `需确认` |
| `S4_v7_repair/fig_pareto_seeds_verification.png` | 855 KB | 种子验证图 | `verify_pareto_seeds.py`（`需确认`） |
| `warm_start_de/patient_params_warm_start.json` | 591 B | warm-start 最优参数 | `warm_start_de_with_fixed_pac.py` |
| `warm_start_de/warm_start_records_OLD.csv` | 921 KB | warm-start 进化记录（OLD） | 同上 |
| `reevaluate_v7/reevaluate_v7_results.csv` | 21471 B | V7 用修复 PAC 重评估结果 | `reevaluate_v7_with_fixed_pac.py` |
| `reevaluate_v7/reevaluate_v7_summary.txt` | 2926 B | 重评估摘要 | 同上 |

### 5.5 Stage 1 进化记录（CSV）`outputs/`

| 文件 | 大小 | 版本 |
|------|------|------|
| `evolution_fig7_v7_records.csv` | 1466 KB | V7（含 `_OLD` 1468 KB） |
| `evolution_fig7_v6_records.csv` | 1450 KB | V6 |
| `evolution_fig7_v5_records.csv` | 1400 KB | V5 |
| `evolution_fig7_v4_records.csv` / `v4b_records_0418` | 1160 / 1255 KB | V4 / V4b |
| `evolution_fig7_v3_records.csv`（+0411/0412） | ~1.17 MB ×3 | V3 |
| `evolution_fig7_records.csv` | 504 KB | 初版 |

### 5.6 Stage 1 拟合最优参数（JSON）`data/`

`patient_params_fig7_v7_SC4001.json`（921 B，+ `_OLD`）、`…_v6/v5/v4/v4b/v3/v2/v1…_SC4001.json`、`patient_params_fig7_SC4001.json`、`patient_params{,_SC4001}.json`。（v4b 有数个"副本"冗余文件。）

### 5.7 Fig.7 系列绘图 `outputs/`（spectra / residuals / timeseries 三件套）

各版本均存在 PNG：`fig7_v0…v8_{spectra,residuals,timeseries}*.png`（含 0411/0412/0417/0418/0419 日期变体）；`fig7_v3_test_*`（test 脚本输出）；`fig7_personalized_0417_v3.png`。

### 5.8 PAC / 对比 / 诊断图与中间数据 `outputs/`

| 文件 | 大小 | 内容 |
|------|------|------|
| `fig7_pac_v7_v8_compare*.png`、`fig7_pac_v7_v9_compare_*.png` | 189–552 KB | V7 vs V8/V9 PAC + 时序对比 |
| `fig_seed_a_vs_b_1_timeseries.png` / `_2_pac.png` | 580 / 198 KB | Seed A vs B（`plot_seed_a_vs_b.py`） |
| `v7_phase_diagnosis_layer1/3/4.png` + `…_signals.npz`（2.85 MB）+ `…_summary.txt` | — | V7 相位 bug 诊断（`diagnose_v7_phase.py`） |
| `thalamus_sweep_3d.npz`（57 KB）+ `thalamus_sweep_heatmaps.png` + `data/thalamus_sweep_3d.csv`（228 KB） | — | 丘脑 3D 扫描 |
| `c_th2ctx_scan.csv`/`_heatmap.png`/`_summary.txt`、`v6_hotspot_diagnosis.csv`/`.txt`、`golden_point_*.png/.txt`、`psd_validation.png`、`fresh_de_30gen_log.txt` | — | 各类扫描/诊断 |

### 5.9 PAC 修复验证 `valid_scripts/validation_outputs/`

`V1`–`V8` 共 8 张 PNG（up-locked、波形不变性、无耦合基线、三 regime、噪声鲁棒、新旧对比、bin 均匀性、双峰检测）+ `V9_edge_cases.txt` + `SUMMARY.txt`。产出脚本 `validate_compute_pac_metrics_fixed.py`。

### 5.10 早期阶段数据 `data/` & `results/`

`cmat.npy`/`dmat.npy`（各 51 KB，连接矩阵）、`epochs_{n1,n2,n3,rem,wake}.csv`、`psd_{n1,n2,n3,rem,wake}.npy`、`target_psd{,_SC4001}.npy`、`target_freqs{,_SC4001}.npy`、`bifurcation_summary.json`、`results/psd/`。

### 5.11 文档与报告 `docs/`（已存在文件）

- **Stage 2 报告**：`sbi_5dim_review_20260523.md`、`sbi_5dim_diagnostic_20260523.md`、`sbi_5dim_preflight_report.md`、`sbi_report_0511.{md,pdf}`、`Review_SBI_522.pdf`、`sbi_obs_vs_sim_comparison.{md,html}`（未提交）。
- **汇总/看板**：`stage2_master_report.md`、`stage2_master_dashboard.html`（57 KB）。
- **方法/一致性**：`Stat_Computation_Consistency_Check.md`、`PAC0502.{md,pdf}`、`T7_Spindle_Envelope_Burstiness_0501.md`、`c_ctx2th_unidentifiability_evidence.md`、`Why_fresh_DE_more_trustworthy.pdf`。
- **计划/清单**：`sprint1_phase2_plan{,_v2}.md`、`sprint2_startup_checklist.md`、`preflight_steps234_report.md`。
- **过程日志**：`0315/0316/0404_Progress.md`、`Progress_0422.md`、`Progress_0506.pdf`、`数字孪生策略报告.pdf`。
- **分函数说明**：`s4_personalize_fig7_v3_overview.md` 等。

### 5.12 sbi 框架自动日志 `sbi-logs/NPE_C/`

8 个时间戳目录的 TensorBoard `events.out.tfevents.*`（May 6 ×6、May 22 ×2）。训练损失曲线原始数据，需 TensorBoard 读取。

---

## 6. (b) 代码可产出但磁盘上未找到保存输出

| 预期产物 | 应由谁产出 | 现状 |
|----------|-----------|------|
| **7 维 x_obs（`x_obs_v4.npz`）** | `compute_xobs_from_eeg_v4.py`（默认输出 `S4_sbi/x_obs.npz` 或 `--output`） | **未在磁盘找到** `x_obs_v4.npz`；现存仅 `x_obs_v3.npz`（5 维）与 `legacy/x_obs.npz`（旧）。v4 脚本存在但其 7 维输出未保存 |
| **基于 v4（7 维含 T6_ibi_cv+MI 重引入）的完整 SBI run** | `run_sbi.py` 配 v4 x_obs | 未找到对应 sbi_outputs；现存只有 5 维（v3）与 7 维归档（May 7，特征集与 v4 不同，`需确认`） |
| `compute_xobs_from_eeg_v4.py` 的 stdout 报告 | 脚本 print | 无保存日志文件 |
| concat-vs-mean PSD 之外的 MI/T6 扫描脚本来源 | `legacy/scan_xobs_params.py` 等 | 图存在（5.3）但产出脚本对应关系 `需确认` |
| 当前 V7 主 run（60s sim×160×30gen）的"本次"日志 | `s4_personalize_fig7_v7.py` | CSV 存在；独立运行日志仅见 `fresh_de_30gen_log.txt`，是否覆盖 V7 主 run `需确认` |

---

## 7. 缺失资产清单（论文常需、但当前磁盘上没有保存的）

> 仅陈述"未保存"的事实，不评估必要性。

1. **统一的复现实验环境锁定**：仅 `requirements.txt`（102 B，无版本钉死）；无 `environment.yml`、无 conda lock。FOOOF 是否安装会改变 `shape_r` 路径，但无运行时记录标明本批结果用了哪条路径。
2. **观测向量对比表的数值源文件**：`sbi_obs_vs_sim_comparison.{md,html}` 已存在，但其依赖的逐维 obs/sim 数值是否有独立 CSV/NPZ `需确认`（未单独找到）。
3. **SBC 原始秩数组**：`fig_sbc.png` 存在，但秩（ranks）数组 / KS 统计量的机器可读文件（CSV/NPZ）未单独保存；KS 数值仅散落在 `sbi_log.txt` 与 review md。
4. **PPC 原始分布数据**：`fig_ppc.png` 存在，但每维预测分布与 x_obs 百分位的机器可读数组未单独保存（仅图 + md 表）。
5. **后验样本表**：`*_posterior.pkl` 为 sbi 对象；无导出的纯样本数组（CSV/NPZ）或 MAP/CI 的机器可读 JSON（CI 仅在 md 表中）。
6. **5 维 vs 7 维对比的脚本化产物**：对比结论写在 `sbi_5dim_review_20260523.md §6`，但对比图/表的独立生成脚本与数据文件 `需确认`。
7. **种子→后验 log_prob 的数据文件**：log_prob 数值仅在 review md；无独立 CSV。
8. **统计显著性 / 不确定度区间的数值表**：分散在多份 md/pdf，无集中数据文件。
9. **LICENSE**：仓库无许可证文件（README 已注明）。
10. **数据可获得性说明的可机读 manifest schema**：`manifest.csv` 含本机绝对路径（README 已警示），无脱敏/schema 版本。

---

## 8. 不确定项汇总（`需确认`）

- `fig_sbc.png`（5 维）由 `replot_sbc_5dim.py` 事后补出，与主 run 日志 KS 是否完全一致未核对。
- `fig_mi_prominence_scan.png`、`fig_t6_threshold_scan.png`、`fig_pareto_2d.png`、`fig_pareto_seeds_verification.png`、`pareto_seeds.json` 的确切产出脚本仅按命名推断。
- 7 维归档（`sbi_outputs_7dim_archive_20260507`）的特征集与 `compute_xobs_from_eeg_v4.py` 设计的 7 维是否同一套，未逐字段核对。
- `fresh_de_30gen_log.txt` 是否对应当前 `evolution_fig7_v7_records.csv` 未核对。
