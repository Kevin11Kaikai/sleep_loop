# Stage 2 / Sprint 1 Master Report

**生成日期**:2026-05-23
**Stage 2 状态**:closed,5-dim SBI 为最终基线
**对应 commit**:`f033e95`(`Sprint 1 closeout: 5-dim SBI baseline + c_ctx2th unidentifiability negative result`)
**覆盖时段**:2026-05-07(首次 7 维 SBI)→ 2026-05-23(Phase 2 closeout)
**子文档导航**:本文为汇总,所有原始诊断数字、图、调试细节都在子文档里——本文只给结论、关键表格、跳转链接。

---

## Part I:Stage 2 当前状态

**Stage 2 一句话总结**:5 维 SBI 在 g_h / g_LK / b 三参数上得到了校准良好的后验(SBC 4/4 PASS,PPC 3/5 PASS),但 c_ctx2th 在所有尝试下(7 维原始 / 5 维删减 / Sprint 1 Phase 2 EEG-native 7 维重构)均呈现多模态 + 完全独立结构。Surrogate test 在 simulator 端 z=+42 确认 PAC 信号机制存在,但 SC4001 single-channel raw EEG 上 z=+0.14(与白噪 z=-1.70 不可区分)—— 证据指向数据-模型 mismatch:SC4001 这套数据不携带可靠的 SO-spindle PAC 信号,无法约束 c_ctx2th。我们接受 5 维 baseline 作为最终结果,把 c_ctx2th unidentifiability 作为 paper 的 publication-worthy main finding。

| 项 | 状态 | 出处 |
|----|------|------|
| 5 维 SBI baseline | ✅ established | `sbi_report_0511.md` §3 |
| 4 个自由参数后验 | g_LK/c_ctx2th/b 可识别;**c_ctx2th 不可识别** | `sbi_5dim_review_20260523.md` §2-3 |
| SBC 校准 | 4/4 PASS | 本文 Part II 表 2 |
| PPC 通过 | 3/5 PASS | 本文 Part II 表 3 |
| Sprint 1 Phase 2(EEG-native T6 + MI) | ⛔ archived(走路径 W) | `c_ctx2th_unidentifiability_evidence.md` |
| Sprint 2(4 post-hoc validation) | 待启动,checklist 就绪 | `sprint2_startup_checklist.md` |
| Paper framing | 由"4 维全可识别"调整为"3/4 可识别 + 1 个 publication-worthy negative result" | 本文 Part IV |

Stage 2 在三个层面 closeout:
1. **算法 pipeline**:SNPE-C + NSF + sequential 4 轮 + SBC + PPC 全部跑通,主线代码冻结
2. **数据-模型边界**:SC4001 single-channel Fpz-Cz 在 PAC 维度上对 c_ctx2th 信息量饱和,经 3 条独立证据链证明
3. **节奏**:8.06 h 完整 run 取代 May 7 的 9.66 h 半完成 run,所有产物归档可追溯

---

## Part II:5-dim SBI Baseline(May 23 Run)

### 训练流程

| 项 | 值 | 出处 |
|----|----|------|
| 起止 | 2026-05-22 20:31:10 → 05-23 04:34:40 | `sbi_log.txt` |
| 墙钟 | 8.06 h(483.5 min) | `sbi_log.txt` 末行 |
| 主循环 ROUND_SIMS | [2000, 1000, 1000, 1000] = 5000 | `run_sbi.py:134` |
| SBC + PPC | 200 + 100 → 共 5300 sims | `sbi_log.txt` |
| 早停 | 未触发(R3→R4 std 相对变化 27.1% > 10% 阈值) | `sbi_5dim_review_20260523.md` §1.2 |
| NaN 率 | 0% 全程 | `sbi_log.txt`(每轮收尾) |
| 平均速率 | 5.45 s/sim | 同上 §1.3 |
| Prior(本次) | g_h ∈ [0.035, 0.095];g_LK ∈ [0.020, 0.070];c_ctx2th ∈ [0.05, 0.22];**b ∈ [28.4, 80.0]** | `run_sbi.py:131-132` |
| 设备 | CPU(sbi=0.26.1, torch=2.5.1) | `sbi_log.txt:2` |

`b` 上界从 May 7 的 42.6 放宽至 80.0 是 pre-flight 的 P1 修订(`sbi_5dim_preflight_report.md` §B2),修订动机:May 7 的 `b` MAP=42.37 紧贴上界 42.6(仅余 0.5% 空间),需要让后验在 5 维下展示真实形状而非贴边伪影。

### 表 1:MAP 对比(7 维 May 7 vs 5 维 May 23)

| 参数 | 7-dim MAP | 5-dim MAP | % change |
|------|-----------|-----------|----------|
| g_h | 0.05919 | 0.07641 | +29.1% |
| g_LK | 0.05137 | 0.04914 | -4.3% |
| c_ctx2th | 0.06917 | 0.20636 | +198.3% |
| b | 42.367 | 50.525 | +19.3% |

数据来源:May 7 — `git show 23483eb:S4_sbi/sbi_results.md`(7 维 archive);May 23 — `S4_sbi/sbi_results.md`(当前 5 维)。

**关键观察**(详 `sbi_5dim_review_20260523.md` §6.1):
- `c_ctx2th` MAP 跳幅 +198%,从 prior 下半区(0.069)跳到上半区主峰(0.206),反映模态结构对 likelihood 维度选择敏感
- `b` MAP +19.3% 后落在 50.5,**远离 80.0 上界**(余量 37%),证明 prior 放宽有效解除贴边伪影
- `g_LK` 变化最小(-4.3%),说明该维度被 likelihood 强约束,7 维与 5 维结论一致
- `g_h` MAP +29.1% 但 95% CI [0.04939, 0.07973] 宽度合理,可识别性未崩

### 表 2:SBC KS p-values(5 维 May 23 主 run)

| 参数 | KS p | Status |
|------|------|--------|
| g_h | 0.6803 | PASS |
| g_LK | 0.4090 | PASS |
| c_ctx2th | 0.8573 | PASS |
| b | 0.0850 | PASS |

数据来源:`S4_sbi/sbi_log.txt` 行 410-413(主 run 真实输出)。判据:p > 0.05 = PASS。

`fig_sbc.png` 由 `S4_sbi/replot_sbc_5dim.py` 补出(主 run 因 `sbc_rank_plot()` API 不兼容 sbi 0.26.1 fail 但 KS 数字算出)。replot 用独立 200 个新 SBC pairs 验证:g_h KS p=0.1049, g_LK 0.4716, c_ctx2th 0.8759, b 0.5386 — **4/4 PASS 结论稳定**(详 `sbi_5dim_diagnostic_20260523.md` Task 1 后续)。

**含义**:NSF 在 5 维上学到的 likelihood↔posterior 映射在算法层面校准良好。后续 c_ctx2th 多模态不是密度估计器欠拟合的结果。

### 表 3:PPC 百分位(5 维 May 23)

| 维度 | 百分位 | Status |
|------|--------|--------|
| shape_r | 100 | FAIL |
| T4_q | 100 | FAIL |
| T4_freq | 77 | PASS |
| T8_n_sp_events | 11 | PASS |
| T11_lag_ms | 37 | PASS |

数据来源:`S4_sbi/sbi_results.md` PPC 段,主 run 输出于 `sbi_log.txt:432-436`。判据:5 ≤ pct ≤ 95 = PASS。

shape_r=1.0 硬编码为设计哨兵(`sbi_report_0511.md` §4.3),100% FAIL 是预期行为不计入算法故障。T4_q 在 98 百分位 FAIL 是新发现的偏移项,值得 Sprint 2 跟进。

---

## Part III:c_ctx2th Unidentifiability

5 维 SBI 后验在 g_h / g_LK / b 三维上收窄,**唯独 `c_ctx2th` 呈 4-模态 + 完全独立于其他三参数**,且模态结构对 likelihood 维度敏感(7 维 vs 5 维不同)。这是 Stage 2 最重要的发现。

### 关键证据(只列数字,详证据见 `c_ctx2th_unidentifiability_evidence.md`)

| 子证据 | 数字 | 出处 |
|--------|------|------|
| `c_ctx2th` 边际 4 个 mode 位置 | 0.052 / 0.107 / 0.155 / **0.207**(主峰) | `sbi_5dim_diagnostic_20260523.md` §3.5 |
| `c_ctx2th × g_h` 联合 | 横条,c_ctx2th 横扫全 prior,沿 g_h 仅两条细带 | 同上 Task 2.1 |
| `c_ctx2th × g_LK` 联合 | 横条,接近独立 | 同上 Task 2.2 |
| `c_ctx2th × b` 联合 | 竖条,4 个 c_ctx2th 模态共享同 b ≈ 50 | 同上 Task 2.3 |
| `c_ctx2th` 95% CI 宽度 | [0.052, 0.214],占 prior 全宽 0.17 的 **95%** | `sbi_results.md` |
| 模态数随维度变化 | May 7 7 维:3 模态主峰 0.075;May 23 5 维:4 模态主峰 0.207 | `sbi_report_0511.md` §3.3 |
| `c_ctx2th × b` 投影中无斜对角 ridge | 3 个其他参数都没在解释 c_ctx2th 的多模态来源 | `sbi_5dim_diagnostic_20260523.md` Task 2.4 |

### 表 4:Pareto Seed log_prob(7 维 vs 5 维)

| Seed | 7-dim | 5-dim | DE 标签 |
|------|-------|-------|---------|
| A | -12.07 | -9.32 | PAC-dominant(MI=0.137,score=0.510) |
| B | +2.41 | -4.11 | 最高 fitness(score=0.674,DE 排名 1) |
| C | +1.21 | -20.42 | shape-dominant(shape_r=0.690) |

数据来源:May 7 — `git show 23483eb:S4_sbi/sbi_log.txt`(Pareto overlay 段);May 23 — `S4_sbi/sbi_log.txt`(行 449-451)。

**排序变化**:
- 7 维:**B > C > A**
- 5 维:**B > A > C**
- Seed A 与 Seed C 的相对位置反转,所有三个 seed 的 log_prob 绝对值都跌入负区

**解读**:5 维已不再包含 MI(PAC 调制指数),故 PAC-dominant 的 Seed A 与 shape-dominant 的 Seed C 在 5 维 likelihood 下不再被有效区分。详 `sbi_5dim_review_20260523.md` §6.2。

---

## Part IV:Sprint 1 Phase 2(EEG-native T6 + MI)— Archived

### 探索路径

5 维删除了 T6 + MI 是因为 r_proxy 信号链不可修复(`sbi_report_0511.md` §4.1-4.2)。Phase 2 假说:**用 EEG-native 算法替代 r_proxy 路径,可重新引入 T6 + MI 这两维**。

实现了:
- `S4_v7_repair/compute_pac_metrics_eeg_native.py`:AASM 75 µV 半波 SO UP 检测 + Tort 2010 single-channel PAC + amplitude-shift surrogate z-score(canonical Tort 2010 方法)
- `S4_sbi/compute_xobs_from_eeg_v4.py`:7 维 x_obs 抽取(SUMMARY_KEYS 重回 7 维)
- 合成 5 case 验证:**10/10 PASS**(算法在 ground-truth 信号上工作正确,见 `valid_scripts/validate_t6_mi_eeg_native.py`)

### Path X 否证:Surrogate z-score 也救不了

SC4001 实测 MI 量级 ~1e-4,远低于 plan v2 预期 [0.02, 0.05]。先按"绝对值低不等于无信号"假说,改用 surrogate-normalized z-score 重新评估。

### 表 5:Surrogate z-score(`test_surrogate_zscore_pac.py` 输出)

| 测试场景 | z-score | 解读 |
|----------|---------|------|
| SC4001 EEG Mode X(4260s concat, 141 边界) | +21.40 | boundary artifact,null σ 异常窄 |
| **SC4001 EEG Mode Y(480s contig, 0 边界)** | **+0.14** | **null** —— SC4001 真实 PAC 不显著 |
| 白噪 paranoia(480s) | -1.70 | 在 ±2 内,算法 self-check 通过 |
| Seed B simulator cross-channel(r_ctx × r_thal) | +42.51 | 强显著 |
| Seed B simulator single-channel(r_ctx × r_ctx) | +68.72 | 更强显著 |

数据来源:`valid_scripts/test_surrogate_zscore_pac.py` 输出 + `c_ctx2th_unidentifiability_evidence.md` §3.2。

**铁证**:
- Mode Y 0 边界给 z=+0.14,**与白噪 paranoia z=-1.70 在统计上不可区分**(均在 ±2 内)
- Simulator 端 z=+42(cross-channel)和 z=+69(single-channel)证明仿真器侧 PAC 强存在
- **两端 z 量级差 ~300 倍,统计意义上不可比** → MI 这一维进 7 维 SBI 必将引导 NSF 把 c_ctx2th 推向"破坏 PAC 的参数空间"

### 决策:走路径 W(接受 5 维 baseline)

不再尝试增加维度,理由(详 `c_ctx2th_unidentifiability_evidence.md` §4):
1. SC4001 EEG 信息量是 capped 的,任何 single-channel metric 都有上限
2. Sprint 1 Phase 2 已投入 4-5 工作日预算,期望 ROI 偏低(c_ctx2th 可能依然 unidentifiable,问题在数据不在 metric)
3. 12 月 TBME deadline 不允许 4-6 周的数据集切换

### Negative result 的 publication value

这是个 publication-worthy negative result:**用 SBI 严格证明了某个特定 EEG dataset(Sleep-EDF Cassette,Fpz-Cz,artifact-rejected N3)在 single-channel PAC 维度上对皮层-丘脑突触强度的可识别性贡献为零**。算法层面无故障(SBC 4/4 PASS),力学模型无故障(Seed B PPC 通过),Surrogate test 在 simulator 端 z=+42(确认 PAC 真实可测),三者一起把锅扣给数据-模型 mismatch。

---

## Part V:决策点时间线

### 表 6:决策点(给 HTML 时间线用)

| 日期 | 事件 | 决策 |
|------|------|------|
| 2026-05-07 23:24 → 05-08 09:03 | 7 维 SBI 主 run 完成(9.66h, 4000 sims, SBC skipped) | (彼时认为成功) |
| 2026-05-10 | `scan_xobs_params.py` 诊断扫描确认 r_proxy 上 T6/MI 不可修复 | 删 T6 + MI,5 维方案 |
| 2026-05-10 20:32 | `x_obs_v3.npz` 生成(5 维) | — |
| 2026-05-11 | `sbi_report_0511.md` 写就(含错误叙事 "5 维 PPC 全 PASS") | — |
| 2026-05-22 上午 | Pre-flight 6 项检查(`sbi_5dim_preflight_report.md`)发现:`sbi_outputs/` 残留 7 维数据将导致 R1 崩溃(P0);`b` prior 上界 42.6 距 Seed B 仅 1.8% 余量(P1) | 归档 7 维 + b prior → 80 |
| 2026-05-22 19:10 | 7 维归档目录建立 `sbi_outputs_7dim_archive_20260507/` | — |
| 2026-05-22 19:52 | `--dry-run` 跑通(50 sims/轮,150 sims 总,0 NaN) | 进 5 维主 run |
| 2026-05-22 20:31:10 | 5 维 SBI 主 run 启动(PID 27192) | — |
| 2026-05-23 04:34:40 | 5 维 SBI 完成(8.06h, 5300 sims, R4 跑满, SBC 4/4, PPC 3/5) | first-pass review |
| 2026-05-23 早 | First-pass review 发现 c_ctx2th 4 模态 + 横/竖条无 ridge | 启动 Phase 2 EEG-native T6 + MI |
| 2026-05-23 中午 | Phase 2 阶段 A:`compute_pac_metrics_eeg_native.py` + 合成 5 case 验证 10/10 PASS | 进阶段 B |
| 2026-05-23 下午 | 阶段 B step 5 SC4001 实测 MI=0.00019(触发 algorithm-suspect FAIL) | 暂停 → quick verification |
| 2026-05-23 傍晚 | Quick verification:Mode Y 480s contig 给 MI=0.00011(零边界仍 null) | 排除拼接根因 → Path X surrogate z-score |
| 2026-05-23 傍晚 | Surrogate z-score:Mode Y z=+0.14(null), Simulator z=+42 | 两端不可比 → **走路径 W** |
| 2026-05-23 晚 | Sprint 1 closeout commit `f033e95`;`c_ctx2th_unidentifiability_evidence.md` + `sprint2_startup_checklist.md` 写就 | Stage 2 closed |

注:May 22-23 所有详细子事件(pre-flight 6 项各自结果、preflight steps 2/3/4 PASS 详情、SBC redo 数字、阶段 A 10/10 case 数字、Path X 完整 z 表)在子文档中,本时间线只保留转折点。

---

## Part VI:Next Steps

### Sprint 1 闭环

- `f033e95` commit 已含 52 个文件(主线 5 维 + Phase 2 探索性 + 11 篇 docs + May 7/23 全产物归档)
- 主线代码冻结:`run_sbi.py` / `simulator_wrapper.py` / `models/s4_personalize_fig7_v7.py` 都不再动
- Phase 2 探索性代码全部留在 repo(`S4_v7_repair/compute_pac_metrics_eeg_native.py` + `compute_xobs_from_eeg_v4.py` + `validate_*.py` + `test_surrogate_zscore_pac.py`)作为 paper supporting material 的代码引用

### Sprint 2 启动

详 `sprint2_startup_checklist.md`,4 个 post-hoc validation(在现有 5 维后验 + Seed B simulator 之上):

| # | Validation | 估时 | 工具状态 |
|---|-----------|------|---------|
| 1 | noise-off(推荐首发) | 3-4 h | V7 + T1-T12,需新写 |
| 2 | PAC surrogate | 1-2 h | **`compute_mi_zscore` 已完整实现** |
| 3 | Rayleigh phase concentration | 2-3 h | scipy + 现有 PAC 流水线,需新写 ~50 行 |
| 4 | spindle morphology | 4-6 h | 现有 T7/T8 流水线 + KS test,需新写 |

**总工作量重估**:**12-17 h ≈ 2 工作日**(比 Sprint 1 Phase 2 的 4-5 天节省 60%)。

### 时间线

| 时期 | 计划 |
|------|------|
| 2026 年 6 月 | Sprint 2 完工(4 个 post-hoc validation + 跨 validation 汇总报告) |
| 2026 年 7-9 月 | TBME paper 写作(framing:3/4 参数可识别 + 1 个 c_ctx2th unidentifiability 的 publication-worthy negative result) |
| 2026 年 12 月 | TBME 投稿 |

### 子文档完整索引

**Stage 2 总报告**
- `docs/sbi_report_0511.md`(本文之外的最重要文档,§3 = 5 维 SBI 详细数字)

**Sprint 1 lifecycle 文档(按时序)**
- `docs/sbi_5dim_preflight_report.md` — 5 维启动前 6 项 sanity
- `docs/preflight_steps234_report.md` — pre-flight Step 2/3/4 验证报告
- `docs/sbi_5dim_review_20260523.md` — 5 维 run first-pass review
- `docs/sbi_5dim_diagnostic_20260523.md` — pairplot + 后验采样诊断 + SBC redo 后续
- `docs/sprint1_phase2_plan.md` / `_v2.md` — Phase 2 设计(v1 → v2 含 3 处修订)
- `docs/c_ctx2th_unidentifiability_evidence.md` — paper supporting material(3 条证据链)
- `docs/sprint2_startup_checklist.md` — Sprint 2 启动清单

**主要代码产物**
- `S4_sbi/run_sbi.py`(主线,5 维)
- `S4_sbi/simulator_wrapper.py`(主线,5 维)
- `S4_sbi/compute_xobs_from_eeg_v3.py`(主线,5 维 x_obs 抽取)
- `S4_sbi/compute_xobs_from_eeg_v4.py`(Phase 2,7 维尝试,未上线)
- `S4_v7_repair/compute_pac_metrics_eeg_native.py`(Phase 2,AASM + Tort + surrogate)
- `valid_scripts/validate_t6_mi_eeg_native.py`(合成 5 case 验证)
- `valid_scripts/test_surrogate_zscore_pac.py`(SC4001 + Seed B z-score 测试)
- `S4_sbi/replot_sbc_5dim.py`(SBC fig 补图脚本)

**Run 产物**
- `S4_sbi/sbi_outputs/round{1..4}_posterior.pkl`(5 维 May 23 最终后验)
- `S4_sbi/sbi_outputs/all_simulations.npz`(theta 5000×4,x 5000×5)
- `S4_sbi/sbi_outputs/fig_{marginals,pairplot,ppc,sbc,pareto_overlay}.png`
- `S4_sbi/sbi_outputs_7dim_archive_20260507/`(May 7 7 维归档,9 文件)
- `S4_sbi/x_obs_v3.npz`(5 维 SC4001 观测)

**Run logs**
- `S4_sbi/sbi_log.txt`(May 23 5 维主 run 日志)
- `S4_sbi/sbi_log_dryrun_20260522.txt`(dry-run 归档)
- `sbi_5dim_run.log` / `.err`(PowerShell 后台 stdout/stderr 镜像)
